#!/usr/bin/env python3
"""
doppler_locate.py — find where (and when) a video was filmed from the Doppler
curves of aircraft flying overhead.

Steps:
  1. spectro : video audio -> spectrogram (look for the S-shaped curve)
  2. fit     : track the curve and fit the Doppler model
               -> time of closest approach (CPA), speed, distance at CPA,
                  engine tone
  3. locate  : compare the curves with public ADS-B tracks (OpenSky,
               ADS-B Exchange...) and search for the observer position that
               best explains them. Two aircraft = one precise point.
  4. demo    : generate a synthetic recording (2 propeller planes, 920 s apart)
               plus ADS-B tracks, then run the whole pipeline to check it.

Requirements: numpy, scipy, matplotlib, ffmpeg on PATH.

Examples:
  python doppler_locate.py spectro video.mp4 --fmin 50 --fmax 400
  python doppler_locate.py fit video.mp4 --fmin 60 --fmax 220 --t0 0 --t1 300 -o plane1.json
  python doppler_locate.py locate plane1.json plane2.json --tracks adsb.csv --tol 120
  python doppler_locate.py demo
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

C_SOUND = 343.0          # m/s, speed of sound (~20 °C)
EARTH_R = 6_371_000.0    # m


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def load_audio(path: str, sr: int = 4000) -> tuple[np.ndarray, int]:
    """Decode any audio/video file to mono float32 through ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy(), sr


def creation_time(path: str) -> float | None:
    """Recording time (unix) read from the file metadata, if present."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=True).stdout.strip()
        if out:
            return dt.datetime.fromisoformat(out.replace("Z", "+00:00")).timestamp()
    except Exception:
        pass
    return None


def parse_time(s: str) -> float:
    """'2026-11-17T18:18:34Z' or a unix timestamp -> unix."""
    try:
        return float(s)
    except ValueError:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t.timestamp()


def fmt_time(unix: float) -> str:
    return dt.datetime.fromtimestamp(unix, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# --------------------------------------------------------------------------- #
# Spectrogram
# --------------------------------------------------------------------------- #
def spectrogram(x, sr, fmin, fmax, win=1.0, hop=0.25):
    """Spectrogram in dB plus a whitened version (z-score per frequency bin).

    Whitening removes stationary noise (fans, 50/60 Hz mains hum), which shows
    up as horizontal lines. What is left is what moves: the aircraft.
    """
    from scipy.signal import spectrogram as sp
    nper = int(win * sr)
    f, t, S = sp(x, fs=sr, window="hann", nperseg=nper, noverlap=nper - int(hop * sr),
                 nfft=max(nper, int(sr / 0.5)), scaling="spectrum", mode="psd")
    band = (f >= fmin) & (f <= fmax)
    f, S = f[band], S[band]
    db = 10 * np.log10(S + 1e-20)
    med = np.median(db, axis=1, keepdims=True)
    mad = np.median(np.abs(db - med), axis=1, keepdims=True) * 1.4826 + 1e-6
    z = (db - med) / mad
    return f, t, db, z


def track_ridge(z, freqs, max_jump_hz=1.5, jump_cost=0.6):
    """Follow the strongest, most continuous curve (Viterbi).

    Returns, for each frame, the frequency (with parabolic interpolation) and
    the signal strength (z-score) along the curve.
    """
    F, T = z.shape
    df = freqs[1] - freqs[0]
    J = max(1, int(round(max_jump_hz / df)))          # max jump between two frames
    zc = np.clip(z, -2, 8)
    score = zc[:, 0].copy()
    back = np.zeros((F, T), dtype=np.int32)
    offs = np.arange(-J, J + 1)
    idx = np.arange(F)
    for k in range(1, T):
        cand = np.full((len(offs), F), -np.inf)
        for i, o in enumerate(offs):
            sh = np.roll(score, o)
            if o > 0:
                sh[:o] = -np.inf
            elif o < 0:
                sh[o:] = -np.inf
            cand[i] = sh - jump_cost * abs(o)
        best = cand.argmax(axis=0)
        back[:, k] = idx - offs[best]
        score = cand[best, idx] + zc[:, k]
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(score.argmax())
    for k in range(T - 1, 0, -1):
        path[k - 1] = back[path[k], k]
    # parabolic interpolation around the peak to beat the FFT resolution
    fr = freqs[path].astype(float)
    a = z[np.clip(path - 1, 0, F - 1), np.arange(T)]
    b = z[path, np.arange(T)]
    c = z[np.clip(path + 1, 0, F - 1), np.arange(T)]
    den = a - 2 * b + c
    delta = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
    fr += np.clip(delta, -0.5, 0.5) * df
    return fr, b


# --------------------------------------------------------------------------- #
# Doppler model (aircraft on a straight line at constant speed)
# --------------------------------------------------------------------------- #
def doppler_model(t, t0, f0, v, d, c=C_SOUND):
    """Frequency heard at time t.

    t0 : time when the aircraft is at its closest point (CPA); it is heard
         d/c later (the propagation delay r/c is handled exactly below).
    f0 : emitted frequency (engine / propeller), Hz
    v  : aircraft speed, m/s
    d  : minimum aircraft-observer distance (slant range), m

    Sound received at t was emitted at tau = t - r(tau)/c. With u = tau - t0
    and T = t - t0: c²(T-u)² = d² + v²u²  -> a quadratic in u.
    """
    T = t - t0
    a = c * c - v * v
    disc = c * c * v * v * T * T + a * d * d
    u = (c * c * T - np.sqrt(disc)) / a
    r = np.sqrt(d * d + v * v * u * u)
    vr = v * v * u / r                      # radial speed (>0 = moving away)
    return f0 * c / (c + vr)


def fit_doppler(t, f, w):
    from scipy.optimize import least_squares
    keep = w > 0
    t, f, w = t[keep], f[keep], w[keep]
    sw = np.sqrt(w / w.max())
    fhi, flo = np.percentile(f, 95), np.percentile(f, 5)
    R = max(fhi / max(flo, 1e-3), 1.001)
    v_guess = float(np.clip(C_SOUND * (R - 1) / (R + 1), 20, 280))
    f0_guess = 2 * fhi * flo / (fhi + flo)
    mid = np.argmin(np.abs(f - f0_guess))
    best = None
    for t0g in (t[mid], np.median(t), t[len(t) // 4], t[3 * len(t) // 4]):
        for dg in (300, 1000, 3000, 8000):
            p0 = [t0g, f0_guess, v_guess, dg]
            lo = [t.min() - 120, flo * 0.7, 15, 30]
            hi = [t.max() + 120, fhi * 1.3, 320, 30000]
            p0 = list(np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6))
            res = least_squares(lambda p: sw * (doppler_model(t, *p) - f), p0,
                                bounds=(lo, hi), loss="soft_l1", f_scale=1.5)
            if best is None or res.cost < best.cost:
                best = res
    t0, f0, v, d = best.x
    resid = f - doppler_model(t, *best.x)
    return dict(t0=float(t0), f0=float(f0), v=float(v), d=float(d),
                rms_hz=float(np.sqrt(np.mean(resid ** 2))))


# --------------------------------------------------------------------------- #
# Command: spectro
# --------------------------------------------------------------------------- #
def cmd_spectro(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, sr = load_audio(a.input, a.sr)
    f, t, db, z = spectrogram(x, sr, a.fmin, a.fmax, a.win, a.hop)
    out = a.out or str(Path(a.input).with_suffix("")) + "_spectro.png"
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.pcolormesh(t, f, np.clip(z, -2, 6), cmap="magma", shading="auto")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"{Path(a.input).name} — whitened spectrogram ({a.fmin:g}-{a.fmax:g} Hz)\n"
                 "Look for a falling S-shaped curve: that is an aircraft flying by.")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"spectrogram -> {out}  ({len(x) / sr:.0f} s of audio)")


# --------------------------------------------------------------------------- #
# Command: fit
# --------------------------------------------------------------------------- #
def run_fit(path, fmin, fmax, t_start=None, t_end=None, sr=4000, win=1.0, hop=0.25,
            start=None, plot=None, min_z=2.0):
    x, sr = load_audio(path, sr)
    i0 = int((t_start or 0) * sr)
    i1 = int(t_end * sr) if t_end else len(x)
    f, t, db, z = spectrogram(x[i0:i1], sr, fmin, fmax, win, hop)
    t = t + (t_start or 0)
    ridge, strength = track_ridge(z, f)
    w = np.clip(strength - min_z, 0, None)
    if (w > 0).sum() < 10:
        raise SystemExit("No clear enough curve in this window: widen the band or change "
                         "--t0/--t1 (check the spectrogram first).")
    p = fit_doppler(t, ridge, w)
    if start is None:
        start = creation_time(path)
    res = dict(source=str(path), start_unix=start, fmin=fmin, fmax=fmax,
               t=t[w > 0].round(3).tolist(), f=ridge[w > 0].round(3).tolist(),
               w=w[w > 0].round(3).tolist(), fit=p)
    v, d = p["v"], p["d"]
    print(f"--- {Path(path).name} [{t[0]:.0f}-{t[-1]:.0f} s]")
    print(f"  closest approach (CPA) : t = {p['t0']:.1f} s"
          + (f"  ->  {fmt_time(start + p['t0'])}" if start else ""))
    print(f"  engine tone            : {p['f0']:.1f} Hz")
    print(f"  aircraft speed         : {v:.0f} m/s ({v * 3.6:.0f} km/h, {v * 1.944:.0f} kt)")
    print(f"  distance at CPA        : {d:.0f} m  (= height above you if it flies straight overhead)")
    print(f"  fit RMS error          : {p['rms_hz']:.2f} Hz")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.pcolormesh(t, f, np.clip(z, -2, 6), cmap="Blues_r", shading="auto")
        ax.plot(t[w > 0], ridge[w > 0], ".", ms=2, color="white", alpha=.6, label="tracked curve")
        tt = np.linspace(t[0], t[-1], 600)
        ax.plot(tt, doppler_model(tt, p["t0"], p["f0"], p["v"], p["d"]), color="crimson", lw=2,
                label="fitted Doppler model")
        ax.axvline(p["t0"], color="crimson", ls="--", lw=1)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frequency (Hz)")
        ax.set_title(f"CPA t={p['t0']:.1f} s · f0={p['f0']:.1f} Hz · v={v * 3.6:.0f} km/h · "
                     f"d={d:.0f} m")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(plot, dpi=130)
        print(f"  plot -> {plot}")
    return res


def cmd_fit(a):
    start = parse_time(a.start) if a.start else None
    out = a.out or str(Path(a.input).with_suffix("")) + "_doppler.json"
    res = run_fit(a.input, a.fmin, a.fmax, a.t0, a.t1, a.sr, a.win, a.hop, start,
                  plot=str(Path(out).with_suffix(".png")), min_z=a.min_z)
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"  curve -> {out}")
    if res["start_unix"] is None:
        print("  (no recording time in the metadata: pass --start to locate)")


# --------------------------------------------------------------------------- #
# ADS-B tracks
# --------------------------------------------------------------------------- #
ID_COLS = ("flight", "icao24", "callsign", "hex", "id")
TIME_COLS = ("time", "timestamp", "ts", "unix")
ALT_M_COLS = ("alt_m", "geoaltitude", "baroaltitude", "altitude_m")
ALT_FT_COLS = ("alt_ft", "alt_baro", "alt_geom", "altitude_ft", "altitude")


def load_tracks(path):
    """ADS-B CSV, one row per position. Accepted columns:
    id (flight/icao24/callsign), time (unix or ISO), lat, lon,
    alt_m (or geoaltitude/baroaltitude, OpenSky) or alt_ft (ADS-B Exchange)."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"{path} is empty")
    cols = {k.lower().strip(): k for k in rows[0].keys()}
    pick = lambda opts: next((cols[o] for o in opts if o in cols), None)
    cid, ct = pick(ID_COLS), pick(TIME_COLS)
    clat, clon = pick(("lat", "latitude")), pick(("lon", "lng", "longitude"))
    calm, calf = pick(ALT_M_COLS), pick(ALT_FT_COLS)
    if not (cid and ct and clat and clon and (calm or calf)):
        raise SystemExit(f"unrecognized columns in {path}: {list(rows[0].keys())}")
    tracks = {}
    for r in rows:
        try:
            alt = float(r[calm]) if calm else float(r[calf]) * 0.3048
            p = (parse_time(r[ct]), float(r[clat]), float(r[clon]), alt)
        except (ValueError, TypeError):
            continue
        tracks.setdefault(r[cid].strip(), []).append(p)
    out = {}
    for k, pts in tracks.items():
        a = np.array(sorted(pts))
        _, u = np.unique(a[:, 0], return_index=True)
        if len(u) >= 2:
            out[k] = a[u]
    return out


class Proj:
    """Local flat (equirectangular) projection in meters around a center."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.k = math.cos(math.radians(lat0))

    def xy(self, lat, lon):
        return (np.radians(np.asarray(lon) - self.lon0) * EARTH_R * self.k,
                np.radians(np.asarray(lat) - self.lat0) * EARTH_R)

    def latlon(self, x, y):
        return (self.lat0 + np.degrees(np.asarray(y) / EARTH_R),
                self.lon0 + np.degrees(np.asarray(x) / (EARTH_R * self.k)))


def predict_g(tr, ox, oy, oz, t_abs, c=C_SOUND):
    """Predicted Doppler factor c/(c+v_r) for one track and many observers.

    tr : (N,4) time, x, y, z ; ox/oy : (G,1) ; t_abs : (M,)  -> (G,M)
    Points outside the track -> NaN.
    """
    tt, X, Y, Z = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
    tau = np.broadcast_to(t_abs, (ox.shape[0], t_abs.size)).astype(float)
    for _ in range(3):                                     # emission time
        px, py, pz = np.interp(tau, tt, X), np.interp(tau, tt, Y), np.interp(tau, tt, Z)
        r = np.sqrt((px - ox) ** 2 + (py - oy) ** 2 + (pz - oz) ** 2)
        tau = t_abs - r / c
    h = 1.0
    vx = (np.interp(tau + h, tt, X) - np.interp(tau - h, tt, X)) / (2 * h)
    vy = (np.interp(tau + h, tt, Y) - np.interp(tau - h, tt, Y)) / (2 * h)
    vz = (np.interp(tau + h, tt, Z) - np.interp(tau - h, tt, Z)) / (2 * h)
    vr = ((px - ox) * vx + (py - oy) * vy + (pz - oz) * vz) / np.maximum(r, 1.0)
    g = c / (c + vr)
    g[(tau < tt[0]) | (tau > tt[-1])] = np.nan
    return g


def curve_cost(curve, tracks_xy, ox, oy, oz, shift):
    """For each observer: RMS error (Hz) of the best-matching track."""
    t_abs = curve["start_unix"] + curve["t"] + shift
    f, w = curve["f"], curve["w"]
    best = np.full(ox.shape[0], np.inf)
    who = np.full(ox.shape[0], -1)
    for k, tr in enumerate(tracks_xy):
        if t_abs[-1] < tr[0, 0] or t_abs[0] > tr[-1, 0]:
            continue
        g = predict_g(tr, ox, oy, oz, t_abs)
        ok = np.isfinite(g)
        cover = (ok * w).sum(1) / w.sum()
        gg = np.where(ok, g, 0.0)
        f0 = (w * f * gg).sum(1) / np.maximum((w * gg * gg).sum(1), 1e-12)
        err = np.sqrt((w * ok * (f - f0[:, None] * gg) ** 2).sum(1) / np.maximum((w * ok).sum(1), 1e-12))
        err[cover < 0.8] = np.inf
        better = err < best
        best[better], who[better] = err[better], k
    return best, who


def load_curve(path, start_override=None, max_pts=120):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    t, f, w = (np.array(c[k], float) for k in ("t", "f", "w"))
    if len(t) > max_pts:
        i = np.linspace(0, len(t) - 1, max_pts).round().astype(int)
        t, f, w = t[i], f[i], w[i]
    start = start_override if start_override is not None else c.get("start_unix")
    if start is None:
        raise SystemExit(f"{path}: unknown recording time -> add --start")
    return dict(name=Path(path).stem, t=t, f=f, w=w, start_unix=float(start), fit=c["fit"])


def run_locate(curve_paths, tracks_path, start=None, tol=0.0, step=5.0, elev=0.0,
               grid=120, margin_km=5.0, plot=None, truth=None):
    curves = [load_curve(p, start) for p in curve_paths]
    raw = load_tracks(tracks_path)
    # keep the tracks that are active around the recordings
    tmin = min(c["start_unix"] + c["t"][0] for c in curves) - tol - 120
    tmax = max(c["start_unix"] + c["t"][-1] for c in curves) + tol + 120
    raw = {k: v[(v[:, 0] >= tmin - 300) & (v[:, 0] <= tmax + 300)] for k, v in raw.items()}
    raw = {k: v for k, v in raw.items() if len(v) >= 2 and v[0, 0] <= tmax and v[-1, 0] >= tmin}
    if not raw:
        raise SystemExit("no ADS-B track inside the recording time window")
    allp = np.vstack(list(raw.values()))
    proj = Proj(float(np.median(allp[:, 1])), float(np.median(allp[:, 2])))
    names = list(raw)
    tracks_xy = []
    for k in names:
        x, y = proj.xy(raw[k][:, 1], raw[k][:, 2])
        tracks_xy.append(np.column_stack([raw[k][:, 0], x, y, raw[k][:, 3]]))
    # search area: where aircraft fly during the recording
    act = np.vstack([tr[(tr[:, 0] >= tmin) & (tr[:, 0] <= tmax)] for tr in tracks_xy if
                     ((tr[:, 0] >= tmin) & (tr[:, 0] <= tmax)).any()])
    m = margin_km * 1000
    x0, x1 = act[:, 1].min() - m, act[:, 1].max() + m
    y0, y1 = act[:, 2].min() - m, act[:, 2].max() + m
    shifts = np.arange(-tol, tol + step / 2, step) if tol > 0 else np.array([0.0])

    def search(xs, ys):
        gx, gy = np.meshgrid(xs, ys)
        ox, oy = gx.reshape(-1, 1), gy.reshape(-1, 1)
        best = (np.full(ox.shape[0], np.inf), None, None)
        for s in shifts:
            tot = np.zeros(ox.shape[0])
            whos = []
            for c in curves:
                e, who = curve_cost(c, tracks_xy, ox, oy, elev, s)
                tot += e ** 2
                whos.append(who)
            tot = np.sqrt(tot / len(curves))
            better = tot < best[0]
            if best[1] is None:
                best = (tot, np.full(tot.shape, s), np.array(whos))
            else:
                best[0][better] = tot[better]
                best[1][better] = s
                best[2][:, better] = np.array(whos)[:, better]
        return gx, gy, best

    print(f"{len(names)} ADS-B tracks, area {(x1 - x0) / 1000:.0f} x {(y1 - y0) / 1000:.0f} km, "
          f"{len(shifts)} clock offset(s) tested")
    xs, ys = np.linspace(x0, x1, grid), np.linspace(y0, y1, grid)
    gx, gy, (cost, sh, who) = search(xs, ys)
    # two successive zooms around the best point
    bx, by = gx.flat[int(np.argmin(cost))], gy.flat[int(np.argmin(cost))]
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    zooms = []
    for _ in range(2):
        fx = np.linspace(bx - 3 * dx, bx + 3 * dx, 41)
        fy = np.linspace(by - 3 * dy, by + 3 * dy, 41)
        FX, FY, (fc, fsh, fwho) = search(fx, fy)
        j = int(np.argmin(fc))
        bx, by = FX.flat[j], FY.flat[j]
        dx, dy = fx[1] - fx[0], fy[1] - fy[0]
        zooms.append((FX, FY, fc))
    FX, FY, fc1 = zooms[0]                             # first zoom, used for the detail map
    lat, lon = proj.latlon(bx, by)
    # uncertainty: area where the error stays < 1.25x the minimum (first zoom)
    good = fc1 < fc1.min() * 1.25
    unc = float(np.sqrt(((FX.ravel()[good] - bx) ** 2 + (FY.ravel()[good] - by) ** 2).max())) if good.sum() > 1 else 0.0
    res = dict(lat=float(lat), lon=float(lon), rms_hz=float(fc[j]), clock_shift_s=float(fsh[j]),
               uncertainty_m=unc,
               planes={c["name"]: names[int(fwho[n, j])] for n, c in enumerate(curves)})
    print("\n=== RESULT ===")
    print(f"  estimated position : {lat:.5f}, {lon:.5f}   (± ~{max(unc, 50):.0f} m)")
    print(f"  https://www.google.com/maps?q={lat:.5f},{lon:.5f}")
    print(f"  RMS error : {fc[j]:.2f} Hz   clock offset : {fsh[j]:+.0f} s")
    for n, c in enumerate(curves):
        print(f"  {c['name']} <-> aircraft {res['planes'][c['name']]}"
              f"   (closest at {fmt_time(c['start_unix'] + c['fit']['t0'] + fsh[j])})")
    if len(curves) == 1:
        print("  Note: with a single aircraft there is a mirror solution on the other side of "
              "its flight path. A second aircraft removes the ambiguity.")
    if truth is not None:
        tx, ty = proj.xy(*truth)
        err = math.hypot(bx - float(tx), by - float(ty))
        res["error_vs_truth_m"] = err
        print(f"  true position : {truth[0]:.5f}, {truth[1]:.5f}  ->  error {err:.0f} m")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
        panels = [(axes[0], gx, gy, cost, (x0, x1, y0, y1), "overview"),
                  (axes[1], FX, FY, fc1, (FX.min(), FX.max(), FY.min(), FY.max()), "zoom")]
        for ax, X, Y, Cst, (a0, a1, b0, b1), title in panels:
            im = ax.pcolormesh(X / 1e3, Y / 1e3, np.minimum(Cst.reshape(X.shape), 30), cmap="viridis_r",
                               shading="auto")
            for k, tr in zip(names, tracks_xy):
                sel = (tr[:, 0] >= tmin) & (tr[:, 0] <= tmax)
                used = k in res["planes"].values()
                ax.plot(tr[sel, 1] / 1e3, tr[sel, 2] / 1e3, "-", color="white" if used else "0.6",
                        lw=2 if used else 1)
                if used and title != "zoom":
                    inside = sel & (tr[:, 1] > a0) & (tr[:, 1] < a1) & (tr[:, 2] > b0) & (tr[:, 2] < b1)
                    if inside.any():
                        q = np.flatnonzero(inside)[len(np.flatnonzero(inside)) // 3]
                        ax.annotate(k, (tr[q, 1] / 1e3, tr[q, 2] / 1e3), color="white", fontsize=9,
                                    xytext=(5, 5), textcoords="offset points")
            ax.plot(bx / 1e3, by / 1e3, "*", ms=18, color="gold", mec="black", label="estimate")
            if truth is not None:
                ax.plot(float(tx) / 1e3, float(ty) / 1e3, "x", ms=11, mew=3, color="red", label="truth")
            ax.set_xlim(a0 / 1e3, a1 / 1e3)
            ax.set_ylim(b0 / 1e3, b1 / 1e3)
            ax.set_aspect("equal")
            ax.set_xlabel("east (km)")
            ax.set_ylabel("north (km)")
            ax.set_title(title)
            ax.legend(loc="lower right")
        fig.colorbar(im, ax=axes, shrink=.8, label="audio vs ADS-B mismatch (Hz) — lower is better")
        fig.suptitle(f"Estimated position: {lat:.5f}, {lon:.5f}  (± ~{max(unc, 50):.0f} m)")
        fig.savefig(plot, dpi=130)
        print(f"  map -> {plot}")
    return res


def cmd_locate(a):
    start = parse_time(a.start) if a.start else None
    out = a.out or "location.png"
    res = run_locate(a.curves, a.tracks, start, a.tol, a.step, a.elev, a.grid, a.margin, plot=out)
    Path(out).with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Demo: synthetic recording + synthetic ADS-B tracks, then the whole pipeline
# --------------------------------------------------------------------------- #
def cmd_demo(a):
    from scipy.io import wavfile
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    sr = 4000
    dur = 1200.0
    start = parse_time("2026-11-17T18:15:00Z")
    obs_lat, obs_lon = 48.7310, 2.2015            # arbitrary point
    proj = Proj(obs_lat, obs_lon)

    # aircraft: (name, f0 Hz, speed m/s, heading °, altitude m, lateral offset m, CPA time s)
    planes = [("F-GKAP", 100.0, 62.0, 75.0, 900.0, 650.0, 180.0),
              ("F-HBEL", 130.0, 70.0, 200.0, 1300.0, -1100.0, 1100.0)]   # 920 s apart
    decoys = [("F-GXYZ", 55.0, 110.0, 1500.0, 9000.0, 400.0),
              ("F-BOOM", 80.0, 340.0, 1100.0, -12000.0, 900.0),
              ("AFR123", 220.0, 20.0, 3500.0, 15000.0, 700.0),
              ("F-GLOP", 58.0, 300.0, 700.0, 6000.0, 1050.0)]

    def state(v, hdg, alt, lateral, tc):
        h = math.radians(hdg)
        ux, uy = math.sin(h), math.cos(h)           # direction (east, north)
        nx, ny = uy, -ux                            # perpendicular
        return lambda tau: (lateral * nx + v * (tau - tc) * ux,
                            lateral * ny + v * (tau - tc) * uy,
                            np.full_like(np.asarray(tau, float), alt))

    t = np.arange(int(dur * sr)) / sr
    x = rng.normal(0, 0.05, t.size).astype(np.float64)
    x += 0.02 * np.sin(2 * np.pi * 50 * t)          # mains hum
    for name, f0, v, hdg, alt, lat_off, tc in planes:
        pos = state(v, hdg, alt, lat_off, tc)
        tau = t.copy()
        for _ in range(4):
            px, py, pz = pos(tau)
            r = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
            tau = t - r / C_SOUND
        amp = 25.0 / r
        x += amp * (np.sin(2 * np.pi * f0 * tau) + 0.3 * np.sin(2 * np.pi * 2 * f0 * tau + 1.0))
    wav = out / "demo_recording.wav"
    wavfile.write(wav, sr, (x / np.abs(x).max() * 0.9).astype(np.float32))

    # ADS-B tracks (one position every 5 s, ~15 m GPS noise)
    csv_path = out / "demo_adsb.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["flight", "time", "lat", "lon", "alt_m"])
        tt = np.arange(-600, dur + 600, 5.0)
        allp = [(n, state(v, h, al, lo, tc)) for n, _, v, h, al, lo, tc in planes]
        allp += [(n, state(v, h, al, lo, tc)) for n, v, h, al, lo, tc in decoys]
        for name, pos in allp:
            px, py, pz = pos(tt)
            px = px + rng.normal(0, 15, tt.size)
            py = py + rng.normal(0, 15, tt.size)
            la, lo = proj.latlon(px, py)
            for i in range(tt.size):
                wr.writerow([name, f"{start + tt[i]:.1f}", f"{la[i]:.6f}", f"{lo[i]:.6f}", f"{pz[i]:.0f}"])
    print(f"synthetic recording -> {wav}\nsynthetic ADS-B tracks -> {csv_path}\n")

    c1 = run_fit(str(wav), 60, 200, 0, 420, start=start, plot=str(out / "demo_plane1.png"))
    c2 = run_fit(str(wav), 60, 200, 880, 1200, start=start, plot=str(out / "demo_plane2.png"))
    for n, (c, p) in enumerate(((c1, planes[0]), (c2, planes[1])), 1):
        dtrue = math.hypot(p[4], p[5])
        print(f"  truth, plane {n}: CPA {p[6]:.0f} s, f0 {p[1]:.0f} Hz, v {p[2]:.0f} m/s, d {dtrue:.0f} m")
        Path(out / f"demo_plane{n}.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
    print(f"  gap between the two CPAs: {c2['fit']['t0'] - c1['fit']['t0']:.0f} s\n")
    run_locate([out / "demo_plane1.json", out / "demo_plane2.json"], csv_path, tol=30, step=5,
               plot=str(out / "demo_location.png"), truth=(obs_lat, obs_lon))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("spectro", help="spectrogram to spot aircraft curves")
    s.add_argument("input")
    s.add_argument("--fmin", type=float, default=40)
    s.add_argument("--fmax", type=float, default=500)
    s.add_argument("--sr", type=int, default=4000)
    s.add_argument("--win", type=float, default=1.0, help="FFT window (s) — 1 s = 1 Hz resolution")
    s.add_argument("--hop", type=float, default=0.25)
    s.add_argument("-o", "--out")
    s.set_defaults(func=cmd_spectro)

    f = sub.add_parser("fit", help="track a Doppler curve and fit it")
    f.add_argument("input")
    f.add_argument("--fmin", type=float, required=True)
    f.add_argument("--fmax", type=float, required=True)
    f.add_argument("--t0", type=float, help="window start (s)")
    f.add_argument("--t1", type=float, help="window end (s)")
    f.add_argument("--start", help="recording start time (ISO UTC or unix) if missing from the metadata")
    f.add_argument("--sr", type=int, default=4000)
    f.add_argument("--win", type=float, default=1.0)
    f.add_argument("--hop", type=float, default=0.25)
    f.add_argument("--min-z", type=float, default=2.0, help="curve sharpness threshold")
    f.add_argument("-o", "--out")
    f.set_defaults(func=cmd_fit)

    l = sub.add_parser("locate", help="find the position using ADS-B tracks")
    l.add_argument("curves", nargs="+", help="JSON files produced by fit")
    l.add_argument("--tracks", required=True, help="ADS-B CSV (OpenSky / ADS-B Exchange)")
    l.add_argument("--start", help="override the recording start time")
    l.add_argument("--tol", type=float, default=60, help="clock uncertainty (s), swept")
    l.add_argument("--step", type=float, default=5)
    l.add_argument("--elev", type=float, default=0, help="observer ground elevation (m)")
    l.add_argument("--grid", type=int, default=120)
    l.add_argument("--margin", type=float, default=5, help="margin around the tracks (km)")
    l.add_argument("-o", "--out")
    l.set_defaults(func=cmd_locate)

    d = sub.add_parser("demo", help="end-to-end test on synthetic data")
    d.add_argument("--dir", default="demo_out")
    d.set_defaults(func=cmd_demo)

    try:
        sys.stdout.reconfigure(encoding="utf-8")       # non-ASCII output on Windows consoles
    except Exception:
        pass
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
