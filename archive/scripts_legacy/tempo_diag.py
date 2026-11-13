"""Resolve tempo octave ambiguity for the Ajja corpus.

The pipeline's get_native_bpm() can return 2x the true tempo for slow tracks,
and a naive autocorrelation peak-pick can return 0.5x the true tempo for fast
tracks (the 2-beat peak is usually taller than the 1-beat peak in 4/4 music).

Strategy that survives both biases: score every candidate lag by its
autocorrelation, then among local maxima within 85% of the best score choose
the FASTEST tempo. A genuinely slow track has no onset energy at half its beat
period, so its 2x candidate scores poorly and is not selected; a fast track's
half-tempo candidate is strong but loses to the faster one by the tie-break.

Writes ajja_tempo.json.
"""

import json
import os
import sys
import typing
from concurrent.futures import ProcessPoolExecutor, as_completed

typing.cast(typing.Any, sys.stdout).reconfigure(
    line_buffering=True, encoding="utf-8", errors="replace"
)

MUSIC = os.path.join(os.path.expanduser("~"), "Music")


def pos_int(value, fallback):
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return n if n > 0 else fallback


def tempo_candidates(y, sr, lo=40.0, hi=150.0, bpm_lo=60.0, bpm_hi=200.0):
    """Return (chosen_bpm, [(bpm, score), ...] top peaks) from kick-band ACF."""
    import numpy as np
    from scipy.signal import butter, lfilter, sosfilt

    try:
        nyq = float(sr) / 2.0
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None, []
    if nyq <= 0:
        return None, []

    sos = butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype="band", output="sos")
    x = np.asarray(sosfilt(sos, y), dtype=np.float64)

    env = np.abs(x)
    w = pos_int(sr * 0.004, 1)
    env = lfilter(np.ones(w) / w, [1.0], env)
    env = np.diff(env)
    env[env < 0] = 0.0

    dec = pos_int(sr / 400.0, 1)
    env = env[::dec]
    try:
        fs = float(sr) / dec
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None, []
    if len(env) < pos_int(fs * 10, 1):
        return None, []

    env = env - env.mean()
    acf = np.correlate(env, env, mode="full")[len(env) - 1 :]
    if acf[0] <= 0:
        return None, []
    acf = acf / acf[0]

    lag_lo = pos_int(fs * 60.0 / bpm_hi, 2)
    lag_hi = pos_int(min(len(acf) - 2, fs * 60.0 / bpm_lo), 2)
    if lag_hi <= lag_lo + 2:
        return None, []

    # local maxima of the ACF inside the window
    peaks = []
    for lag in range(lag_lo, lag_hi):
        v = acf[lag]
        if v >= acf[lag - 1] and v >= acf[lag + 1] and v > 0.02:
            try:
                peaks.append((60.0 * fs / lag, float(v)))
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                continue
    if not peaks:
        return None, []

    peaks.sort(key=lambda p: -p[1])
    best = peaks[0][1]
    # among near-best peaks, prefer the fastest tempo (smallest lag)
    strong = [p for p in peaks if p[1] >= 0.85 * best]
    strong.sort(key=lambda p: -p[0])
    return strong[0][0], peaks[:5]


def analyse(path):
    import numpy as np
    import soundfile as sf
    from autodj.analysis import get_native_bpm

    try:
        info = sf.info(path)
        sr = info.samplerate
        frames = min(info.frames, pos_int(sr * 120, 1))
        y, sr = sf.read(path, frames=frames, dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) == 2:
            y = np.mean(y, axis=1)

        native, _, _ = get_native_bpm(y, sr)
        chosen, peaks = tempo_candidates(y, sr)
        return {
            "path": path,
            "native": float(native),
            "fundamental": float(chosen) if chosen else None,
            "peaks": peaks,
            "dur": info.frames / info.samplerate,
            "err": None,
        }
    except Exception as e:  # noqa: BLE001 - report and continue
        return {
            "path": path,
            "native": None,
            "fundamental": None,
            "peaks": [],
            "dur": None,
            "err": f"{type(e).__name__}: {e}",
        }


def main():
    try:
        with open("ajja_modern_scan.json", encoding="utf-8") as fh:
            prev = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read ajja_modern_scan.json: {exc}")
        return 1

    paths = [r["path"] for r in prev["all"] if r.get("bpm") and 60 < r["bpm"] < 220]
    print(f"analysing {len(paths)} tracks")

    out = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        futs = {ex.submit(analyse, p): p for p in paths}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            n = r.get("native") or 0.0
            fn = r.get("fundamental") or 0.0
            mark = ""
            if n and fn:
                ratio = n / fn
                if 1.8 < ratio < 2.2:
                    mark = "  native=2x"
                elif 0.45 < ratio < 0.55:
                    mark = "  native=0.5x"
                elif abs(n - fn) > 3:
                    mark = "  differs"
            print(
                f"  [{i:3d}] native={n:6.1f}  fundamental={fn:6.1f}{mark:<12} "
                f"{os.path.basename(r['path'])[:48]}"
            )

    try:
        with open("ajja_tempo.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    except OSError as exc:
        print(f"warning: could not write ajja_tempo.json: {exc}")

    print()
    print("=== candidates for the ambiguous ones ===")
    for r in out:
        if r.get("native") and r.get("fundamental"):
            ratio = r["native"] / r["fundamental"]
            if not (0.9 < ratio < 1.1):
                tops = ", ".join(f"{b:.1f}({s:.2f})" for b, s in r["peaks"][:4])
                print(
                    f"  native={r['native']:6.1f}  fund={r['fundamental']:6.1f}  "
                    f"top ACF peaks: {tops}"
                )
                print(f"      {os.path.relpath(r['path'], MUSIC)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
