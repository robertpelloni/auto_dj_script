"""Independent tempo check via kick-band autocorrelation.

The main pipeline's get_native_bpm() sometimes locks onto the *double* of a
slow track's real tempo (a classic octave error): the downtempo albums
(Chillsome, Spira Mirabilis) came back at 150-199 BPM, which is really
75-99 BPM doubled.

Psytrance has an extremely strong, regular kick. Band-passing to the kick
band (40-150 Hz) and autocorrelating the onset envelope gives a very direct
tempo estimate, so we can cross-check and decide the correct octave.
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
AUDIO = (".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aiff", ".aif")


def pos_int(value, fallback):
    """int(value) guarded against inf/nan/None/garbage."""
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return n if n > 0 else fallback


def kick_bpm(y, sr, lo=40.0, hi=150.0, bpm_lo=60.0, bpm_hi=200.0):
    """Estimate tempo from the kick band by envelope autocorrelation."""
    import numpy as np
    from scipy.signal import butter, lfilter, sosfilt

    try:
        nyq = float(sr) / 2.0
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    if nyq <= 0:
        return None

    sos = butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype="band", output="sos")
    x = np.asarray(sosfilt(sos, y), dtype=np.float64)

    # onset strength envelope: rectify, smooth ~4 ms, then difference
    env = np.abs(x)
    w = pos_int(sr * 0.004, 1)
    env = lfilter(np.ones(w) / w, [1.0], env)
    env = np.diff(env)
    env[env < 0] = 0.0

    # decimate to ~400 Hz so the ACF is cheap and stable
    dec = pos_int(sr / 400.0, 1)
    env = env[::dec]
    try:
        fs = float(sr) / dec
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    if len(env) < pos_int(fs * 8, 1):
        return None

    env = env - env.mean()
    acf = np.correlate(env, env, mode="full")[len(env) - 1 :]
    if acf[0] <= 0:
        return None
    acf = acf / acf[0]

    # ACF peak inside the plausible beat-period window
    lag_lo = pos_int(fs * 60.0 / bpm_hi, 1)
    lag_hi = pos_int(min(len(acf) - 1, fs * 60.0 / bpm_lo), 1)
    if lag_hi <= lag_lo:
        return None
    seg = acf[lag_lo:lag_hi]
    if len(seg) == 0:
        return None
    lag = lag_lo + pos_int(np.argmax(seg), 0)
    if lag <= 0:
        return None
    return 60.0 * fs / lag


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
        kick = kick_bpm(y, sr)
        return {
            "path": path,
            "native": float(native),
            "kick": float(kick) if kick else None,
            "dur": info.frames / info.samplerate,
            "err": None,
        }
    except Exception as e:  # noqa: BLE001 - report and continue
        return {
            "path": path,
            "native": None,
            "kick": None,
            "dur": None,
            "err": f"{type(e).__name__}: {e}",
        }


def load_previous():
    try:
        with open("ajja_modern_scan.json", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read ajja_modern_scan.json: {exc}")
        return None


def main():
    prev = load_previous()
    if prev is None:
        return 1

    paths = [r["path"] for r in prev["all"] if r.get("bpm") and 60 < r["bpm"] < 220]
    print(f"verifying {len(paths)} tracks")

    out = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        futs = {ex.submit(analyse, p): p for p in paths}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            if r.get("native") and r.get("kick"):
                n, k = r["native"], r["kick"]
                ratio = n / k
                flag = ""
                if 1.8 < ratio < 2.2:
                    flag = "  <-- native = 2x kick (octave error?)"
                elif 0.45 < ratio < 0.55:
                    flag = "  <-- native = 0.5x kick"
                print(
                    f"  [{i:3d}] native={n:6.1f}  kick={k:6.1f}{flag}  "
                    f"{os.path.basename(r['path'])[:50]}"
                )
            else:
                print(
                    f"  [{i:3d}] {os.path.basename(r['path'])[:60]}  "
                    f"(native={r.get('native')}, kick={r.get('kick')})"
                )

    try:
        with open("ajja_bpm_verify.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    except OSError as exc:
        print(f"warning: could not write ajja_bpm_verify.json: {exc}")

    print()
    print("=== disagreements of ~2x ===")
    for r in out:
        if r.get("native") and r.get("kick"):
            ratio = r["native"] / r["kick"]
            if 1.8 < ratio < 2.2 or 0.45 < ratio < 0.55:
                print(
                    f"  native={r['native']:6.1f}  kick={r['kick']:6.1f}  "
                    f"{os.path.relpath(r['path'], MUSIC)}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
