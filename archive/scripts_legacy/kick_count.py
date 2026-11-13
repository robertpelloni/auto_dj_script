"""Definitive tempo check: count kick onsets per second.

Ambiguity between T and 2T comes from whole-track autocorrelation, which is
easily fooled by a two-beat bass pattern. Counting actual kick transients is
much more direct:

  psytrance 143 BPM -> a kick on every beat -> ~2.4 kicks/second
  downtempo  72 BPM -> a kick on every beat -> ~1.2 kicks/second

So kicks_per_second * 60 is the tempo at which auto_dj will actually pace the
track, and the jitter tells us whether the source is machine-regular (a real
kick pattern) or loose (a mis-detection).

Usage: python kick_count.py <file> [<file> ...]
       python kick_count.py --suspects
"""

import os
import sys
import math
import typing

typing.cast(typing.Any, sys.stdout).reconfigure(
    line_buffering=True, encoding="utf-8", errors="replace"
)

MUSIC = os.path.join(os.path.expanduser("~"), "Music")

SUSPECTS = [
    r"Ajja - Tulpa\03 - Ajja - Bazinga.mp3",
    r"Ajja - Tulpa\10 - Ajja - Three Dots.mp3",
    r"Ajja - Tulpa\09 - Ajja - Widdershins.mp3",
    r"Ajja - Oddular\08 - Ajja - Aria 51.mp3",
    r"Ajja - Oddular\07 - Ajja - Original Spin.mp3",
    r"Ajja - Chillsome\06 - Ajja - Chillsome.mp3",
    r"Ajja - Chillsome\05 - Ajja - Cajun.mp3",
    r"Ajja - Chillsome\03 - Ajja - Good Morning Albert.mp3",
    r"Ajja - Chillsome\01 - Ajja - Full Moon Koh Pha Ngan.mp3",
    r"Ajja - Spira Mirabilis\07 - Ajja - Entangled.mp3",
    r"Mekkanikka & Ajja - Ultrasonic\02 - Mekkanikka & Ajja - Ultrasonic.mp3",
    r"Undefined Behavior (Burn In Noise & Ajja) - Howling at the Moon"
    r"\01 - Undefined Behavior (Burn In Noise & Ajja) - Howling at the Moon.mp3",
    r"Ajja - Psychogenica\03 - Ajja - Juicy Shrooms.mp3",
]


def safe_int(value, fallback):
    """int(value) guarded against None/inf/nan/garbage."""
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return n if n > 0 else fallback


def safe_float(value, fallback=0.0):
    """float(value) guarded against None/inf/nan/garbage."""
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return f if math.isfinite(f) else fallback


def kick_stats(path, lo=40.0, hi=120.0, analyze_secs=90.0):
    """Kick rate and median inter-onset interval from the kick band."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import butter, find_peaks, lfilter, sosfilt

    info = sf.info(path)
    sr = info.samplerate
    frames = min(info.frames, safe_int(sr * analyze_secs, 1))
    data, sr = sf.read(path, frames=frames, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) == 2:
        data = np.mean(data, axis=1)
    y = np.asarray(data, dtype=np.float64)

    nyq = safe_float(sr / 2.0, 0.0)
    if nyq <= 0:
        raise ValueError("bad sample rate")

    sos = butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype="band", output="sos")
    x = np.asarray(sosfilt(sos, y), dtype=np.float64)

    # amplitude envelope of the kick band, smoothed ~5 ms
    env = np.asarray(np.abs(x), dtype=np.float64)
    w = safe_int(sr * 0.005, 1)
    env = np.asarray(lfilter(np.ones(w) / w, [1.0], env), dtype=np.float64)

    # normalise and drop the quietest 20% (intro/outro silence)
    peak = safe_float(np.max(env), 0.0)
    if peak <= 0:
        raise ValueError("silent kick band")
    env = env / (peak + 1e-12)
    floor = safe_float(np.percentile(env, 20), 0.0)
    env = np.clip(env - floor, 0.0, None)

    # peaks at least 0.24 s apart (<= 250 BPM)
    min_dist = safe_int(sr * 0.24, 1)
    prom = safe_float(np.percentile(env, 75), 0.0) * 0.35
    peaks, _ = find_peaks(env, distance=min_dist, prominence=prom)

    dur = len(env) / sr if sr else 0.0
    rate = len(peaks) / dur * 60.0 if dur > 0 else 0.0
    if len(peaks) > 3:
        ioi = np.diff(peaks)
        med_ioi = safe_float(np.median(ioi), 0.0) / sr if sr else 0.0
        q1 = safe_float(np.percentile(ioi, 25), 0.0)
        q3 = safe_float(np.percentile(ioi, 75), 0.0)
        reg = (q3 - q1) / sr if sr else 0.0
    else:
        med_ioi, reg = 0.0, 0.0

    return {
        "kicks": len(peaks),
        "secs": dur,
        "kicks_per_sec": rate,
        "median_ioi": med_ioi,
        "regularity": reg,
    }


def report(path):
    name = os.path.basename(path)
    try:
        s = kick_stats(path)
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"  {name[:50]:50s}  FAIL {type(e).__name__}: {e}")
        return
    kps = s["kicks_per_sec"]
    if kps >= 2.0:
        verdict = "  => ~%d BPM (psytrance, kick every beat)" % round(kps * 60)
    elif kps >= 1.5:
        verdict = "  => ~%d BPM (mid-tempo)" % round(kps * 60)
    else:
        verdict = "  => ~%d BPM (slow / downtempo)" % round(kps * 60)
    print(
        f"  {name[:50]:50s}  kicks/s={kps:5.2f}  "
        f"IOI={s['median_ioi'] * 1000:6.0f}ms  "
        f"jitter={s['regularity'] * 1000:5.0f}ms{verdict}"
    )


def main():
    args = sys.argv[1:]
    if not args:
        args = ["--suspects"]
    if args == ["--suspects"]:
        for rel in SUSPECTS:
            report(os.path.join(MUSIC, rel))
    else:
        for p in args:
            report(p if os.path.isabs(p) else os.path.join(MUSIC, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
