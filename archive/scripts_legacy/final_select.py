"""Final track selection for the modern Ajja mix.

Tempo resolution
----------------
get_native_bpm() makes octave errors in BOTH directions on this corpus:
  Ultrasonic        native 190.1  real 145.0
  Howling at Moon   native  99.1  real 145.0
so neither "trust native" nor "always halve the big ones" works.

The kick-interval measurement is decisive when the kick is machine-regular:
a jitter of ~0 ms means every kick lands exactly on the beat grid, so
60/median_IOI is the true tempo. When jitter is large the detected peaks are
not a clean kick pattern and we fall back to an octave-corrected native BPM.

Selection
---------
  * drop non-tracks       : anything longer than 15 min (live sets, old renders)
  * drop downtempo albums : Spira Mirabilis, Alien Jams, MOS 6581 (genre tags)
  * resolve tempo as above
  * keep > 130 BPM
"""

import json
import math
import os
import shutil
import sys
import typing
from concurrent.futures import ProcessPoolExecutor, as_completed

typing.cast(typing.Any, sys.stdout).reconfigure(
    line_buffering=True, encoding="utf-8", errors="replace"
)

MUSIC = os.path.join(os.path.expanduser("~"), "Music")
SRCDIR = "psyset_source"

MAX_TRACK_SECS = 15 * 60
MIN_BPM = 130.0
OCTAVE_CUTOFF = 152.0
JITTER_TRUST = 0.020  # <= 20 ms of jitter => trust the kick interval

DOWNTEMPO_MARKERS = (
    "spira mirabilis",
    "alien jams",
    "hr giger",
    "mos 6581",
)


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

    env = np.asarray(np.abs(x), dtype=np.float64)
    w = safe_int(sr * 0.005, 1)
    env = np.asarray(lfilter(np.ones(w) / w, [1.0], env), dtype=np.float64)

    peak = safe_float(np.max(env), 0.0)
    if peak <= 0:
        raise ValueError("silent kick band")
    env = env / (peak + 1e-12)
    floor = safe_float(np.percentile(env, 20), 0.0)
    env = np.clip(env - floor, 0.0, None)

    min_dist = safe_int(sr * 0.24, 1)
    prom = safe_float(np.percentile(env, 75), 0.0) * 0.35
    peaks, _ = find_peaks(env, distance=min_dist, prominence=prom)

    if len(peaks) <= 3 or sr <= 0:
        return {"median_ioi": 0.0, "regularity": 1.0}

    ioi = np.diff(peaks)
    med_ioi = safe_float(np.median(ioi), 0.0) / sr
    q1 = safe_float(np.percentile(ioi, 25), 0.0)
    q3 = safe_float(np.percentile(ioi, 75), 0.0)
    return {"median_ioi": med_ioi, "regularity": (q3 - q1) / sr}


def analyse(path):
    """Return (path, ioi_bpm, jitter_secs) or (path, None, None) on failure."""
    try:
        s = kick_stats(path)
    except Exception:  # noqa: BLE001 - treat as unusable measurement
        return path, None, None
    ioi = safe_float(s["median_ioi"], 0.0)
    jit = safe_float(s["regularity"], 1.0)
    if ioi <= 0:
        return path, None, None
    return path, 60.0 / ioi, jit


def load_scan():
    try:
        with open("ajja_modern_scan.json", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read ajja_modern_scan.json: {exc}")
        return None


def stage(tracks):
    try:
        if os.path.isdir(SRCDIR):
            for f in os.listdir(SRCDIR):
                fp = os.path.join(SRCDIR, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        else:
            os.makedirs(SRCDIR, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot prepare {SRCDIR}: {exc}")
        return 0

    n = 0
    for t in tracks:
        try:
            shutil.copy2(t["path"], os.path.join(SRCDIR, os.path.basename(t["path"])))
            n += 1
        except OSError as exc:
            print(f"  copy failed: {os.path.basename(t['path'])}: {exc}")
    return n


def main():
    scan = load_scan()
    if scan is None:
        return 1

    candidates = []
    dropped = []
    for r in scan["all"]:
        rel = os.path.relpath(r["path"], MUSIC)
        dur = safe_float(r.get("dur"), 0.0)
        native = safe_float(r.get("bpm"), 0.0)
        low = rel.lower()
        if dur > MAX_TRACK_SECS:
            dropped.append((rel, f"set/mix ({dur / 60:.0f} min)", native))
        elif any(m in low for m in DOWNTEMPO_MARKERS):
            dropped.append((rel, "downtempo album", native))
        else:
            candidates.append(
                {"path": r["path"], "rel": rel, "dur": dur, "native": native}
            )

    print(f"candidates for tempo analysis: {len(candidates)}")

    ioi_bpm = {}
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        futs = [ex.submit(analyse, c["path"]) for c in candidates]
        for f in as_completed(futs):
            p, b, j = f.result()
            ioi_bpm[p] = (b, j)

    keep = []
    for c in candidates:
        b, j = ioi_bpm.get(c["path"], (None, None))
        native = c["native"]
        tempo = native
        source = "native"

        if b and j is not None and j <= JITTER_TRUST and 60 < b < 250:
            tempo = b
            source = f"kick (jitter {j * 1000:.0f}ms)"
        elif native > OCTAVE_CUTOFF:
            tempo = native / 2.0
            source = "native halved"
        elif native < 120 and b and 120 < b < 200:
            tempo = b
            source = "kick (native was halved)"

        if tempo <= MIN_BPM:
            dropped.append((c["rel"], f"<={MIN_BPM:.0f} BPM", tempo))
            continue
        c["tempo"] = tempo
        c["source"] = source
        keep.append(c)

    keep.sort(key=lambda x: x["tempo"])

    print(f"\nkept: {len(keep)}   dropped: {len(dropped)}")
    print(f"\n=== SELECTED ({len(keep)} tracks, >{MIN_BPM:.0f} BPM) ===")
    for t in keep:
        print(
            f"  {t['tempo']:6.1f}  {t['dur'] / 60:4.1f}min  "
            f"[{t['source']:<22}] {t['rel'][:58]}"
        )

    print("\n=== DROPPED ===")
    for rel, why, bpm in dropped:
        print(f"  {why:<24} {bpm:6.1f}  {rel[:68]}")

    if len(keep) < 2:
        print("\nNot enough tracks to mix.")
        return 1

    n = stage(keep)
    print()
    print(f"staged {n} tracks into {SRCDIR}/")
    print(f"total runtime ~{sum(t['dur'] for t in keep) / 60:.0f} min")
    print(f"median tempo  ~{keep[len(keep) // 2]['tempo']:.1f} BPM")

    try:
        with open("ajja_modern_selection.json", "w", encoding="utf-8") as fh:
            json.dump(keep, fh, indent=1)
    except OSError as exc:
        print(f"warning: could not write ajja_modern_selection.json: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
