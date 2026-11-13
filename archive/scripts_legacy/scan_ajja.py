"""Scan every Ajja track in the Soulseek library (~/Music) and report BPM.

Ajja appears both as a primary artist ("Ajja - Album") and in collaborations
("Mekkanikka & Ajja - ...", "Alpha Portal & Ajja - ...", "GMS; Ajja; Pixel - ..."),
so we match on any path component containing "ajja".

Phase 1 of the "modern Ajja >130 BPM" mix: just report, no mixing.
"""

import os
import re
import sys
import json
import time
import typing
from concurrent.futures import ProcessPoolExecutor, as_completed

# sys.stdout is typed as TextIO, which lacks reconfigure(); cast to silence that.
typing.cast(typing.Any, sys.stdout).reconfigure(
    line_buffering=True, encoding="utf-8", errors="replace"
)

MUSIC = os.path.join(os.path.expanduser("~"), "Music")
AUDIO = (".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aiff", ".aif")


def find_ajja_tracks(root):
    """All audio files whose path mentions Ajja in any component."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in ("node_modules", ".git", "__pycache__")
        ]
        # a path component (dir or the file itself) mentioning ajja
        hay = os.path.join(dirpath, "").lower()
        for fn in filenames:
            if not fn.lower().endswith(AUDIO):
                continue
            if "ajja" in hay or "ajja" in fn.lower():
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def norm_key(path):
    """Dedupe key: filename minus duplicate markers like ' (1).flac'."""
    base = os.path.basename(path)
    base = re.sub(r"\.(mp3|flac|wav|m4a|ogg|aiff|aif)$", "", base, flags=re.I)
    base = re.sub(r"\s*\(\d+\)$", "", base)  # 'name (1)'
    base = re.sub(r"\s*[-_]\s*\d+$", "", base)  # stray trailing numbers
    return base.strip().lower()


def scan_one(args):
    (path,) = args
    import soundfile as sf
    from autodj.analysis import get_native_bpm

    t0 = time.time()
    try:
        info = sf.info(path)
        sr = info.samplerate
        # 150 s is plenty for tempo estimation and much faster than full decode
        frames = min(info.frames, int(sr * 150))
        data, sr = sf.read(path, frames=frames, dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) == 2:
            data = data.mean(axis=1)
        bpm, _, _ = get_native_bpm(data, sr)
        dur = info.frames / info.samplerate
        return {
            "path": path,
            "bpm": float(bpm),
            "dur": dur,
            "sr": sr,
            "secs": time.time() - t0,
            "err": None,
        }
    except Exception as e:
        return {
            "path": path,
            "bpm": None,
            "dur": None,
            "sr": None,
            "secs": time.time() - t0,
            "err": f"{type(e).__name__}: {e}",
        }


def main():
    tracks = find_ajja_tracks(MUSIC)
    print(f"Ajja audio files found: {len(tracks)}")

    # drop duplicate rips of the same title
    by_key = {}
    for t in tracks:
        k = norm_key(t)
        if k not in by_key:
            by_key[k] = t
        else:
            # prefer the shorter path (usually the non-"(1)" original)
            if len(t) < len(by_key[k]):
                by_key[k] = t
    deduped = sorted(by_key.values())
    print(f"after dedupe by title : {len(deduped)}")
    print()

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        futs = {ex.submit(scan_one, (t,)): t for t in deduped}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            name = os.path.basename(r["path"])
            if r["err"]:
                print(f"  [{i:3d}/{len(deduped)}] FAIL {name[:60]:60s} {r['err'][:60]}")
            else:
                print(
                    f"  [{i:3d}/{len(deduped)}] {r['bpm']:6.1f} BPM  "
                    f"{r['dur'] / 60:5.1f}min  {name[:64]}"
                )

    ok = [r for r in results if r["bpm"] and 60 < r["bpm"] < 220]
    over130 = [r for r in ok if r["bpm"] > 130]
    over130.sort(key=lambda r: r["bpm"])

    print()
    print(f"scanned in {time.time() - t0:.0f}s")
    print(f"valid BPM : {len(ok)}")
    print(f"> 130 BPM : {len(over130)}")
    print()
    print("=== >130 BPM, sorted ===")
    for r in over130:
        print(
            f"  {r['bpm']:6.1f}  {r['dur'] / 60:5.1f}min  "
            f"{os.path.relpath(r['path'], MUSIC)[:80]}"
        )

    try:
        with open("ajja_modern_scan.json", "w", encoding="utf-8") as fh:
            json.dump({"all": results, "over130": over130}, fh, indent=1)
    except OSError as exc:
        print(f"warning: could not write ajja_modern_scan.json: {exc}")
    else:
        print()
        print(f"wrote ajja_modern_scan.json ({len(results)} entries)")


if __name__ == "__main__":
    main()
