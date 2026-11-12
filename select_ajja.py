"""Select Ajja's uptempo catalogue for the "modern" mix.

Decision rules, in order:

1. Drop non-tracks. Six files are full festival live sets (90-120 min) and two
   are previous auto_dj renders (288 min / the old all-ajja render). A DJ set
   is not a track and would dominate a mix.

2. Drop the downtempo albums. Their own genre tags say so:
     Spira Mirabilis .............. Downtempo
     The Alien Jams (HR Giger) .... Downtempo / Electronic  (also a live jam)
     MOS 6581 Remixes ............. Ambient; Electronic; Chillout
   (Carbon Based Lifeforms - MOS 6581 (Ajja remix))

3. Correct the octave. get_native_bpm() returns 2x the real tempo on slow
   material, so anything reading above 152 BPM is halved. Ajja's uptempo
   catalogue sits at 142-150; readings of 153-199 are all doubled downtempo.

4. Keep what is still above 130 BPM.

Writes ajja_modern_selection.json and prints the list.
"""

import json
import math
import os
import shutil
import sys
import typing

typing.cast(typing.Any, sys.stdout).reconfigure(
    line_buffering=True, encoding="utf-8", errors="replace"
)

MUSIC = os.path.join(os.path.expanduser("~"), "Music")
SRCDIR = "psyset_source"

MAX_TRACK_SECS = 15 * 60  # longer than this => a set, not a track
OCTAVE_CUTOFF = 152.0  # above this we assume the estimate is doubled
MIN_BPM = 130.0

# albums whose own genre tag is downtempo/chillout
DOWNTEMPO_ALBUMS = {
    "Spira Mirabilis",
    "The Alien Jams: Live at the HR Giger Museum",
    "MOS 6581 Remixes",
}


def safe_float(value, fallback=0.0):
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return f if math.isfinite(f) else fallback


def main():
    try:
        with open("ajja_modern_scan.json", encoding="utf-8") as fh:
            scan = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read ajja_modern_scan.json: {exc}")
        return 1

    keep, dropped = [], []
    for r in scan["all"]:
        path = r["path"]
        bpm = safe_float(r.get("bpm"), 0.0)
        dur = safe_float(r.get("dur"), 0.0)
        rel = os.path.relpath(path, MUSIC)
        album = rel.split(os.sep)[0]

        if not bpm:
            dropped.append((rel, "no BPM", bpm))
            continue
        if dur > MAX_TRACK_SECS:
            dropped.append((rel, f"set/mix ({dur / 60:.0f} min)", bpm))
            continue
        if album in DOWNTEMPO_ALBUMS:
            dropped.append((rel, "downtempo album", bpm))
            continue

        corrected = bpm / 2.0 if bpm > OCTAVE_CUTOFF else bpm
        if corrected <= MIN_BPM:
            dropped.append((rel, f"<= {MIN_BPM:.0f} BPM", corrected))
            continue
        keep.append(
            {"path": path, "rel": rel, "bpm": corrected, "raw": bpm, "dur": dur}
        )

    keep.sort(key=lambda x: x["bpm"])

    print(f"kept   : {len(keep)}")
    print(f"dropped: {len(dropped)}")
    print()
    print(f"=== SELECTED (>{MIN_BPM:.0f} BPM, sorted) ===")
    for t in keep:
        note = f"  (raw {t['raw']:.1f} halved)" if t["raw"] != t["bpm"] else ""
        print(f"  {t['bpm']:6.1f}  {t['dur'] / 60:4.1f}min  {t['rel'][:74]}{note}")

    print()
    print("=== DROPPED ===")
    for rel, why, bpm in dropped:
        print(f"  {why:<22} {bpm:6.1f}  {rel[:76]}")

    if len(keep) < 2:
        print("\nNot enough tracks to mix.")
        return 1

    # stage into psyset_source
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
        return 1

    for t in keep:
        dest = os.path.join(SRCDIR, os.path.basename(t["path"]))
        try:
            shutil.copy2(t["path"], dest)
        except OSError as exc:
            print(f"  copy failed: {os.path.basename(t['path'])}: {exc}")

    try:
        entries = os.listdir(SRCDIR)
    except OSError as exc:
        print(f"error: cannot read {SRCDIR}: {exc}")
        return 1
    staged = [
        f
        for f in entries
        if f.lower().endswith((".mp3", ".flac", ".wav", ".m4a", ".aiff"))
    ]
    print()
    print(f"staged {len(staged)} tracks into {SRCDIR}/")
    print(f"total runtime ~{sum(t['dur'] for t in keep) / 60:.0f} min")
    print(f"median BPM    ~{keep[len(keep) // 2]['bpm']:.1f}")

    try:
        with open("ajja_modern_selection.json", "w", encoding="utf-8") as fh:
            json.dump(keep, fh, indent=1)
    except OSError as exc:
        print(f"warning: could not write ajja_modern_selection.json: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
