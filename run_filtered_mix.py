"""Run mix with BPM pre-filtering — only include tracks close to target BPM."""

import sys
import os
import time
import shutil
import glob

sys.stdout.reconfigure(line_buffering=True)

import soundfile as sf
import numpy as np

HOME = os.path.expanduser("~/Music")
SRCDIR = "psyset_source"
OUTDIR = "artist_mixes"

artist = sys.argv[1]
target_bpm = int(sys.argv[2])
max_tracks = int(sys.argv[3]) if len(sys.argv) > 3 else 15

print(f"=== {artist} @ {target_bpm} BPM (max {max_tracks} tracks) ===")

# Collect all tracks
all_tracks = []
for album in glob.glob(os.path.join(HOME, f"{artist}*")):
    if not os.path.isdir(album):
        continue
    for f in glob.glob(os.path.join(album, "*")):
        if f.lower().endswith((".flac", ".mp3", ".wav")):
            all_tracks.append(f)
all_tracks.sort()

print(f"  Total tracks found: {len(all_tracks)}")

# Quick BPM scan using the improved engine analysis
from autodj.analysis import get_native_bpm

track_bpms = []
for f in all_tracks:
    try:
        data, sr = sf.read(f, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = np.mean(data, axis=1)  # average channels, keep time
        bpm, _, _ = get_native_bpm(data, sr)
        if 60 < bpm < 220:
            track_bpms.append((f, bpm))
    except Exception:
        pass  # skip corrupt files

if not track_bpms:
    print("  No valid tracks found!")
    sys.exit(1)

# Filter to tracks within 25% of target BPM
close_tracks = [
    (f, b) for f, b in track_bpms if abs(b - target_bpm) / target_bpm < 0.25
]
# Sort by distance from target
close_tracks.sort(key=lambda x: abs(x[1] - target_bpm))
selected = close_tracks[:max_tracks]

print(f"  Scanned {len(track_bpms)} tracks with valid BPM")
print(f"  Selected {len(selected)} within 25% of {target_bpm} BPM:")
for f, b in selected:
    delta = b - target_bpm
    sign = "+" if delta >= 0 else ""
    print(f"    {os.path.basename(f):60s} {b:5.1f} ({sign}{delta:.0f})")

if len(selected) < 2:
    print("  Not enough tracks!")
    sys.exit(1)

# Clear and copy
for f in glob.glob(f"{SRCDIR}/*"):
    os.remove(f)
for f, b in selected:
    shutil.copy2(f, os.path.join(SRCDIR, os.path.basename(f)))

# Run mix
from autodj.core import compile_master_set
import config

safe_name = artist.lower().replace(" ", "_")
output_file = f"./{safe_name}_mix.flac"


class Args:
    input = config.INPUT_FOLDER
    output = output_file
    bpm = target_bpm
    end_bpm = None
    dbfs = config.TARGET_DBFS
    lowpass = config.LOWPASS_CUTOFF
    highpass = config.HIGHPASS_CUTOFF
    transition_bars = config.TRANSITION_BARS
    beats_per_bar = config.BEATS_PER_BAR
    dry_run = False
    reorder = False
    archetype = "auto"
    source_plugin = "local_folder"
    output_plugin = "local_file"
    mastering_intensity = 0.5
    dynamic_transitions = True
    genre_aware_mastering = True
    dynamic_energy_mastering = True
    adaptive_spectral_balancing = True
    broadcast_mode = False
    stream_url = None


status = {"status": f"Starting {artist}...", "progress": 0, "tracklist": []}
t0 = time.time()

try:
    compile_master_set(Args(), status_obj=status)
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

elapsed = time.time() - t0

info = sf.info(output_file)
print(
    f"\n{artist} complete: {info.frames / info.samplerate / 60:.1f} min in {elapsed:.0f}s"
)
