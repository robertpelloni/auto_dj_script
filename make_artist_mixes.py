import os
import sys
import subprocess
import glob
import shutil

sys.stdout.reconfigure(line_buffering=True)

HOME = os.path.expanduser("~/Music")
SRC_DIR = "psyset_source"

# Artists and their target BPMs (based on genre knowledge)
TARGETS = {
    "1200 Micrograms": 145,
    "Space Tribe": 148,
    "Mad Tribe": 147,
    "Astrix": 145,
    "Electric Universe": 146,
    "Outsiders": 146,
    "Cosmosis": 146,
    "Koxbox": 148,
    "Tristan": 146,
    "Avalon": 146,
    "Mekkanikka": 145,
    "Menog": 147,
}

# Only include artists with sufficient tracks
ARTISTS = list(TARGETS.keys())

for artist in ARTISTS:
    print(f"\n{'=' * 60}")
    print(f"=== {artist.upper()} @ {TARGETS[artist]} BPM ===")
    print("=" * 60)

    # Collect all tracks
    all_tracks = []
    for album_dir in glob.glob(os.path.join(HOME, f"{artist}*")):
        if not os.path.isdir(album_dir):
            continue
        for f in glob.glob(os.path.join(album_dir, "*")):
            ext = f.lower()
            if ext.endswith((".flac", ".mp3", ".wav")):
                all_tracks.append(f)

    print(f"Found {len(all_tracks)} total tracks")
    if not all_tracks:
        continue

    # Quick BPM scan using a simpler method
    import numpy as np
    import soundfile as sf

    class TrackInfo:
        pass

    bpms = []
    for i, f in enumerate(all_tracks[:50]):  # Scan up to 50
        try:
            data, sr = sf.read(f, dtype="float32", always_2d=False)
            if data.ndim == 2:
                mono = np.mean(data, axis=0)[: sr * 20]  # First 20 sec
            else:
                mono = data[: sr * 20]
            # Simple onset detection
            from scipy.signal import hilbert

            envelope = np.abs(hilbert(mono[:: max(1, int(sr / 100))]))  # 100Hz envelope
            # Find peaks
            threshold = np.mean(envelope) * 1.5
            peaks = []
            for j in range(1, len(envelope) - 1):
                if (
                    envelope[j] > threshold
                    and envelope[j] > envelope[j - 1]
                    and envelope[j] > envelope[j + 1]
                ):
                    peaks.append(j)
            if len(peaks) > 4:
                intervals = [
                    peaks[k + 1] - peaks[k] for k in range(min(len(peaks) - 1, 10))
                ]
                avg_interval = sum(intervals) / len(intervals)
                bpm = 6000 / avg_interval  # 100Hz * 60 / interval
                if 60 < bpm < 200:
                    bpms.append((f, bpm, i))
        except Exception:
            pass

    if not bpms:
        print(f"  No BPM data for {artist}, skipping")
        continue

    target = TARGETS[artist]

    # Filter to tracks within 15% of target
    good = [(f, b) for f, b, i in bpms if abs(b - target) / target < 0.15]
    good.sort(key=lambda x: abs(x[1] - target))

    # Take top 25 tracks (or fewer if less available)
    selected = good[:25]

    print(f"  Scanned {len(bpms)} tracks with BPM")
    print(f"  Selected {len(selected)} tracks closest to {target} BPM:")
    for f, b in selected:
        basename = os.path.basename(f)[:55]
        print(f"    {basename:55s} {b:5.1f} BPM")

    if len(selected) < 2:
        print(f"  Not enough tracks for {artist}")
        continue

    # Clear and copy selected tracks
    shutil.rmtree(SRC_DIR, ignore_errors=True)
    os.makedirs(SRC_DIR, exist_ok=True)
    for f, b in selected:
        shutil.copy2(f, os.path.join(SRC_DIR, os.path.basename(f)))

    # Run mix
    output = f"final_{artist.lower().replace(' ', '_')}.flac"

    # Build run script
    code = f'''
import sys, time
sys.stdout.reconfigure(line_buffering=True)
from autodj.core import compile_master_set
import config

class Args:
    input = config.INPUT_FOLDER
    output = \'{output}\'
    bpm = {target}
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

status = {{"status": "Starting {artist}...", "progress": 0, "tracklist": []}}
t0 = time.time()
compile_master_set(Args(), status_obj=status)
elapsed = time.time()-t0

import soundfile as sf
info = sf.info("{output}")
print(f"\n{artist} complete: {{info.frames/info.samplerate/60:.1f}} min in {{elapsed:.0f}}s")
'''

    with open("_run_artist.py", "w") as f:
        f.write(code)

    result = subprocess.run(
        ["python", "-u", "_run_artist.py"], capture_output=True, text=True, timeout=1800
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-300:])
