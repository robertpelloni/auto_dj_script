"""Run a mix for a single artist at specified BPM."""

import sys
import time

sys.stdout.reconfigure(line_buffering=True)
from autodj.core import compile_master_set
import config

artist = sys.argv[1]
target_bpm = int(sys.argv[2])
output_file = f"./{artist.lower().replace(' ', '_')}_mix.flac"


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

import soundfile as sf

info = sf.info(output_file)
print(
    f"\n{artist} complete: {info.frames / info.samplerate / 60:.1f} min in {elapsed:.0f}s"
)
