"""Render the modern Ajja mix from the selection staged in psyset_source/.

Tracks were selected by final_select.py (all Ajja >130 BPM from the Soulseek
library, minus live sets and the downtempo albums).

Modern-psytrance settings are all enabled: dynamic transitions, genre-aware
mastering, dynamic energy mastering and adaptive spectral balancing.
"""

import os
import sys
import time
import typing

typing.cast(typing.Any, sys.stdout).reconfigure(
    line_buffering=True, encoding="utf-8", errors="replace"
)

import config  # noqa: E402
from autodj.core import compile_master_set  # noqa: E402

TARGET_BPM = 145
OUTPUT = os.path.join("artist_mixes", "ajja_modern_mix.flac")


class Args:
    input = config.INPUT_FOLDER
    output = OUTPUT
    bpm = TARGET_BPM
    end_bpm = None
    dbfs = config.TARGET_DBFS
    lowpass = config.LOWPASS_CUTOFF
    highpass = config.HIGHPASS_CUTOFF
    transition_bars = config.TRANSITION_BARS
    beats_per_bar = config.BEATS_PER_BAR
    bass_transition = getattr(config, "BASS_TRANSITION_STYLE", "swap")
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


def count_tracks():
    try:
        entries = os.listdir(config.INPUT_FOLDER)
    except OSError:
        return 0
    return len(
        [
            f
            for f in entries
            if f.lower().endswith((".mp3", ".flac", ".wav", ".m4a", ".aiff"))
        ]
    )


def main():
    n = count_tracks()
    print(f"=== MODERN AJJA MIX @ {TARGET_BPM} BPM ===")
    print(f"source : {config.INPUT_FOLDER} ({n} tracks)")
    print(f"output : {OUTPUT}")
    print()

    if n < 2:
        print("Not enough tracks staged. Run final_select.py first.")
        return 1

    outdir = os.path.dirname(OUTPUT)
    try:
        if outdir:
            os.makedirs(outdir, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create {outdir}: {exc}")
        return 1

    status = {"status": "Starting modern Ajja mix...", "progress": 0, "tracklist": []}
    t0 = time.time()
    try:
        compile_master_set(Args(), status_obj=status)
    except Exception as e:  # noqa: BLE001 - surface the full traceback
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
        return 1

    elapsed = time.time() - t0

    try:
        import soundfile as sf

        info = sf.info(OUTPUT)
        mins = info.frames / info.samplerate / 60
        size = os.path.getsize(OUTPUT) / 1e9
        print()
        print(
            f"=== DONE: {mins:.1f} min, {size:.2f} GB, rendered in "
            f"{elapsed / 60:.1f} min ==="
        )
    except Exception as e:  # noqa: BLE001 - report and continue
        print(
            f"render finished in {elapsed / 60:.1f} min "
            f"(could not stat output: {type(e).__name__}: {e})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
