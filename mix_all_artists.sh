#!/bin/bash
set -e

HOME="$HOME/Music"
OUTDIR="$(pwd)/artist_mixes"
mkdir -p "$OUTDIR"

declare -A ARTIST_BPMS
ARTIST_BPMS["1200 Micrograms"]=145
ARTIST_BPMS["Space Tribe"]=148
ARTIST_BPMS["Mad Tribe"]=147
ARTIST_BPMS["Astrix"]=145
ARTIST_BPMS["Electric Universe"]=146
ARTIST_BPMS["Outsiders"]=146
ARTIST_BPMS["Cosmosis"]=146
ARTIST_BPMS["Koxbox"]=148
ARTIST_BPMS["Tristan"]=146
ARTIST_BPMS["Avalon"]=146
ARTIST_BPMS["Mekkanikka"]=145

for artist in "${!ARTIST_BPMS[@]}"; do
	bpm="${ARTIST_BPMS[$artist]}"
	echo ""
	echo "============================================"
	echo "=== $artist @ $bpm BPM ==="
	echo "============================================"

	# Clear source dir
	rm -f psyset_source/*

	# Find all tracks, keep first 25 (good representation across albums)
	count=0
	while IFS= read -r f; do
		if [[ "$f" =~ \.(flac|mp3|wav)$ ]]; then
			cp "$f" psyset_source/ 2>/dev/null
			((count++))
			[ $count -ge 25 ] && break
		fi
	done < <(find "$HOME" -maxdepth 2 -path "$HOME/${artist}*" -type f | sort)

	echo "Copied $count tracks"

	if [ "$count" -lt 2 ]; then
		echo "Not enough tracks, skipping"
		continue
	fi

	SAFE_NAME=$(echo "$artist" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
	OUTFILE="$OUTDIR/final_${SAFE_NAME}.flac"

	python -u -c "
import sys, time
sys.stdout.reconfigure(line_buffering=True)
from autodj.core import compile_master_set
import config

class Args:
    input = config.INPUT_FOLDER
    output = '$OUTFILE'
    bpm = $bpm
    end_bpm = None
    dbfs = config.TARGET_DBFS
    lowpass = config.LOWPASS_CUTOFF
    highpass = config.HIGHPASS_CUTOFF
    transition_bars = config.TRANSITION_BARS
    beats_per_bar = config.BEATS_PER_BAR
    dry_run = False
    reorder = False
    archetype = 'auto'
    source_plugin = 'local_folder'
    output_plugin = 'local_file'
    mastering_intensity = 0.5
    dynamic_transitions = True
    genre_aware_mastering = True
    dynamic_energy_mastering = True
    adaptive_spectral_balancing = True
    broadcast_mode = False
    stream_url = None

status = {'status': 'Starting $artist...', 'progress': 0, 'tracklist': []}
t0 = time.time()
compile_master_set(Args(), status_obj=status)
elapsed = time.time()-t0

import soundfile as sf
info = sf.info('$OUTFILE')
print(f'\n$artist complete: {info.frames/info.samplerate/60:.1f} min in {elapsed:.0f}s')
"

	echo ""
	echo "--- DONE: $artist ---"
done

echo ""
echo "===== ALL ARTIST MIXES COMPLETE ====="
ls -la "$OUTDIR"/*.flac 2>/dev/null
