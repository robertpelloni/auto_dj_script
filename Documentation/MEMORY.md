# 🧠 Project Memory & Technical Observations (8.14.0)

## Design Preferences

- **Functional DSP**: Prefer pure functions for DSP operations in \`dsp.py\` to simplify parallel execution.
- **Pydub Interop**: Use \`pydub\` for I/O and slicing, but convert to \`numpy\` arrays for all signal processing.
- **Camelot Dominance**: Always use the Camelot wheel for key compatibility logic as it is the industry standard for DJing.

## Ongoing Observations

1. **Octave Error Sensitivity**: \`librosa.beat_track\` is sensitive to high-frequency transients. Implementing a "tempo prior" or "sanity floor" (doubling < 100 BPM) has significantly improved reliability for Psytrance.
2. **Phase Vocoder Artifacts**: While Phase Vocoders are superior to interpolation, warping beyond 10% can introduce robotic artifacts. The engine should prioritize set-wide tempo adjustments to minimize per-track warp ratios.
3. **SA Cooling Schedule**: The current cooling rate in \`core.py\` is optimized for ~20 track sets. Larger sets (50+) may require a logarithmic cooling schedule.
4. **LUFS vs Peak**: Moving from Peak-normalization to -14.0 LUFS integrated loudness has fixed the "level jumping" issue observed in v2.x.

## Session Insights (v8.14.0)

### Bass Ducking Removal & LUFS Normalization

- **Removed beat-synced bass ducking** that caused false kick peak detections in quiet sections (ambient intros, breakdowns), creating audible 25ms volume dips every beat.
- **Replaced simple RMS normalization** with ITU-R BS.1770 K-weighted LUFS normalization for perceptually accurate loudness.
- **Chunked loudness measurement** uses 120-second processing windows to prevent 8+ GB out-of-memory on 416-minute mixes.
- **Corrected stereo channel summing** using BS.1770 power sum with -0.691dB correction.

### Zero-Phase Crossover Fix

- **Replaced lfilter with filtfilt** in the bass swap crossover for zero-phase bass extraction. The lfilter group delay (~1.7ms) caused a +2.1dB frequency boost around 150-200Hz during transitions, producing crosstalk ducking artifacts.
- **Max L/R channel imbalance reduced from 7.0 dB to 1.0 dB** during transitions.

### Large File Export

- **Fixed 4GB WAV limit** by switching to direct soundfile FLAC export for large mixes.
- **Export-only-on-last-track fix**: Wrapped final mix normalization and export in `if i == num_tracks - 1:` to prevent running on every iteration.

### Previous Session (v8.13.0)

- **Hardware Abstraction**: MIDI integration in `autodj/midi.py` follows a threaded observer pattern.
- **Library Ingestion**: Rekordbox XML parsing requires robust handling of URL-encoded strings.
- **Quantum SA Optimization**: Parallelizing the Simulated Annealing engine into 4 concurrent cooling branches.
- Consolidating 8+ instruction files into a single `GLOBAL_LLM_DIRECTIVE.md`.
- Single-source versioning via `VERSION.md`.
