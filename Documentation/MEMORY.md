# 🧠 Project Memory & Technical Observations (8.16.0)

## Design Preferences

- **Functional DSP**: Prefer pure functions for DSP operations in \`dsp.py\` to simplify parallel execution.
- **Pydub Interop**: Use \`pydub\` for I/O and slicing, but convert to \`numpy\` arrays for all signal processing.
- **Camelot Dominance**: Always use the Camelot wheel for key compatibility logic as it is the industry standard for DJing.

## Ongoing Observations

1. **Octave Error Sensitivity**: \`librosa.beat_track\` is sensitive to high-frequency transients. Implementing a "tempo prior" or "sanity floor" (doubling < 100 BPM) has significantly improved reliability for Psytrance.
2. **Phase Vocoder Artifacts**: While Phase Vocoders are superior to interpolation, warping beyond 10% can introduce robotic artifacts. The engine should prioritize set-wide tempo adjustments to minimize per-track warp ratios.
3. **SA Cooling Schedule**: The current cooling rate in \`core.py\` is optimized for ~20 track sets. Larger sets (50+) may require a logarithmic cooling schedule.
4. **LUFS vs Peak**: Moving from Peak-normalization to -14.0 LUFS integrated loudness has fixed the "level jumping" issue observed in v2.x.

## Session Insights (v8.16.0)

### Pure-NumPy Spectral Flux & Kick-IOI BPM Detection

- **NumPy 2.5 / `numba` Compatibility**: `librosa.onset` triggers a broken lazy `numba` import on Python 3.14 + NumPy 2.5. `autodj/analysis.py` now computes 4-band spectral flux + multi-scale autocorrelation purely in NumPy/SciPy.
- **Two-Beat ACF Octave Trap**: Even multi-band ACF can lock onto two-beat or half-beat lags on psytrance tracks (producing ~99 BPM or ~190 BPM for 145 BPM tracks). `_kick_interval_bpm()` measures median inter-onset intervals in the `40–120 Hz` kick band and overrides ACF whenever interquartile jitter is `<= 20 ms`.
- **Soundfile Channel Axis**: `soundfile.read(..., always_2d=False)` returns `(N_samples, 2)` for stereo files (`axis=1` averages channels), whereas `pydub_to_ndarray` uses `(2, N_samples)`. Always check shape or average along the smaller 2-channel axis.

### Equal-Power Crossfades & Dual Bass Transition Modes

- **Fader Exponents Removed**: Asymmetric `x**0.8` / `x**1.2` exponents inside `cos`/`sin` crossfades caused a `-1.1 dB` power dip at `x = 0.5`. Standard `cos(pi*x/2)` and `sin(pi*x/2)` restore flat `cos^2 + sin^2 = 1.0` power summation.
- **Dual Bass Curves (`--bass-transition [swap|fade]`)**: `swap` holds outgoing bass for 40% then swaps over 40% of the overlap; `fade` crossfades low-end across 100% of the overlap for tracks whose outro/intro kicks have very different timbres.
- **Plugin Registration Scope**: `@PluginRegistry.register_tool` classes (such as `SmartReplenishTool`) must live at module top level so registration runs on import.

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
