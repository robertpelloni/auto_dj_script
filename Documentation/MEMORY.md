# 🧠 Project Memory & Technical Observations (8.11.0)

## Design Preferences
- **Functional DSP**: Prefer pure functions for DSP operations in \`dsp.py\` to simplify parallel execution.
- **Pydub Interop**: Use \`pydub\` for I/O and slicing, but convert to \`numpy\` arrays for all signal processing.
- **Camelot Dominance**: Always use the Camelot wheel for key compatibility logic as it is the industry standard for DJing.

## Ongoing Observations
1. **Octave Error Sensitivity**: \`librosa.beat_track\` is sensitive to high-frequency transients. Implementing a "tempo prior" or "sanity floor" (doubling < 100 BPM) has significantly improved reliability for Psytrance.
2. **Phase Vocoder Artifacts**: While Phase Vocoders are superior to interpolation, warping beyond 10% can introduce robotic artifacts. The engine should prioritize set-wide tempo adjustments to minimize per-track warp ratios.
3. **SA Cooling Schedule**: The current cooling rate in \`core.py\` is optimized for ~20 track sets. Larger sets (50+) may require a logarithmic cooling schedule.
4. **LUFS vs Peak**: Moving from Peak-normalization to -14.0 LUFS integrated loudness has fixed the "level jumping" issue observed in v2.x.

## Session Insights (v8.11.0)
- **Hardware Abstraction**: MIDI integration in `autodj/midi.py` follows a threaded observer pattern, ensuring that tactile input does not block the real-time telemetry loop.
- **Library Ingestion**: Rekordbox XML parsing requires robust handling of URL-encoded strings (e.g., `%20` for spaces) to resolve local file paths accurately.
- **Quantum SA Optimization**: Parallelizing the Simulated Annealing engine into 4 concurrent cooling branches (v8.11.0) provides a 4x increase in exploration density, significantly reducing the likelihood of getting stuck in local minima for large sets.
- Consolidating 8+ instruction files into a single \`GLOBAL_LLM_DIRECTIVE.md\` has significantly reduced agent context pollution.
- Single-source versioning via \`VERSION.md\` prevents synchronization drifts between the package and the documentation.
