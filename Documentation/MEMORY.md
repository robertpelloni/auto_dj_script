# 🧠 Project Memory & Technical Observations (v5.5.0)

## Design Preferences
- **Functional DSP**: Prefer pure functions for DSP operations in \`dsp.py\` to simplify parallel execution.
- **Pydub Interop**: Use \`pydub\` for I/O and slicing, but convert to \`numpy\` arrays for all signal processing.
- **Camelot Dominance**: Always use the Camelot wheel for key compatibility logic as it is the industry standard for DJing.

## Ongoing Observations
1. **Parallel Warp Efficiency**: Implementing the Parallel Warp Engine (v5.3.0) using \`ProcessPoolExecutor\` reduced mix rendering time by ~70% for a 12-track set.
2. **Interactive Overrides**: The addition of manual archetype selection in the GUI (v5.5.0) has resolved user-reported "clashing styles" in complex transitions.
3. **Octave Error Sensitivity**: \`librosa.beat_track\` is sensitive to high-frequency transients. Implementing a "tempo prior" or "sanity floor" (doubling < 100 BPM) has significantly improved reliability for Psytrance.
4. **Phase Vocoder Artifacts**: While Phase Vocoders are superior to interpolation, warping beyond 10% can introduce robotic artifacts. The engine should prioritize set-wide tempo adjustments to minimize per-track warp ratios.
5. **SA Cooling Schedule**: The Logarithmic cooling schedule (v5.4.1) provides much faster convergence for the global sequence than the previous linear rate.
6. **LUFS vs Peak**: Moving from Peak-normalization to -14.0 LUFS integrated loudness has fixed the "level jumping" issue.

## Session Insights (v5.5.0)
- Consolidating 8+ instruction files into a single \`GLOBAL_LLM_DIRECTIVE.md\` has significantly reduced agent context pollution.
- Single-source versioning via \`VERSION.md\` prevents synchronization drifts between the package and the documentation.
