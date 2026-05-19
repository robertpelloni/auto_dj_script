# Changelog

## [5.6.0] - 2024-05-21
### Added
- **Multi-band Compression**: Added advanced dynamics processing in the mastering chain. Applies `apply_multiband_compression` to the final master mix.

## [5.5.0] - 2024-05-21
### Added
- **Manual Archetype Overrides**: Added a dropdown to the GUI to manually select mixing styles (Bass-Swap, Echo-Out, etc.).
- **GUI UX Overhaul**: Integrated a live metadata tracklist and improved status polling.

## [5.4.0] - 2024-05-21
### Added
- **Parallel Warp Engine**: Parallelized the Phase Vocoder warping and key-sync pipeline across multiple CPU cores.
- **Logarithmic SA Cooling**: Upgraded the Sequencing engine for superior global optimization.

## [5.2.0] - 2024-05-20
### Added
- **True-Peak Limiter**: Integrated a soft-knee look-ahead limiter into the mastering chain in \`dsp.py\`.

## [5.1.0] - 2024-05-20
### Added
- **Universal Version Synchronization**: Established \`VERSION.md\` as the single source of truth.
- **Directive Consolidation**: Unified all project instructions into \`GLOBAL_LLM_DIRECTIVE.md\`.

## [5.0.0] - 2024-05-19
### Added
- **Interactive Command Console**: Launched the v5.0 Web Dashboard with real-time WebSocket telemetry.
- **WaveSurfer.js Integration**: Added visual waveform rendering to the dashboard.

## [4.0.0] - 2024-05-19
### Added
- **Parallel Processing Pipeline**: Multi-core metadata analysis.
- **Omni-Directive Framework**: Initial unified autonomous agent standards.
