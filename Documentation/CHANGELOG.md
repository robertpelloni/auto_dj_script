# Changelog

## [6.8.0] - 2024-05-27
### Added
- **AI Genre Inference Engine**: Implemented an MLP-based classification model in `autodj/models.py`.
- **Probabilistic Stylistic Activation**: Transitions are now guided by probabilistic genre maps (Ambient, Techno, House, High-Energy).
- **MIR Rationale Logic**: The engine now provides mathematical explanations for its genre choices in the Web Dashboard.
- **UI Enhancements**: Added an "AI Rationale" column to the live tracklist.

## [6.7.0] - 2024-05-27
### Added
- **High-Performance Parallel Engine**: Transitioned metadata analysis and audio warping to a multi-core `ProcessPoolExecutor` architecture.
- **Sample-Accurate Cross-Correlation Looping**: Refactored `identify_loopable_phrase` using cross-correlation to ensure artifact-free tail extensions.
- **AI Genre Inference Pipeline**: Integrated high-dimensional spectral feature extraction as a foundation for the upcoming CNN-based classification.
- **Segmented Mixing Engine**: The final mix is now assembled from pre-rendered parallelized track segments.

## [6.6.0] - 2024-05-26
### Added
- **Intelligent Phrase Looping**: Automatically identifies and loops rhythmically compatible phrases to extend short track outros/intros.
- **Tail Extension UI**: Added "Auto-Looping" status indicators to the live tracklist.
- **Rhythmic Similarity Analysis**: New MIR module for detecting seamless loop points.

## [6.5.0] - 2024-05-26
### Added
- **Segmented Parallel Mixing Engine**: Transitions and track bodies are now rendered in parallel across multi-core processors.
- **Mix Stitching Architecture**: High-speed assembly of pre-rendered audio chunks for sub-second master set reconstruction.
### Performance
- Reduced total mixing time by ~40% for sets with >10 tracks.

## [6.4.0] - 2024-05-26
### Added
- **Adaptive Spectral Balancing**: Real-time frequency clash detection and automatic gain compensation during transitions.
- **Dynamic Energy Mastering**: Automatic scaling of mastering intensity based on track energy profiles to prevent "ear fatigue" in high-energy sets.
- **Spectral UI**: Real-time clash detection status integrated into the Command Console telemetry.

## [6.3.0] - 2024-05-26
### Added
- **Live Broadcast Client**: Integrated FFmpeg-based RTMP/Icecast streaming capability.
- **Broadcast UI**: Added "Live Broadcast Mode" and Stream URL input to the Command Console.

## [6.2.0] - 2024-05-26
### Added
- **AI-Enhanced Genre Classification (v3)**: Integrated MFCC and Spectral Contrast for superior stylistic detection.
- **Genre-Aware Mastering Profiles**: Automatic dynamics processing (Thresholds/Makeup Gain) tailored to the specific genre (Ambient, Techno, House, High-Energy).
- **GUI Controls**: Added toggle for AI Genre-Aware Mastering in the Command Console.
- **Library Inventory**: Added `Documentation/LIB_VERSIONS.md` for dependency tracking.

## [6.1.0] - 2024-05-26
### Added
- **Intelligent Transition Selector**: The engine now chooses transition archetypes based on energy dynamics (e.g., choosing Echo-Out for energy drops, Bass-Swap for energy matches).
- **Auto-Gain Compensation**: Integrated make-up gain into the 3-band compressor to maintain consistent output volume regardless of compression depth.
- **AI Rationale Display**: The Web Dashboard now displays the mathematical rationale behind each automatic transition choice.

## [6.0.0] - 2024-05-25
### Added
- **Plugin-based Archetype Architecture**: Transition archetypes are now modular plugins, allowing for easy expansion and dynamic loading.
- **Low-Cut Build Archetype**: Added a new transition style that applies tension using high-pass filters on both tracks.
- **Dynamic Transition Dropdown**: The Web Dashboard now automatically lists all registered transition plugins.
### Refactored
- Ported existing transition logic (Bass-Swap, Echo-Out, HPF-Sweep, Classic) to the new Plugin framework in `dsp.py`.

## [5.9.0] - 2024-05-24
### Added
- **Phrase-Aware Dynamic Transitions**: The engine now analyzes phrase boundaries to automatically calculate the optimal transition length (8, 16, or 32 bars).
- **Dynamic Transition UI**: Added a toggle to the Web Dashboard and integrated transition duration metadata into the live tracklist.
- **Repository Synchronization**: Merged upstream changes and synchronized feature branches for unified development flow.

## [5.8.0] - 2024-05-23
### Added
- **Enhanced Genre Inference (v2)**: Incorporated BPM and spectral flatness into the classification heuristic for superior stylistic detection.
- **3-Band Multi-band Compression**: Upgraded the mastering chain from 2-band to 3-band (Low, Mid, High) for professional-grade dynamics control.
- **Mastering Intensity Control**: Added a real-time slider to the Web Dashboard to scale compression thresholds.

## [5.7.0] - 2024-05-22
### Added
- **Interactive Tempo Ramping**: Added "End BPM" control to the Web Dashboard for set-wide tempo evolution.
### Fixed
- **BPM Analysis Signature**: Resolved a critical argument mismatch in `get_native_bpm` that prevented track analysis.
- **Dependency Sync**: Synchronized `pyproject.toml` with `requirements.txt`.

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
