# Changelog

## [8.11.0] - 2025-01-24
### Added
- **Preference-Aware Orchestration**: Enhanced `SmartReplenishTool` with harmonic scoring and user preference integration (Genre/Energy).
- **Expanded Scheduler API**: Added `set_preference`, `clear_queue`, and `trigger_replenish` actions.
- **Dynamic Bias Controls**: Real-time sliders for Energy Bias and Genre Preference in the Command Console.
- **Scheduler History UI**: Persistent reverse-chronological log of executed and pending events.
- **Heuristic Harmonic Selection**: Automated queue replenishment now prioritizes tracks with matching keys and energy profiles.

## [8.8.0] - 2025-01-24
### Added
- **Tactile MIDI Hardware Integration**: Introduced `autodj/midi.py` with threaded `MidiHandler` supporting real-time CC and Note mapping.
- **Rekordbox Library Ingestion**: Added `RekordboxSourcePlugin` to parse Pioneer `pioneer.xml` files with full URL-encoded path resolution.
- **Hardware Telemetry Dashboard**: New "HARDWARE & MIDI" panel in the Command Console with device discovery and live activity monitoring.
- **Hardware-Informed Mastering**: Live parameter overrides for EQ gains and Mastering Intensity via MIDI CC 16-19.

## [8.7.0] - 2025-01-24
### Added
- **Quantum Sequence Optimizer**: Upgraded Simulated Annealing to run 4 parallel cooling branches for superior global set-flow optimization.
- **Visual Sync Waveforms**: Integrated Hot Cue markers and phrase boundaries directly into the WaveSurfer.js waveform display.
- **3D Beat-Grid Overlay**: Added rhythmic lattice visualization on top of the Spectral Terrain v1.0 engine.
- **Session Archiving**: Automatic backup of mix metadata, energy profiles, and console logs into versioned `.json` archives.

## [7.8.0] - 2024-06-01
### Added
- **Advanced Monitoring Dashboard**: Comprehensive system-wide health telemetry and resource charting.
- **Disk I/O & Network Telemetry**: Integrated real-time tracking of disk and network throughput using `psutil`.
- **Real-time Resource Charting**: Integrated Chart.js for visual health trends in the Command Console.
- **Granular Task Tracking**: Detailed progress display for background analysis and mixing jobs.
- **Health Guardrail Indicators**: Visual alerts and status pills for system throttling and resource exhaustion.

## [7.7.0] - 2024-05-31
### Added
- **Modular Plugin System**: Introduced a unified framework for `SourcePlugin`, `OutputPlugin`, and `ToolPlugin`.
- **Dynamic Plugin Loading**: Automatic registration of external modules from the `plugins/` directory.
- **Refactored Core Pipeline**: Abstracted track discovery and mix export into modular components.
- **Tool Hooks**: Added execution hooks (`pre_mix`, `post_mix`, `on_track_start`) for utility integration.
- **UI Plugin Management**: Real-time selection of Input Sources and Output Sinks via the Web Dashboard.

## [7.6.0] - 2024-05-30
### Added
- **Spectral Terrain 3D Visualizer**: Integrated Three.js for real-time WebGL rendering of Mel-Spectrogram maps.
- **Energy-Reactive Heightmaps**: Audio energy is now mapped to 3D topology with fly-over navigation and pulsing emissive shaders.
- **Visual-to-Audio Sync**: The visualizer state is driven by the internal MIR pipeline's terrain data.
- **UI Mode Toggle**: Seamlessly switch between the 3D performance view and the standard command console.

## [7.5.0] - 2024-05-30
### Added
- **Live Deck Playlist Manager**: Real-time track selection and session building.
- **Dynamic Track Injection**: Support for adding tracks to the queue while a mix is rendering.
- **Just-in-Time Warping**: On-the-fly analysis and time-stretching for late-injected tracks.
- **Live Deck UI**: Two-pane track manager for folder browsing and queue manipulation.

## [7.4.0] - 2024-05-29
### Added
- **Fault-Tolerant Mixing Engine**: Implemented "Retry-with-Fallback" strategy for cluster tasks.
- **Incident Recovery Console**: Real-time robustness telemetry and structured incident logging.
- **Node-Specific Metrics**: Tracking success/failure rates per cluster node.
- **Auto-Recovery UI**: Added manual node reset and error rate visualization.

## [7.3.0] - 2024-05-29
### Added
- **Integration Bridge (Rekordbox)**: Automated generation of Pioneer DJ compatible XML files with metadata and transition cue points.
- **Docker Staging Infrastructure**: Full containerization support for master and cluster node environments.
- **Integration Test Suite**: New validation layer for output compatibility and XML integrity.

## [7.2.0] - 2024-05-29
### Added
- **Automated Health Guardrails**: The engine now automatically throttles or pauses processing if CPU or RAM exceeds safety thresholds (85% and 90% respectively).
- **Manual Pause/Resume**: Added full execution control to the Web Dashboard.
- **Dynamic BPM Overrides**: Real-time adjustment of target BPM during the active mixing cycle.
- **Enhanced Telemetry UI**: Color-coded system load alerts and color-shifting telemetry cards.

## [7.1.0] - 2024-05-28
### Added
- **System Telemetry Module**: Integrated `psutil` for real-time tracking of CPU, RAM, and active threads.
- **Live Parameter Controls**: Added sliders to the Command Console for real-time Mastering Intensity and BPM adjustments.
- **Engine Hot-Reload**: Refactored the core loop to poll live parameters during each track rendering phase.

## [7.0.0] - 2024-05-28
### Added
- **Quantum Network Cluster**: Implemented remote node registration and orchestration via `/cluster/join`.
- **Spectral Terrain Data Pipeline**: High-resolution energy map extraction for 3D visualization.
- **In-Memory Audio I/O**: Refactored the internal pipeline to use `io.BytesIO`, eliminating temporary file overhead.
### Performance
- Reduced track preparation latency by ~15% via memory-buffer optimization.

## [6.9.0] - 2024-05-27
### Added
- **Distributed Cluster Rendering**: Introduced `autodj/cluster.py` for orchestrating rendering tasks across multiple nodes.
- **Persistent Cluster Executor**: Optimized the mixing pipeline to use node-aware process pools.
- **Cluster Monitor UI**: Real-time status display of available rendering nodes in the Command Console.
- **Asynchronous Task Dispatching**: Refactored warping and mixing to support non-blocking cluster distribution.

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
- **True-Peak Limiter**: Integrated a soft-knee look-ahead limiter into the mastering chain in `dsp.py`.

## [5.1.0] - 2024-05-20
### Added
- **Universal Version Synchronization**: Established `VERSION.md` as the single source of truth.
- **Directive Consolidation**: Unified all project instructions into `GLOBAL_LLM_DIRECTIVE.md`.

## [5.0.0] - 2024-05-19
### Added
- **Interactive Command Console**: Launched the v5.0 Web Dashboard with real-time WebSocket telemetry.
- **WaveSurfer.js Integration**: Added visual waveform rendering to the dashboard.

## [4.0.0] - 2024-05-19
### Added
- **Parallel Processing Pipeline**: Multi-core metadata analysis.
- **Omni-Directive Framework**: Initial unified autonomous agent standards.
