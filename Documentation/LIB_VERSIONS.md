# 📦 Library & Dependency Inventory (v6.2.0)

This document tracks all external libraries used in the Auto DJ Script, their versions, and their specific roles in the ecosystem.

| Name | Version | Location | Purpose | Relationship to Project |
| :--- | :--- | :--- | :--- | :--- |
| **librosa** | 0.11.0 | MIR / Analysis | Audio feature extraction (BPM, Key, MFCC, CQT). | The "Brain" of the engine; used in `autodj/analysis.py`. |
| **numpy** | 2.4.6 | Core DSP | High-performance array manipulation. | Essential for all DSP operations and float-domain processing. |
| **pydub** | 0.25.1 | Mechanics | High-level audio slicing and crossfading. | Orchestrates the physical assembly of the master mix in `autodj/core.py`. |
| **soundfile** | 0.13.1 | I/O | High-fidelity audio reading/writing (24-bit/32-bit). | Handles lossless file exports in `autodj/core.py`. |
| **scipy** | 1.17.1 | DSP | Low-level signal processing (Butterworth filters). | Implements the surgical 10th-order filters in `autodj/dsp.py`. |
| **fastapi** | 0.136.1 | Web / UI | Async web framework for the dashboard. | Power the Command Console in `autodj/gui.py`. |
| **uvicorn** | 0.47.0 | Server | ASGI server for FastAPI. | Entry point for the web interface. |
| **websockets** | 16.0 | Telemetry | Real-time bidirectional communication. | Streams mixing progress and tracklist updates to the frontend. |
| **pyloudnorm** | 0.2.0 | Mastering | BS.1770-4 loudness normalization. | Ensures perception-based volume consistency across the mix. |
| **tqdm** | 4.67.3 | UX | CLI Progress bars. | Provides visual feedback during the analysis/warp phases. |
| **jinja2** | 3.1.6 | UI | HTML templating. | Renders the dashboard UI from `templates/index.html`. |
| **python-multipart** | 0.0.29 | Web | Form parsing. | Handles user input from the dashboard. |

---
*Last Audit: v6.2.0 Cycle Start.*
