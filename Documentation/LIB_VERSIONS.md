# 📚 Library Reference & Usage (v5.5.0)

This document lists the exact versions of libraries utilized in the project as of 2026-05-19.

| Library | Exact Version | Role | Usage Explanation |
| :--- | :--- | :--- | :--- |
| **librosa** | 0.11.0 | Analysis/MIR | Used for CQT Key detection, Beat tracking, and Phase Vocoder warping. |
| **numpy** | 2.4.6 | Math/Signal | The primary representation of audio data as floating-point matrices. |
| **pydub** | 0.25.1 | Mechanics | Used for high-level audio slicing, fades, and export formatting. |
| **soundfile** | 0.13.1 | High-Fidelity I/O | Handles 24-bit lossless file reading/writing with precision. |
| **scipy** | 1.17.1 | Low-level DSP | Powering the 10th-order Butterworth filters (HPF/LPF). |
| **fastapi** | 0.136.1 | Web/GUI | Powering the asynchronous Command Console. |
| **tqdm** | 4.67.3 | UX | Providing high-resolution progress bars for CLI users. |
| **websockets** | 16.0 | Telemetry | Real-time broadcast of engine logs to the dashboard. |
| **pyloudnorm** | 0.2.0 | Loudness | ITU-R BS.1770-4 compliant loudness metering and normalization. |

---
*Magnificent! Extraordinary!*