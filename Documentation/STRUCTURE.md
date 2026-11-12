# 🏗 Project Directory, Code Structure & Structural Map (8.16.0)

## Repository & Submodule Structural Map

| Component / Module | Location | Remote Source URL | Branch / Tracking | Submodules |
|---|---|---|---|---|
| **Root Repository (`auto_dj_script`)** | `./` | `https://github.com/robertpelloni/auto_dj_script` | `main` (`v8.16.0`, standalone root, not a fork) | None (`0` nested `.gitmodules`) |

> Verified on `2026-11-12`: `auto_dj_script` has no upstream fork parent and zero nested git submodules. All historical pull request refs (`origin/pr/1`–`origin/pr/5`) are merged into `main`.

## Directory Layout

- **`autodj/`**: Core Python package
  - **`core.py`**: Mix assembly pipeline, Simulated Annealing sequence optimizer, equal-power crossfades, `--bass-transition [swap|fade]` low-end curves, and `SmartReplenishTool`.
  - **`analysis.py`**: Pure-NumPy 4-band spectral flux BPM engine, jitter-gated kick-IOI octave verification (`_kick_interval_bpm`), key/Camelot wheel mapping, energy profiling, and phrase detection.
  - **`dsp.py`**: Zero-phase `filtfilt` bass crossover, chunked ITU-R BS.1770 LUFS loudness normalization, soft-knee peak limiter, `calculate_spectral_clash`, and `SpectralBalancedMix`.
  - **`np_signal.py`**: Pure-NumPy/SciPy Butterworth filter design and IIR helpers.
  - **`plugins.py`**: `SourcePlugin`, `OutputPlugin` (including >4 GB streaming `soundfile` FLAC writer and Rekordbox XML exporter), and `ToolPlugin` registry.
  - **`gui.py` + `templates/index.html`**: FastAPI + WebSocket Command Console dashboard.
  - **`cli.py`**: CLI argument parser (`auto-dj` / `python auto_dj.py`).
  - **`cluster.py`**, **`scheduling.py`**, **`midi.py`**, **`monitoring.py`**, **`models.py`**, **`utils.py`**, **`version.py`**.
- **`tests/`**: Pytest suite (7 modules, 15 tests) covering analysis, kick-IOI BPM, DSP filters, Rekordbox XML export, MIDI mapping, spectral clash detection, and scheduling.
- **Root Execution & Batch Scripts**:
  - **`start.bat`** / **`build.bat`**: Windows launcher (GUI / CLI pass-through) and build/test verification script.
  - **`auto_dj.py`**, **`config.py`**, **`convert_to_mp3.py`**: Primary CLI wrapper, configuration constants, and MP3 transcoder.
  - **`make_artist_mixes.py`**, **`run_artist_mix.py`**, **`run_filtered_mix.py`**, **`mix_all_artists.sh`**: Multi-artist and BPM-filtered batch compilation pipelines targeting `artist_mixes/`.
  - **`scan_ajja.py`**, **`select_ajja.py`**, **`final_select.py`**, **`verify_bpm.py`**, **`kick_count.py`**, **`tempo_diag.py`**, **`run_ajja_modern.py`**: Uptempo (>130 BPM) catalogue discovery, kick-interval tempo verification, staging, and rendering suite.
- **`artist_mixes/`** & **Root `*_rekordbox.xml` / `*_tracklist.txt`**: Exported Pioneer Rekordbox XML cue files and timestamped tracklists (raw `.flac` audio binaries remain gitignored locally).
- **`Documentation/`**: Architecture, changelog, roadmap, todo, memory, manual, deployment, and handoff documentation.
- **`.pi/`**, **`.pi-lens/`**, **`.tormentnexus/`**, **`data/lancedb/`**, **`*.db`**: Tracked agent memory blocks, session logs, and non-sensitive SQLite/LanceDB state (`borg.db`, `metamcp.db`, `tormentnexus.db`).

## Code Design Patterns

1. **Zero-Phase Subtractive Crossover**: Extract low frequencies via `scipy.signal.filtfilt` and subtract `bass * (1 - gain_curve)` so unity gain reconstructs the exact original signal with zero group-delay phase bumps.
2. **Pure-NumPy Tempo Resilience**: Spectral flux + kick-band IOI verification operates without `librosa.onset`/`numba`, staying compatible with Python 3.14+ and NumPy 2.5+.
3. **Streaming Lossless I/O**: Chunked BS.1770 LUFS measurement and direct `soundfile` FLAC writing support 7+ hour lossless mixes (>2.9 GB FLAC / >8 GB PCM) in ~120 MB RAM.
