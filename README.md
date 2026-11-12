# Auto DJ (autodj)

A Python package that assembles DJ mixes automatically: analyze tracks
(BPM/key/energy with pure-NumPy spectral flux + librosa), sequence them for
harmonic and energy flow, render transitions with DSP (zero-phase `filtfilt`
bass-swap or gradual equal-power bass crossfades, pitch/tempo adjustment),
normalize loudness (chunked ITU-R BS.1770 LUFS), and export — including
Rekordbox XML playlists. A FastAPI web dashboard (`autodj/gui.py`) wraps it.

**Version: 8.16.0** (`pyproject.toml` / `VERSION.md`). The earlier README
advertised 8.11.0 and wore static badges claiming "tests passing" and an MIT
license — there is no LICENSE file in the repo and the badges were not
connected to any CI. Its feature copy ("world-class", "studio-grade,
harmonically perfect", "zero human intervention") was marketing, not
measurement.

## The package

- `autodj/core.py` (~42 KB) — mix assembly engine, equal-power crossfades,
  `--bass-transition [swap|fade]` low-end curves, and `SmartReplenishTool`
- `autodj/analysis.py` (~33 KB) — pure-NumPy 4-band spectral flux BPM detector
  with jitter-gated kick-interval (IOI) octave verification, key/Camelot wheel,
  and phrase analysis
- `autodj/dsp.py`, `autodj/np_signal.py` — zero-phase crossover DSP, chunked
  BS.1770 LUFS normalization, limiter, and `SpectralBalancedMix` archetype
- `autodj/plugins.py` — plugin system incl. >4 GB streaming FLAC output and
  Rekordbox XML exporter (`tests/test_rekordbox_plugin.py`)
- `autodj/gui.py` + `templates/` — FastAPI dashboard
- `autodj/cli.py`, `autodj/cluster.py`, `autodj/scheduling.py`,
  `autodj/midi.py`, `autodj/monitoring.py`, `autodj/models.py`
- `tests/` — seven pytest modules (15 tests covering analysis, kick-IOI BPM,
  DSP, integration, MIDI, Rekordbox, spectral clash, and scheduler)
- Root scripts: `start.bat`, `build.bat`, `make_artist_mixes.py`,
  `run_artist_mix.py`, `run_filtered_mix.py`, `mix_all_artists.sh`,
  `scan_ajja.py`, `select_ajja.py`, `final_select.py`, `verify_bpm.py`,
  `run_ajja_modern.py`, `convert_to_mp3.py`, driven by `config.py`;
  `auto_dj.py` is a thin wrapper over the package. `Dockerfile` +
  `docker-compose.yml` included. Docs in `Documentation/`.

Dependencies (`requirements.txt`): librosa, numpy, scipy, pydub,
soundfile, pyloudnorm, fastapi, uvicorn, jinja2. FFmpeg is required by
pydub.

## The root directory is also the output folder

Most of the repo root is *generated output from real runs*: per-artist
Rekordbox XML + tracklist pairs (Ajja, Astrix, Avalon, Cosmosis, 1200
Micrograms, Space Tribe, Tristan, Koxbox, Mad Tribe, GMS family, Electric
Universe, Mekkanikka, Outsiders, classic/modern psytrance sets, a
`final_dj_master_*` set), dashboard screenshots — including error
screenshots (`v8_11_error.png`, `v8_13_version_error.png`) — and GUI logs.
So the tool demonstrably ran and produced mixes; it also demonstrably
threw errors that were screenshotted and committed alongside the wins.

## Usage

```bash
pip install -r requirements.txt
python auto_dj.py --gui                                                        # dashboard (or start.bat)
python auto_dj.py --input ./my_music --reorder --bpm 145 --bass-transition fade # CLI mixdown
python make_artist_mixes.py                                                    # batch artist mixes
build.bat                                                                      # compile + pytest suite
```

## Status

Personal tool, actively maintained through November 2026 (v8.16.0). Functional
core with 15 passing pytest tests; raw/rendered `.flac` audio binaries stay
local while tracklists, Rekordbox XMLs, scripts, and session/database metadata
are tracked in git.
