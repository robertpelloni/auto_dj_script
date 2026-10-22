# Auto DJ (autodj)

A Python package that assembles DJ mixes automatically: analyze tracks
(BPM/key/energy with librosa), sequence them for harmonic and energy flow,
render transitions with DSP (frequency-split/bass-swap style crossfades,
pitch/tempo adjustment), normalize loudness (pyloudnorm), and export —
including Rekordbox XML playlists. A FastAPI web dashboard (`autodj/gui.py`)
wraps it.

**Version: 8.15.0** (`pyproject.toml` / `VERSION.md`). The README this
replaces advertised 8.11.0 and wore static badges claiming "tests passing"
and an MIT license — there is no LICENSE file in the repo and the badges
were not connected to any CI. Its feature copy ("world-class", "studio-
grade, harmonically perfect", "zero human intervention") was marketing,
not measurement.

## The package

- `autodj/core.py` (~40 KB) — mix assembly engine
- `autodj/analysis.py` (~28 KB) — track analysis (librosa)
- `autodj/dsp.py`, `autodj/np_signal.py` — transition DSP
- `autodj/plugins.py` — plugin system incl. the Rekordbox exporter
  (`tests/test_rekordbox_plugin.py`)
- `autodj/gui.py` + `templates/` — FastAPI dashboard
- `autodj/cli.py`, `autodj/cluster.py`, `autodj/scheduling.py`,
  `autodj/midi.py`, `autodj/monitoring.py`, `autodj/models.py`
- `tests/` — seven small pytest files (analysis, DSP, integration, MIDI,
  Rekordbox, feature tests)
- Root scripts: `make_artist_mixes.py`, `run_artist_mix.py`,
  `run_filtered_mix.py`, `convert_to_mp3.py`, driven by `config.py`;
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
python3 auto_dj.py --gui                                   # dashboard
python3 auto_dj.py --input ./my_music --reorder --bpm 140  # CLI mixdown
python3 make_artist_mixes.py                               # batch artist mixes
```

## Status

Personal tool, developed in bursts through July 2026 (last commit is an
automated working-tree checkpoint). Functional core with tests; packaging
and repo hygiene (outputs in git, no license file) are the rough edges.
