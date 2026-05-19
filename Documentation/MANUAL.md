# 📖 Auto DJ Script: The Comprehensive User Manual

## 1. Introduction
The Auto DJ Script is a professional-grade, autonomous audio mixing engine designed for DJs, broadcasters, and audiophiles. It leverages state-of-the-art Music Information Retrieval (MIR) and Digital Signal Processing (DSP) to create seamless, harmonically perfect sets.

## 2. Quick Start
### Installation
Ensure you have Python 3.8+ and FFmpeg installed.
```bash
pip install -r requirements.txt
```

### Basic Command
```bash
python3 auto_dj.py --input ./my_music --output final_mix.flac --reorder
```

## 3. Advanced Features

### 🌍 Global Sequence Optimization (`--reorder`)
Instead of mixing tracks in alphabetical order, the engine uses **Simulated Annealing** to analyze your entire library. It calculates thousands of permutations to find the one that minimizes harmonic dissonance and creates a smooth energy arc.

### 🎹 Harmonic Key Sync
If two adjacent tracks are in different but close keys (within 2 semitones), the engine automatically shifts the pitch of the incoming track to ensure a perfect harmonic match.

### 🧠 Smart Phrase Detection
The engine identifies "structural boundaries" (drops, breakdowns) using spectral novelty detection. Transitions are anchored to these points rather than arbitrary bar counts, ensuring a musically intuitive flow.

### 💎 Professional Mastering (LUFS)
Sets are normalized to **-14.0 LUFS**, the global standard for streaming platforms. This ensures that your 5-hour set maintains consistent perceived loudness without digital clipping.

## 4. Command Line Interface (CLI)

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `./psyset_source` | Source folder for audio files. |
| `--bpm` | `145.0` | Starting tempo. |
| `--end-bpm` | Same as bpm | Ending tempo (linear ramp). |
| `--reorder` | `False` | Enable global sequence optimization. |
| `--dry-run` | `False` | Analysis-only mode (no rendering). |
| `--gui` | `False` | Launch the Web Dashboard. |

## 5. Web Dashboard
Run `python3 auto_dj.py --gui` and navigate to `http://localhost:8000`.
- **Monitor Progress**: Watch real-time status as the engine warps and mixes.
- **Visual Tracklist**: View the optimized sequence and transition timestamps.
- **Download**: Retrieve your finished master directly from the browser.

## 6. Philosophy of the Mix
The Auto DJ Script believes in **transparency**. By using 10th-order Butterworth filters to split frequencies during transitions, we ensure that the listener never hears "clashing basslines" or "muddy mids." It is the sound of a studio engineer working in real-time.

---
*Magnificent! Extraordinary! The Party Never Stops.*
