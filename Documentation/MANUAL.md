# 📖 Auto DJ Script: The Comprehensive User Manual (7.0.0)

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

### ⚡ High-Performance Parallel Engine (7.0.0)
The engine utilizes a multi-core **ProcessPoolExecutor** architecture to parallelize the most intensive tasks:
- **Parallel Metadata Analysis**: Scans your entire library across all available CPU cores simultaneously.
- **Parallel Warp Engine**: Time-stretches and pitch-shifts tracks in parallel, drastically reducing preparation time.
- **Segmented Mixing**: Transitions and track bodies are rendered as independent segments before being stitched into the final master.

### 🔄 Sample-Accurate Cross-Correlation Looping
If a track is too short for a planned transition, the engine identifies rhythmically stable "loopable phrases" using cross-correlation of onset envelopes, ensuring seamless, artifact-free tail extensions.

### 🤖 AI Genre Inference (7.0.0)
The engine utilizes a Multi-Layer Perceptron (MLP) neural classifier to identify stylistic archetypes (Ambient, Techno, House, High-Energy) with probabilistic accuracy. It provides real-time "AI Rationales" explaining the mathematical justification (e.g., spectral centroid) behind each classification.

### 🌐 Distributed Cluster Rendering (7.0.0)
Auto DJ can orchestrate rendering tasks across a cluster of nodes. The Web Dashboard includes a real-time monitor to track node status, core counts, and task distribution, ensuring the engine scales with your hardware.

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
