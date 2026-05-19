# 🏗 Project Directory & Code Structure (v5.1.0)

## Directory Layout
- **\`autodj/\`**: The core Python package.
  - **\`core.py\`**: Orchestration of the mix pipeline and Simulated Annealing.
  - **\`analysis.py\`**: MIR logic (BPM, Key, Energy, Phrasing).
  - **\`dsp.py\`**: Digital Signal Processing (Filters, Normalization, Warping).
  - **\`gui.py\`**: FastAPI/WebSocket web interface.
  - **\`cli.py\`**: Argument parsing and CLI interface.
  - **\`utils.py\`**: Format conversion and helper functions.
  - **\`version.py\`**: Single source of version truth.
- **\`Documentation/\`**: Comprehensive project docs.
- **\`Model Instructions/\`**: Proprietary hints for different LLM agents.
- **\`tests/\`**: Pytest suite for analysis and DSP verification.
- **\`templates/\`**: HTML/JS assets for the Web Dashboard.

## Code Design Patterns
1. **Parallel Pipeline**: Heavy tasks (Analysis/Warping) use \`ProcessPoolExecutor\`.
2. **State Management**: The GUI uses WebSockets to push real-time status updates from the \`Mixer\` class.
3. **NumPy Dominance**: Audio data is treated as high-precision matrices for all mathematical operations.

---
*Outstanding! Simply Extraordinary!*
