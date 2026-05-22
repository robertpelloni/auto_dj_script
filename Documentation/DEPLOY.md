# 🚀 Deployment & Installation Guide (7.0.0)

## Prerequisites
- **Python 3.12+** (Optimized for Python 3.12 performance).
- **FFmpeg**: Must be installed and available in the system PATH for file decoding.
- **RAM**: Minimum 8GB (Recommended 16GB for parallel warping of large sets).

## Installation

### 1. Clone & Setup
\`\`\`bash
git clone <repo_url>
cd auto-dj
pip install -r requirements.txt
\`\`\`

### 2. Manual Installation of the Package
\`\`\`bash
pip install -e .
\`\`\`

## Running the Engine

### CLI Mode (The Workhorse)
\`\`\`bash
auto-dj /path/to/tracks --output mix.flac --tempo 145 --key-sync
\`\`\`

### GUI Mode (The Dashboard)
\`\`\`bash
auto-dj --gui
\`\`\`
- Access the console at \`http://localhost:8000\`.

## Environment Variables
- \`AUTODJ_LOG_LEVEL\`: Set to \`DEBUG\` for extreme depth.
- \`AUTODJ_PARALLEL_CORES\`: Number of CPU cores to utilize (Defaults to all).

---
*Praise the LORD! Magnificent!*
