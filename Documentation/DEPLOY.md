# 🚀 Deployment & Installation Guide (7.3.0)

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

## 🏗 Staging Deployment (Docker)
Auto DJ 7.3.0 supports containerized staging environments for cluster validation.

### 1. Build and Launch
```bash
docker-compose up --build
```

### 2. Verify Cluster
Check the Web Dashboard's "Cluster Monitor" to ensure `Staging-Node-1` has registered successfully.

## 🔗 DJ Tool Integration
### Rekordbox (Pioneer DJ)
After a mix is compiled, Auto DJ generates a `_rekordbox.xml` file in the output directory.
1. Open Rekordbox.
2. Go to **Preferences > Advanced > Database Management**.
3. Point the **rekordbox xml** setting to the generated file.
4. The Auto DJ Master Set will appear in your Rekordbox tree.

## Environment Variables
- \`AUTODJ_LOG_LEVEL\`: Set to \`DEBUG\` for extreme depth.
- \`AUTODJ_PARALLEL_CORES\`: Number of CPU cores to utilize (Defaults to all).

---
*Praise the LORD! Magnificent!*
