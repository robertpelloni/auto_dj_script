"""
Configuration settings for the Auto DJ Script.
"""

# Audio Processing Standards
TARGET_BPM = 145.0
TARGET_LUFS = -14.0  # Industry standard for streaming (Spotify/YouTube)
TARGET_DBFS = -14.0  # Legacy peak-based normalization fallback

# Folder Paths
INPUT_FOLDER = "./psyset_source"
OUTPUT_FILE = "./final_dj_master.flac"

# Supported Audio Extensions
SUPPORTED_EXTENSIONS = [".flac", ".wav", ".mp3", ".aiff", ".m4a"]

# DSP Filter Settings
LOWPASS_CUTOFF = 200.0   # Frequency for outgoing track taper
HIGHPASS_CUTOFF = 150.0  # Frequency for incoming track bass-strip

# Transition Logic
TRANSITION_BARS = 16
BEATS_PER_BAR = 4
PHRASE_ANCHOR_TOLERANCE_MS = 2000 # 2 seconds tolerance for smart phrase detection

# Optimization Parameters
SA_ITERATIONS = 500
SA_COOLING_RATE = 0.99
SA_INITIAL_TEMP = 100.0
