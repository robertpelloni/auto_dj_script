"""
Command Line Interface (CLI) for the Auto DJ script.
Provides a robust set of arguments for controlling the mixing engine.
"""
import argparse
import os
import config
from .version import __version__
from .core import compile_master_set
from .plugins import PluginRegistry

def main():
    # Load modular plugins (v7.7.0)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PluginRegistry.load_plugins(os.path.join(root_dir, "plugins"))

    # Import internal plugins (v8.8.0)
    from . import midi

    """
    Parses command line arguments and initiates the compilation process.
    Supports overrides for all major processing parameters.
    """
    parser = argparse.ArgumentParser(
        description="Auto DJ Script - Automated seamless transitions with frequency splitting, harmonic mixing, and energy profiling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Path Arguments
    parser.add_argument("-i", "--input", default=config.INPUT_FOLDER, help="Input folder containing audio files.")
    parser.add_argument("-o", "--output", default=config.OUTPUT_FILE, help="Output master file path.")

    # Audio Engineering Arguments
    parser.add_argument("-b", "--bpm", type=float, default=config.TARGET_BPM, help="Target BPM (or start BPM if --end-bpm is set).")
    parser.add_argument("--end-bpm", type=float, help="If set, the engine will linearly ramp the tempo from --bpm to this value across the set.")
    parser.add_argument("-d", "--dbfs", type=float, default=config.TARGET_DBFS, help="Target loudness in dBFS (normalization fallback).")

    # DSP Filter Arguments
    parser.add_argument("--lowpass", type=float, default=config.LOWPASS_CUTOFF, help="Low-pass filter cutoff for outgoing track (Hz).")
    parser.add_argument("--highpass", type=float, default=config.HIGHPASS_CUTOFF, help="High-pass filter cutoff for incoming track (Hz).")

    # Transition Arguments
    parser.add_argument("--transition-bars", type=int, default=config.TRANSITION_BARS, help="Number of bars for the crossfade transition.")
    parser.add_argument("--beats-per-bar", type=int, default=config.BEATS_PER_BAR, help="Number of beats per bar (e.g., 4 for 4/4 time).")

    # Mode Arguments
    parser.add_argument("--dry-run", action="store_true", help="Analyze tracks and log details without rendering audio.")
    parser.add_argument("--reorder", action="store_true", help="Intelligently reorder tracks for optimal harmonic compatibility and energy flow.")
    parser.add_argument("--gui", action="store_true", help="Launch the web-based dashboard.")

    args = parser.parse_args()

    if args.gui:
        from .gui import run_gui
        run_gui()
    elif not os.path.exists(args.input):
        os.makedirs(args.input)
        print(f"[INFO] Created directory '{args.input}'. Please place your audio files there and rerun the script.")
    else:
        compile_master_set(args)

if __name__ == "__main__":
    main()
