"""Command Line Interface (CLI) for the Auto DJ Script.

Exposes 100% of core engine, transition archetype, mastering, plugin, batch discography,
system tray, and web console options.
"""

import argparse
import os

import config
from .core import compile_master_set
from .dsp import ArchetypeRegistry
from .plugins import PluginRegistry
from .version import __version__


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PluginRegistry.load_plugins(os.path.join(root_dir, "plugins"))

    # Register internal MIDI hardware tool plugin
    from . import midi  # noqa: F401

    archetype_choices = ["auto"] + sorted(ArchetypeRegistry.get_all().keys())
    source_choices = sorted(PluginRegistry.get_sources().keys())
    output_choices = sorted(PluginRegistry.get_outputs().keys())

    parser = argparse.ArgumentParser(
        description="Auto DJ Script — Sample-accurate beat-walked DJ mixing, harmonic sequencing, 3-band mastering, and live console.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Path & Plugin Arguments
    parser.add_argument(
        "-i",
        "--input",
        default=config.INPUT_FOLDER,
        help="Input folder containing audio files.",
    )
    parser.add_argument(
        "-o", "--output", default=config.OUTPUT_FILE, help="Output master file path."
    )
    parser.add_argument(
        "--source-plugin",
        choices=source_choices,
        default="local_folder",
        help="Track discovery source plugin.",
    )
    parser.add_argument(
        "--rekordbox-xml-path",
        default=None,
        help="Path to Pioneer Rekordbox XML file when --source-plugin=rekordbox_xml.",
    )
    parser.add_argument(
        "--output-plugin",
        choices=output_choices,
        default="local_file",
        help="Master mix output sink plugin.",
    )
    parser.add_argument(
        "--stream-url",
        default=None,
        help="Icecast / Shoutcast / RTMP URL for live broadcast streaming.",
    )
    parser.add_argument(
        "--broadcast-mode",
        action="store_true",
        help="Stream output live to --stream-url in addition to archiving locally.",
    )
    parser.add_argument(
        "--export-mp3",
        action="store_true",
        help="Also export a 320kbps MP3 copy alongside the lossless FLAC master.",
    )

    # Audio Engineering & Mastering Arguments
    parser.add_argument(
        "-b",
        "--bpm",
        type=float,
        default=config.TARGET_BPM,
        help="Target BPM (or starting BPM when --end-bpm is set).",
    )
    parser.add_argument(
        "--end-bpm",
        type=float,
        default=None,
        help="Linearly ramp set tempo from --bpm to --end-bpm across the mix.",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=config.TARGET_LUFS,
        help="Target integrated loudness in LUFS (BS.1770 K-weighted).",
    )
    parser.add_argument(
        "-d",
        "--dbfs",
        type=float,
        default=config.TARGET_DBFS,
        help="Legacy target dBFS parameter.",
    )
    parser.add_argument(
        "--mastering-intensity",
        type=float,
        default=0.35,
        help="3-band zero-phase mastering compression intensity (0.0=bypass, 1.0=heavy).",
    )
    parser.add_argument(
        "--no-genre-mastering",
        dest="genre_aware_mastering",
        action="store_false",
        default=True,
        help="Disable genre-aware multiband threshold selection.",
    )
    parser.add_argument(
        "--no-dynamic-energy-mastering",
        dest="dynamic_energy_mastering",
        action="store_false",
        default=True,
        help="Disable RMS-adaptive mastering scaling.",
    )

    # Transition & DSP Filter Arguments
    parser.add_argument(
        "--archetype",
        choices=archetype_choices,
        default="auto",
        help="Transition DSP archetype plugin.",
    )
    parser.add_argument(
        "--bass-transition",
        choices=["swap", "fade"],
        default=getattr(config, "BASS_TRANSITION_STYLE", "swap"),
        help="Low-frequency transition curve: 'swap' (classic DJ kick swap) or 'fade' (gradual equal-power low-end crossfade).",
    )
    parser.add_argument(
        "--transition-bars",
        type=int,
        default=config.TRANSITION_BARS,
        help="Number of musical bars per crossfade transition.",
    )
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        default=config.BEATS_PER_BAR,
        help="Time signature beats per bar (4 for 4/4 electronic music).",
    )
    parser.add_argument(
        "--dynamic-transitions",
        action="store_true",
        default=False,
        help="Adapt transition phrase length (8/16/32 bars) from track energy novelty.",
    )
    parser.add_argument(
        "--adaptive-spectral-balancing",
        action="store_true",
        default=True,
        help="Automatically tame clashing high-frequency overlap during transitions.",
    )
    parser.add_argument(
        "--lowpass",
        type=float,
        default=config.LOWPASS_CUTOFF,
        help="Low-pass filter cutoff for outgoing track (Hz).",
    )
    parser.add_argument(
        "--highpass",
        type=float,
        default=config.HIGHPASS_CUTOFF,
        help="High-pass / bass-crossover cutoff frequency (Hz).",
    )

    # Sequencing, Priority & Execution Mode Arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze tracks and log BPM/Key/Genre table without rendering audio.",
    )
    parser.add_argument(
        "--reorder",
        action="store_true",
        help="Optimize track order via Simulated Annealing harmonic & energy scoring.",
    )
    parser.add_argument(
        "--no-below-normal",
        dest="below_normal_priority",
        action="store_false",
        default=True,
        help="Keep normal OS CPU scheduling priority instead of Below-Normal.",
    )

    # Batch Discography Mode Arguments
    parser.add_argument(
        "--batch-artist",
        default=None,
        help="Scan --library-dir for matching artist/folder tracks and build an artist mix.",
    )
    parser.add_argument(
        "--library-dir",
        default="~/Music",
        help="Root music library path used with --batch-artist.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=15,
        help="Maximum tracks to stage when running --batch-artist.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="With --batch-artist, preview detected BPM/Key matches without rendering.",
    )

    # Web Console & System Tray Arguments
    parser.add_argument(
        "--gui", action="store_true", help="Launch the FastAPI Web Command Console."
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host for --gui server."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind port for --gui server."
    )
    parser.add_argument(
        "--no-tray",
        dest="enable_tray",
        action="store_false",
        default=True,
        help="Disable the Windows System Tray controller icon when running --gui.",
    )

    args = parser.parse_args()

    if args.gui:
        from .gui import run_gui

        run_gui(host=args.host, port=args.port, enable_tray=args.enable_tray)
    elif args.batch_artist is not None:
        from .batch import run_batch_mix, scan_library_tracks

        if args.scan_only:
            res = scan_library_tracks(
                library_dir=args.library_dir,
                artist_query=args.batch_artist,
                target_bpm=args.bpm,
                max_tracks=args.max_tracks,
            )
            print(
                f"Scanned {res['scanned_count']} tracks | Selected {len(res['selected'])} near {res['target_bpm']} BPM:"
            )
            for item in res["selected"]:
                print(f"  {item['bpm']:5.1f} BPM | {item['key']:16s} | {item['file']}")
        else:
            res = run_batch_mix(
                artist_query=args.batch_artist,
                library_dir=args.library_dir,
                target_bpm=args.bpm,
                end_bpm=args.end_bpm,
                max_tracks=args.max_tracks,
                bass_transition=args.bass_transition,
                archetype=args.archetype,
                reorder=args.reorder,
                export_mp3=args.export_mp3,
            )
            print(
                f"[*] Batch mix complete: {res['output_file']} ({res['duration_min']} min)"
            )
    elif not os.path.exists(args.input) and args.source_plugin == "local_folder":
        os.makedirs(args.input, exist_ok=True)
        print(
            f"[INFO] Created directory '{args.input}'. Place audio files there and rerun."
        )
    else:
        compile_master_set(args)


if __name__ == "__main__":
    main()
