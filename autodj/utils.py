"""Utility functions for audio conversion, process priority, MP3 transcoding, and Rekordbox XML export.

Theoretical Foundations:
1. Pydub interoperability: Pydub uses signed 16-bit PCM for internal storage.
2. Normalization: We convert to float32 range [-1.0, 1.0] for high-precision DSP.
3. Process Priority Guardrail: Heavy DSP/time-stretching tasks default to Below-Normal OS
   scheduling priority so long multi-hour renders never starve interactive desktop workloads.
"""

import os
import numpy as np
from pydub import AudioSegment
from .version import __version__


def set_below_normal_priority() -> str:
    """Lowers the current process CPU scheduling priority to Below Normal.

    Why:
    Multi-core Rubber Band / librosa phase-vocoder warping and zero-phase `filtfilt`
    passes saturate all FPU lanes. Setting `BELOW_NORMAL_PRIORITY_CLASS` on Windows
    (or `nice(10)` on POSIX) keeps UI responsiveness intact without throttling
    throughput when the system is otherwise idle.
    """
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        if os.name == "nt" and hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return "below_normal"
        if hasattr(os, "nice"):
            os.nice(10)
            return "nice_10"
    except Exception:
        pass
    return "normal"


def pydub_to_ndarray(segment):
    """Converts a Pydub AudioSegment into a normalized float32 stereo NumPy array `(2, N)`."""
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels == 2:
        samples = samples.reshape((-1, 2)).T
    else:
        samples = np.vstack([samples, samples])

    samples /= 2**15
    return samples


def ndarray_to_pydub(audio_array, sr):
    """Converts a normalized float32 NumPy array back into a signed 16-bit Pydub AudioSegment."""
    clipped = np.clip(audio_array, -1.0, 1.0)
    pcm = (clipped * (2**15 - 1)).astype(np.int16)

    if pcm.ndim == 2:
        interleaved = pcm.T.flatten()
        return AudioSegment(
            interleaved.tobytes(), frame_rate=sr, sample_width=2, channels=2
        )
    return AudioSegment(pcm.tobytes(), frame_rate=sr, sample_width=2, channels=1)


def get_track_duration(file_path):
    """Retrieves the duration of an audio file in seconds using `soundfile` (fast header read) with `librosa` fallback."""
    try:
        import soundfile as sf

        info = sf.info(file_path)
        if info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    import librosa

    return float(librosa.get_duration(path=file_path))


def transcode_to_mp3(source_path: str, mp3_path: str | None = None, bitrate: str = "320k") -> str:
    """Transcodes a rendered lossless mix (e.g., FLAC/WAV) to CBR MP3 via FFmpeg/Pydub."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source audio file not found: {source_path}")
    if mp3_path is None:
        mp3_path = os.path.splitext(source_path)[0] + ".mp3"
    audio = AudioSegment.from_file(source_path)
    audio.export(mp3_path, format="mp3", bitrate=bitrate)
    return mp3_path


def export_rekordbox_xml(tracklist, output_xml_path):
    """Generates a Pioneer Rekordbox-compatible XML playlist with transition hot-cues (`POSITION_MARK`).

    Emits:
    - Cue 0 (`Transition Start`): Exact timestamp in seconds where the incoming track enters the master mix.
    - Cue 1 (`Full Body Lock`): Timestamp where the overlap crossfade completes (when `overlap_ms` is present).
    - Cue 2 (`Outgoing Handoff`): Timestamp where the next track begins overlapping (when `next_start_ms` is present).
    """
    import urllib.parse
    import xml.etree.ElementTree as ET

    dj_playlists = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(
        dj_playlists,
        "PRODUCT",
        Name="AutoDJ",
        Version=str(__version__),
        Company="robertpelloni",
    )
    collection = ET.SubElement(
        dj_playlists, "COLLECTION", Entries=str(len(tracklist))
    )

    for i, item in enumerate(tracklist):
        raw_path = item.get("path", "")
        if raw_path:
            norm_path = os.path.abspath(raw_path).replace("\\", "/")
            if not norm_path.startswith("/"):
                norm_path = "/" + norm_path
            location = "file://localhost" + urllib.parse.quote(norm_path, safe="/:")
        else:
            location = ""

        start_sec = float(item.get("start_ms", 0)) / 1000.0
        overlap_ms = float(item.get("overlap_ms", 0))
        duration_ms = float(item.get("duration_ms", 0))

        track = ET.SubElement(
            collection,
            "TRACK",
            TrackID=str(i + 1),
            Name=str(item.get("file", f"Track {i + 1}")),
            Artist=str(item.get("artist", "Auto DJ")),
            Genre=str(item.get("genre", "Electronic")),
            Kind="FLAC File",
            Size="0",
            TotalTime=str(int(duration_ms / 1000.0)),
            Location=location,
            AverageBpm=f"{float(item.get('bpm', 0) or 0):.2f}",
            Tonality=str(item.get("key", "Unknown")),
        )

        # Hot Cue 0: Entry / Transition Start in Master Mix
        ET.SubElement(
            track,
            "POSITION_MARK",
            Name="Transition Start",
            Type="0",
            Start=f"{start_sec:.3f}",
            Num="0",
        )

        # Hot Cue 1: Overlap Completion / Bass Lock
        if overlap_ms > 0:
            ET.SubElement(
                track,
                "POSITION_MARK",
                Name="Bass Lock (Crossfade End)",
                Type="0",
                Start=f"{(start_sec + overlap_ms / 1000.0):.3f}",
                Num="1",
            )

        # Hot Cue 2: Next Track Handoff Point
        if i + 1 < len(tracklist):
            next_start_sec = float(tracklist[i + 1].get("start_ms", 0)) / 1000.0
            if next_start_sec > start_sec:
                ET.SubElement(
                    track,
                    "POSITION_MARK",
                    Name="Next Track Handoff",
                    Type="0",
                    Start=f"{next_start_sec:.3f}",
                    Num="2",
                )

    playlists = ET.SubElement(dj_playlists, "PLAYLISTS")
    node = ET.SubElement(playlists, "NODE", Name="ROOT", Type="0")
    playlist = ET.SubElement(
        node,
        "NODE",
        Name="Auto DJ Master Set",
        Type="1",
        KeyType="0",
        Entries=str(len(tracklist)),
    )

    for i in range(len(tracklist)):
        ET.SubElement(playlist, "TRACK", Key=str(i + 1))

    tree = ET.ElementTree(dj_playlists)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)
