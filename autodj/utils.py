"""
Utility functions for data type conversion and audio segment manipulation.

Theoretical Foundations:
1. Pydub interoperability: Pydub uses signed 16-bit PCM for internal storage.
2. Normalization: We convert to float32 range [-1.0, 1.0] for high-precision DSP.
"""
import numpy as np
from pydub import AudioSegment

def pydub_to_ndarray(segment):
    """
    Converts a Pydub AudioSegment into a normalized float32 NumPy array.
    FORCES STEREO output to maintain pipeline consistency.
    """
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels == 2:
        samples = samples.reshape((-1, 2)).T
    else:
        # Force mono to stereo
        samples = np.vstack([samples, samples])
    
    samples /= (2**15)
    return samples

def ndarray_to_pydub(audio_array, sr):
    """
    Converts a normalized float32 NumPy array back into a pristine Pydub AudioSegment.

    Implementation:
    - Scales back to signed 16-bit integer range.
    - Re-interleaves stereo channels if necessary.
    - Packages into the binary buffer format required by Pydub.
    """
    # Inverse scaling
    audio_array = (audio_array * (2**15)).astype(np.int16)

    if audio_array.ndim == 2:
        # Interleave channels: [[L, L], [R, R]] -> [L, R, L, R]
        interleaved = audio_array.T.flatten()
        return AudioSegment(interleaved.tobytes(), frame_rate=sr, sample_width=2, channels=2)
    else:
        return AudioSegment(audio_array.tobytes(), frame_rate=sr, sample_width=2, channels=1)

def get_track_duration(file_path):
    """
    Quickly retrieves the duration of an audio file in seconds.
    """
    import librosa
    return librosa.get_duration(path=file_path)

def export_rekordbox_xml(tracklist, output_xml_path):
    """
    Generates a Rekordbox-compatible XML for the master set (v7.3.0).
    Allows importing the Auto DJ tracklist into Pioneer DJ software.
    """
    import xml.etree.ElementTree as ET
    from datetime import datetime

    dj_playlists = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    product = ET.SubElement(dj_playlists, "PRODUCT", Name="AutoDJ", Version="7.3.0", Company="robertpelloni")
    collection = ET.SubElement(dj_playlists, "COLLECTION", Entries=str(len(tracklist)))

    for i, item in enumerate(tracklist):
        # Rekordbox expects specific path formats and metadata
        track = ET.SubElement(collection, "TRACK",
            TrackID=str(i+1),
            Name=item['file'],
            Artist="Auto DJ",
            Genre=item['genre'],
            Kind="FLAC File",
            Size="0",
            TotalTime=str(int(item.get('duration_ms', 0) / 1000)),
            Location=f"file://localhost{item['path']}" if 'path' in item else "",
            AverageBpm=item.get('bpm', "0"),
            Tonality=item['key']
        )
        # Add Cue point for the transition start
        ET.SubElement(track, "POSITION_MARK",
            Name="Transition Start",
            Type="0",
            Start=str(item['start_ms'] / 1000),
            Num="0"
        )

    playlists = ET.SubElement(dj_playlists, "PLAYLISTS")
    node = ET.SubElement(playlists, "NODE", Name="ROOT", Type="0")
    playlist = ET.SubElement(node, "NODE", Name="Auto DJ Master Set", Type="1", KeyType="0", Entries=str(len(tracklist)))

    for i in range(len(tracklist)):
        ET.SubElement(playlist, "TRACK", Key=str(i+1))

    tree = ET.ElementTree(dj_playlists)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)
