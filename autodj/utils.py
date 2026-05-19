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

    Implementation:
    - Extracts raw sample data.
    - Reshapes into (Channels, Samples) for librosa compatibility.
    - Scales by 2^15 to achieve unity gain in the float domain.
    """
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels == 2:
        # Pydub interleaves [L, R, L, R]. We need [[L, L], [R, R]].
        samples = samples.reshape((-1, 2)).T
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
    return librosa.get_duration(path=file_path)
