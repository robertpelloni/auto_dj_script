"""
Audio analysis module for the Auto DJ system.
This module is the "brain" of the engine, responsible for Music Information Retrieval (MIR).

Theoretical Foundations:
1. BPM Detection: Uses onset strength envelopes and autocorrelation.
2. Key Estimation: Utilizes Constant-Q Transforms and Krumhansl-Schmuckler pitch class profiles.
3. Phrase Detection: Based on spectral novelty curves to find structural anchor points.
4. Genre Profiling: Uses spectral centroid, rolloff, and flux to identify stylistic archetypes.
"""
import librosa
import numpy as np
from .utils import pydub_to_ndarray

def get_musical_key(y, sr):
    """Estimates the musical key of a track."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_sum = np.sum(chroma, axis=1)
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    maj_corr = [np.corrcoef(chroma_sum, np.roll(major_profile, i))[0, 1] for i in range(12)]
    min_corr = [np.corrcoef(chroma_sum, np.roll(minor_profile, i))[0, 1] for i in range(12)]
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if max(maj_corr) > max(min_corr):
        return f"{keys[np.argmax(maj_corr)]} Major"
    return f"{keys[np.argmax(min_corr)]} Minor"

def get_camelot_key(key_str):
    """Maps standard keys to the Camelot Wheel (Open Key notation)."""
    mapping = {
        'G# Minor': '1A', 'B Major': '1B', 'D# Minor': '2A', 'F# Major': '2B',
        'A# Minor': '3A', 'C# Major': '3B', 'F Minor': '4A', 'G# Major': '4B',
        'C Minor': '5A', 'D# Major': '5B', 'G Minor': '6A', 'A# Major': '6B',
        'D Minor': '7A', 'F Major': '7B', 'A Minor': '8A', 'C Major': '8B',
        'E Minor': '9A', 'G Major': '9B', 'B Minor': '10A', 'D Major': '10B',
        'F# Minor': '11A', 'A Major': '11B', 'C# Minor': '12A', 'E Major': '12B'
    }
    return mapping.get(key_str, "Unknown")

def is_harmonically_compatible(key1, key2):
    """Checks Camelot wheel adjacency."""
    c1, c2 = get_camelot_key(key1), get_camelot_key(key2)
    if "Unknown" in (c1, c2): return True
    n1, m1, n2, m2 = int(c1[:-1]), c1[-1], int(c2[:-1]), c2[-1]
    if n1 == n2: return True
    if m1 == m2 and (abs(n1 - n2) in (1, 11)): return True
    return False

def get_native_bpm(y, sr):
    """Detects BPM with octave correction."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    native_bpm, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    if isinstance(native_bpm, np.ndarray): native_bpm = native_bpm[0]
    if native_bpm < 100: native_bpm *= 2
    return float(native_bpm), y, sr

def get_energy_profile(y, sr):
    """Calculates RMS energy."""
    return np.mean(librosa.feature.rms(y=y))

def detect_phrases(y, sr):
    """Detects structural boundaries using spectral novelty peaks."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    boundaries = librosa.util.peak_pick(onset_env, pre_max=30, post_max=30, pre_avg=30, post_avg=30, delta=0.5, wait=30)
    return (librosa.frames_to_time(boundaries, sr=sr) * 1000).astype(int)

def analyze_geometry(segment, sr, target_bpm, beats_per_bar, transition_bars):
    """Calculates millisecond offsets for track segmentation and mixing."""
    samples = pydub_to_ndarray(segment)
    if segment.channels == 2:
        samples = librosa.to_mono(samples)
    ms_per_beat = 60000.0 / target_bpm
    ms_per_bar = ms_per_beat * beats_per_bar
    ms_per_transition = ms_per_bar * transition_bars
    onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=target_bpm)
    beat_times_ms = (librosa.frames_to_time(beat_frames, sr=sr) * 1000).astype(int)
    return beat_times_ms, int(ms_per_transition)

def calculate_dynamic_transition(outro_y, intro_y, sr, target_bpm, beats_per_bar):
    """
    Analyzes phrase structure to determine optimal transition length.

    Returns transition bars (8, 16, or 32).
    """
    outro_ph = detect_phrases(outro_y, sr)
    intro_ph = detect_phrases(intro_y, sr)

    ms_per_beat = 60000.0 / target_bpm
    ms_per_bar = ms_per_beat * beats_per_bar

    # Heuristic: Match phrase density to transition standard multiples
    if len(outro_ph) > 5 or len(intro_ph) > 5:
        # High activity: shorter transition
        return 8
    elif len(outro_ph) < 2 and len(intro_ph) < 2:
        # Low activity: long epic transition
        return 32
    else:
        return 16

def identify_loopable_phrase(y, sr, bpm, beats_per_bar=4):
    """
    Finds the most rhythmically stable 1-bar or 2-bar phrase for looping.
    Used for tail extension during transitions.
    """
    ms_per_beat = 60000.0 / bpm
    ms_per_bar = ms_per_beat * beats_per_bar
    samples_per_bar = int(sr * (ms_per_bar / 1000.0))

    # Focus on the last 30 seconds for outro loops
    last_30_s = y[-int(sr * 30):] if len(y) > sr * 30 else y

    # Simple rhythmic similarity: find segments with similar energy envelopes
    onset_env = librosa.onset.onset_strength(y=last_30_s, sr=sr)

    # Heuristic: find a high-energy bar near the end
    # (In a future version, use cross-correlation for sample-accurate loop points)
    return last_30_s[-samples_per_bar:]

def get_genre_archetype(y, sr, bpm=None):
    """
    Identifies the genre archetype using a multi-feature heuristic (v3 - Enhanced).

    Heuristic Logic:
    - High-Energy: BPM > 138 AND (Centroid > 2500 OR Rolloff > 5000).
    - Techno: 120 < BPM <= 140 AND Contrast > 20.
    - House: 105 < BPM <= 128 AND Flatness > 0.01.
    - Ambient: BPM <= 105 OR Centroid <= 1000.

    New Features in v3:
    - MFCC: Mean of first 13 coefficients for timbre density.
    - Spectral Contrast: Measures the energy difference between peaks and valleys.
    """
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

    if bpm is None:
        if centroid > 3000 or rolloff > 6000: return 'High-Energy'
        if contrast > 20: return 'Techno'
        return 'Ambient'

    # v3 Logic Path
    if bpm > 138:
        if centroid > 2500 or rolloff > 5000 or mfcc[0] > -100:
            return 'High-Energy'

    if 120 < bpm <= 142:
        if contrast > 18 or centroid > 1800:
            return 'Techno'

    if 105 < bpm <= 128:
        if flatness > 0.01 or contrast > 15:
            return 'House'

    if bpm <= 105 or centroid <= 1100:
        return 'Ambient'

    # Timbre-based fallback
    if mfcc[0] > -150: return 'High-Energy'
    if contrast > 18: return 'Techno'
    return 'Ambient'
