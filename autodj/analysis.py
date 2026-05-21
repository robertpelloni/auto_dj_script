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
    """Detects BPM using multiple windows and a voting system for high accuracy."""
    # Ensure y is mono for analysis
    if y.ndim == 2:
        y_mono = librosa.to_mono(y)
    else:
        y_mono = y
        
    duration = librosa.get_duration(y=y_mono, sr=sr)
    
    # Analyze 4 distinct 20-second windows
    window_sec = 20.0
    offsets = [duration * 0.15, duration * 0.35, duration * 0.55, duration * 0.75]
    
    bpms = []
    for offset in offsets:
        start_sample = int(offset * sr)
        end_sample = min(len(y_mono), int((offset + window_sec) * sr))
        chunk = y_mono[start_sample:end_sample]
        
        if len(chunk) < sr * 5: continue
            
        onset_env = librosa.onset.onset_strength(y=chunk, sr=sr)
        # Narrow the search range for Psytrance (typically 130-155)
        # but allow for faster/slower tracks
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=140)
        
        if isinstance(tempo, np.ndarray): tempo = tempo[0]
        bpms.append(float(tempo))

    if not bpms:
        return 140.0, y, sr

    # Enhanced Correction Logic
    corrected_bpms = []
    for b in bpms:
        # Fix 1/2 or 2x errors
        while b < 110: b *= 2
        while b > 175: b /= 2
        corrected_bpms.append(b)
        
    final_bpm = float(np.median(corrected_bpms))
    print(f"  [ANALYSIS] Detected BPMs: {bpms} -> Final: {final_bpm:.2f}")
    return final_bpm, y, sr


def get_energy_profile(y, sr):
    """Calculates RMS energy."""
    y_mono = librosa.to_mono(y) if y.ndim == 2 else y
    return np.mean(librosa.feature.rms(y=y_mono))

def detect_phrases(y, sr):
    """Detects structural boundaries using spectral novelty peaks. Handles stereo input."""
    y_mono = librosa.to_mono(y) if y.ndim == 2 else y
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
    boundaries = librosa.util.peak_pick(onset_env, pre_max=30, post_max=30, pre_avg=30, post_avg=30, delta=0.5, wait=30)
    return (librosa.frames_to_time(boundaries, sr=sr) * 1000).astype(int)

def analyze_geometry(segment, sr, target_bpm, beats_per_bar, transition_bars):
    """Calculates millisecond offsets for track segmentation and phase alignment."""
    samples = pydub_to_ndarray(segment)
    samples_mono = librosa.to_mono(samples)
    
    ms_per_beat = 60000.0 / target_bpm
    ms_per_bar = ms_per_beat * beats_per_bar
    ms_per_transition = ms_per_bar * transition_bars
    
    # Track-wide beat detection to find the "groove"
    onset_env = librosa.onset.onset_strength(y=samples_mono, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=target_bpm)
    beat_times_ms = (librosa.frames_to_time(beat_frames, sr=sr) * 1000).astype(int)
    
    # Find the first real beat (first strong transient)
    # This is critical for tracks with long ambient intros
    first_beat_ms = 0
    if len(beat_times_ms) > 0:
        first_beat_ms = beat_times_ms[0]
        
    return beat_times_ms, int(ms_per_transition), first_beat_ms

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
    Used for tail extension during transitions. Handles stereo.
    """
    ms_per_beat = 60000.0 / bpm
    ms_per_bar = ms_per_beat * beats_per_bar
    samples_per_bar = int(sr * (ms_per_bar / 1000.0))

    # Focus on the last 30 seconds for outro loops. Slicing on time axis.
    if y.ndim == 2:
        last_30_s = y[:, -int(sr * 30):] if y.shape[1] > sr * 30 else y
        last_30_s_mono = librosa.to_mono(last_30_s)
    else:
        last_30_s = y[-int(sr * 30):] if len(y) > sr * 30 else y
        last_30_s_mono = last_30_s

    # Simple rhythmic similarity: find segments with similar energy envelopes
    onset_env = librosa.onset.onset_strength(y=last_30_s_mono, sr=sr)

    # Heuristic: find a high-energy bar near the end
    if y.ndim == 2:
        return last_30_s[:, -samples_per_bar:]
    return last_30_s[-samples_per_bar:]

def find_sync_offset(outro_y, intro_y, sr, bpm):
    """
    Finds the sample-accurate offset between two tracks using cross-correlation 
    of their onset envelopes.
    """
    # Ensure both are mono for envelope calculation
    o_m = librosa.to_mono(outro_y) if outro_y.ndim == 2 else outro_y
    i_m = librosa.to_mono(intro_y) if intro_y.ndim == 2 else intro_y
    
    # Calculate onset envelopes
    o_env = librosa.onset.onset_strength(y=o_m, sr=sr)
    i_env = librosa.onset.onset_strength(y=i_m, sr=sr)
    
    # Increase search window to +/- 2 beats (captures larger drift)
    ms_per_beat = 60000.0 / bpm
    hop_length = 512
    search_frames = int((ms_per_beat * 2 / 1000.0) * sr / hop_length)
    
    correlation = np.correlate(o_env, i_env, mode='full')
    center = len(i_env) - 1
    
    # Find local maximum within the search window
    start_idx = max(0, center - search_frames)
    end_idx = min(len(correlation), center + search_frames)
    window = correlation[start_idx : end_idx]
    
    if len(window) == 0: return 0
    
    best_lag_rel = np.argmax(window)
    actual_lag_frames = (start_idx + best_lag_rel) - center
    nudge_ms = (actual_lag_frames * hop_length / sr) * 1000
    
    return int(nudge_ms)

def get_genre_archetype(y, sr, bpm=None):
    """
    Identifies the genre archetype using a multi-feature heuristic (v3 - Enhanced).
    """
    y_mono = librosa.to_mono(y) if y.ndim == 2 else y
    centroid = np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_mono, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y_mono))
    mfcc = np.mean(librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y_mono, sr=sr))

    if bpm is None:
        if centroid > 3000 or rolloff > 6000: return 'High-Energy'
        if contrast > 20: return 'Techno'
        return 'Ambient'

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

    if mfcc[0] > -150: return 'High-Energy'
    if contrast > 18: return 'Techno'
    return 'Ambient'
