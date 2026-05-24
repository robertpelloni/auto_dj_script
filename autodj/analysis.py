"""
Audio analysis module for the Auto DJ system (v7.0.0).
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
from scipy.signal import butter, sosfilt, sosfiltfilt

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
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=140)
        
        if isinstance(tempo, np.ndarray): tempo = tempo[0]
        bpms.append(float(tempo))

    if not bpms:
        return 140.0, y, sr

    # 3. Detect global beats for linear refinement
    y_mono_full = librosa.to_mono(y) if y.ndim == 2 else y
    onset_full = librosa.onset.onset_strength(y=y_mono_full, sr=sr)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_full, sr=sr, start_bpm=np.median(bpms))
    beat_times_ms = (librosa.frames_to_time(beat_frames, sr=sr) * 1000).astype(int)

    # Enhanced Correction Logic
    corrected_bpms = []
    for b in bpms:
        while b < 110: b *= 2
        while b > 175: b /= 2
        corrected_bpms.append(b)
        
    final_bpm = float(np.median(corrected_bpms))

    # 4. Linear-Fit BPM Refinement (v7.0.0)
    # Fits a linear model to beat timestamps to find the 'True Constant BPM'
    if len(beat_times_ms) > 15:
        # We use a large window for regression to avoid local variations
        x = np.arange(len(beat_times_ms))
        # Use polyfit to find the best constant interval (slope)
        slope, intercept = np.polyfit(x, beat_times_ms, 1)
        fitted_bpm = 60000.0 / slope
        
        # If fit is stable, use it. fitted_bpm is far more accurate for warping.
        if abs(fitted_bpm - final_bpm) < 1.5:
            final_bpm = fitted_bpm
            print(f"  [ANALYSIS] Refined BPM via Linear Fit: {final_bpm:.4f}")

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
    """Calculates millisecond offsets for track segmentation and downbeat anchoring."""
    samples = pydub_to_ndarray(segment)
    samples_mono = librosa.to_mono(samples)
    
    # Force Kick-only analysis for anchoring
    nyquist = 0.5 * sr
    sos_kick = butter(4, 150.0 / nyquist, btype='lowpass', output='sos')
    samples_kick = sosfiltfilt(sos_kick, samples_mono)

    ms_per_beat = 60000.0 / target_bpm
    ms_per_bar = ms_per_beat * beats_per_bar
    ms_per_transition = ms_per_bar * transition_bars
    
    # 1. High-Res Onset Detection on filtered kick signal
    onset_env = librosa.onset.onset_strength(y=samples_kick, sr=sr)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=target_bpm)
    beat_times_ms = (librosa.frames_to_time(beat_frames, sr=sr) * 1000).astype(int)
    
    if len(beat_times_ms) == 0:
        return [], int(ms_per_transition), 0

    # 2. Kick-Locked Downbeat Finder (Version 8.5)
    # We use a pattern-matching heuristic: Verify the 'One' is part of a 4/4 cycle.
    search_limit = min(len(beat_times_ms), 32)
    kick_scores = []

    samples_per_beat = int((ms_per_beat / 1000.0) * sr)

    for i in range(search_limit - 4):
        # Calculate a 'Pattern Score' for this beat being the 'One'
        # Check energy at i, i+1, i+2, i+3
        pattern_energy = 0
        for offset in range(4):
            b_ms = beat_times_ms[i + offset]
            start_s = int(max(0, (b_ms - 15) * sr / 1000))
            end_s = int(min(len(samples_kick), (b_ms + 15) * sr / 1000))
            pattern_energy += np.max(np.abs(samples_kick[start_s:end_s]))

        # Check for 'Silence' between beats (confirms it's a transient, not a drone)
        mid_beat_ms = beat_times_ms[i] + (ms_per_beat / 2.0)
        m_start = int((mid_beat_ms - 15) * sr / 1000)
        m_end = int((mid_beat_ms + 15) * sr / 1000)
        mid_energy = np.max(np.abs(samples_kick[m_start:m_end])) if m_start < len(samples_kick) else 1.0

        # High score = Loud beats + Quiet gaps
        kick_scores.append(pattern_energy / (mid_energy + 0.01))

    # Anchor to the beat with the most consistent 4/4 'pulse'
    downbeat_idx = np.argmax(kick_scores) if kick_scores else 0
    first_beat_ms = beat_times_ms[downbeat_idx]

    print(f"  [ANALYSIS] Kick Pattern Lock: Beat {downbeat_idx} at {first_beat_ms}ms (Score: {max(kick_scores):.2f})")
        
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
    Finds the most rhythmically stable 1-bar or 2-bar phrase for looping (7.0.0).
    Uses Cross-Correlation for sample-accurate loop point detection.
    """
    ms_per_beat = 60000.0 / bpm
    ms_per_bar = ms_per_beat * beats_per_bar
    samples_per_bar = int(sr * (ms_per_bar / 1000.0))

    # Focus on the last 45 seconds for outro loops
    lookback_samples = int(sr * 45)
    if y.ndim == 2:
        outro_segment = y[:, -lookback_samples:] if y.shape[1] > lookback_samples else y
        outro_mono = librosa.to_mono(outro_segment)
    else:
        outro_segment = y[-lookback_samples:] if len(y) > lookback_samples else y
        outro_mono = outro_segment

    # Calculate onset strength envelope
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=outro_mono, sr=sr, hop_length=hop_length)

    # We want to find a bar that highly correlates with the one preceding it
    bar_frames = int(samples_per_bar / hop_length)
    if len(onset_env) < bar_frames * 2:
        # Fallback to the last bar if track is too short
        return y[:, -samples_per_bar:] if y.ndim == 2 else y[-samples_per_bar:]

    # sliding window cross-correlation on the envelope
    search_range = len(onset_env) - bar_frames * 2
    best_score = -1
    best_idx = len(onset_env) - bar_frames

    for i in range(search_range):
        current_bar = onset_env[i : i + bar_frames]
        next_bar = onset_env[i + bar_frames : i + 2 * bar_frames]
        score = np.corrcoef(current_bar, next_bar)[0, 1]
        if score > best_score:
            best_score = score
            best_idx = i

    # Convert frame index back to samples
    loop_start_sample = best_idx * hop_length

    if y.ndim == 2:
        return outro_segment[:, loop_start_sample : loop_start_sample + samples_per_bar]
    return outro_segment[loop_start_sample : loop_start_sample + samples_per_bar]

def find_sync_offset(outro_y, intro_y, sr, bpm):
    """
    Finds the sample-accurate offset using Peak-Centric Alignment (Version 8.0).
    Identifies kick transients and locks their peaks.
    """
    nyquist = 0.5 * sr
    sos = butter(4, [20.0 / nyquist, 150.0 / nyquist], btype='bandpass', output='sos')

    o_m = librosa.to_mono(outro_y) if outro_y.ndim == 2 else outro_y
    i_m = librosa.to_mono(intro_y) if intro_y.ndim == 2 else intro_y
    
    # Isolate kicks (Absolute amplitude for peak finding)
    o_kick = np.abs(sosfiltfilt(sos, o_m))
    i_kick = np.abs(sosfiltfilt(sos, i_m))
    
    # 1. Energy Gating: Skip search if signal is too quiet (ambient)
    if np.max(i_kick) < 0.05:
        return 0

    ms_per_beat = 60000.0 / bpm
    samples_per_beat = int((ms_per_beat / 1000.0) * sr)
    
    # 2. Cross-Correlation on a measure-wide window
    limit = samples_per_beat * 16 # Check 4 bars
    o_slice = o_kick[-limit:] if len(o_kick) > limit else o_kick
    i_slice = i_kick[:limit] if len(i_kick) > limit else i_kick
    
    correlation = np.correlate(o_slice, i_slice, mode='full')
    center = len(i_slice) - 1

    # Search window: +/- 1.5 full beats (catches most phasing errors)
    search_samples = int(samples_per_beat * 1.5)
    start_idx = max(0, center - search_samples)
    end_idx = min(len(correlation), center + search_samples)
    window = correlation[start_idx : end_idx]
    
    if len(window) == 0: return 0
    
    # 3. Confidence Check
    best_lag_rel = np.argmax(window)
    peak_val = window[best_lag_rel]
    avg_val = np.mean(window)
    
    if peak_val < avg_val * 1.25:
        return 0 # Low confidence, trust the grid instead

    actual_lag_samples = (start_idx + best_lag_rel) - center
    return int((actual_lag_samples / sr) * 1000)

def extract_ai_features(y, sr):
    """
    Extracts high-dimensional spectral features for CNN-based genre inference (7.0.0).
    Returns a feature vector containing MFCCs, Spectral Centroid, Contrast, Flatness, and Rolloff.
    """
    y_mono = librosa.to_mono(y) if y.ndim == 2 else y

    # 1. Mel-Frequency Cepstral Coefficients (Timbre)
    mfccs = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)

    # 2. Spectral Centroid (Brightness)
    centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
    centroid_mean = np.mean(centroid)

    # 3. Spectral Contrast (Texture)
    contrast = librosa.feature.spectral_contrast(y=y_mono, sr=sr)
    contrast_mean = np.mean(contrast)

    # 4. Spectral Flatness (Noisiness)
    flatness = librosa.feature.spectral_flatness(y=y_mono)
    flatness_mean = np.mean(flatness)

    # 5. Spectral Rolloff (High-frequency content)
    rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=sr)
    rolloff_mean = np.mean(rolloff)

    return {
        'mfccs': mfccs_mean.tolist(),
        'centroid': float(centroid_mean),
        'contrast': float(contrast_mean),
        'flatness': float(flatness_mean),
        'rolloff': float(rolloff_mean)
    }

def get_genre_archetype(y, sr, bpm=None):
    """
    Identifies the genre archetype using the AI Inference Engine (7.0.0).
    Returns (genre, rationale).
    """
    from .models import GenreClassifier

    features = extract_ai_features(y, sr)
    classifier = GenreClassifier()

    genre = classifier.predict(features)
    rationale = classifier.get_rationale(features)

    # BPM-based sanity check
    if bpm and bpm > 150 and genre == 'Ambient':
        genre = 'High-Energy'
        rationale = f"BPM override ({bpm:.0f} > 150): Ambient classification rejected."

    return genre, rationale

def extract_spectral_terrain(y, sr, bins=64):
    """
    Generates a high-resolution energy map of the track for 3D visualization (v7.0.0).
    Returns a downsampled mel-spectrogram matrix.
    """
    y_mono = librosa.to_mono(y) if y.ndim == 2 else y

    # Generate Mel-Spectrogram
    S = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_mels=bins)

    # Convert to log-scale (dB)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Downsample time-axis for lightweight transmission (aim for ~100 points)
    hop = max(1, S_db.shape[1] // 100)
    terrain = S_db[:, ::hop]

    return terrain.tolist()
