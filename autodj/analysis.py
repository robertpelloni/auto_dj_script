""" Lightweight Audio Analysis Module (v9.2.0 - Scipy-Free).
Uses pure NumPy for stability in restricted environments.
Fixed: analyze_geometry, find_sync_offset broken references.
"""

import numpy as np
import soundfile as sf
from .np_signal import get_butter_coeffs, apply_iir_filter, fast_correlate


def get_musical_key(y, sr):
    """Musical key detection using pure-numpy chromagram + Krumhansl-Schmuckler profiling."""
    if y.ndim == 2:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y

    n_fft = 8192
    hop = n_fft // 2
    y_short = y_mono[:min(len(y_mono), sr * 30)]

    # Pre-compute FFT bin → chroma bin mapping
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    min_freq, max_freq = 65.0, 2000.0
    valid = (freqs >= min_freq) & (freqs <= max_freq)
    valid_freqs = freqs[valid]
    if len(valid_freqs) == 0:
        return "G Minor"
    midi_notes = 12 * np.log2(valid_freqs / 440.0) + 69
    chroma_bins = np.round(midi_notes).astype(int) % 12
    valid_bin_indices = np.where(valid)[0]

    # Vectorized STFT
    window = np.hanning(n_fft)
    num_frames = max(1, (len(y_short) - n_fft) // hop + 1)
    chroma = np.zeros(12)

    if len(y_short) >= n_fft:
        try:
            frames = np.lib.stride_tricks.sliding_window_view(y_short, n_fft)[::hop][:num_frames]
            spectra = np.abs(np.fft.rfft(frames * window, axis=1))
            for j, c_bin in zip(valid_bin_indices, chroma_bins):
                chroma[c_bin] += np.sum(spectra[:, j])
        except Exception:
            # Fallback for very short audio
            frame = y_short[:n_fft] * window if len(y_short) >= n_fft else y_short
            spec = np.abs(np.fft.rfft(frame, n=n_fft))
            for j, c_bin in zip(valid_bin_indices, chroma_bins):
                if j < len(spec):
                    chroma[c_bin] += spec[j]

    if np.max(chroma) < 1e-10:
        return "G Minor"

    # Krumhansl-Schmuckler key-finding algorithm
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    def _norm(p):
        return (p - np.mean(p)) / (np.std(p) + 1e-10)

    maj_n, min_n, chroma_n = _norm(major_profile), _norm(minor_profile), _norm(chroma)

    best_corr, best_key, best_mode = -2, 0, "Major"
    for i in range(12):
        rotated = np.roll(chroma_n, i)
        c_maj = np.dot(rotated, maj_n)
        c_min = np.dot(rotated, min_n)
        if c_maj > best_corr:
            best_corr, best_key, best_mode = c_maj, i, "Major"
        if c_min > best_corr:
            best_corr, best_key, best_mode = c_min, i, "Minor"

    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{notes[best_key]} {best_mode}"

def get_camelot_key(key_str):
    """Maps standard keys to the Camelot Wheel."""
    mapping = {
        'G# Minor': '1A', 'B Major': '1B',
        'D# Minor': '2A', 'F# Major': '2B',
        'A# Minor': '3A', 'C# Major': '3B',
        'F Minor': '4A', 'G# Major': '4B',
        'C Minor': '5A', 'D# Major': '5B',
        'G Minor': '6A', 'A# Major': '6B',
        'D Minor': '7A', 'F Major': '7B',
        'A Minor': '8A', 'C Major': '8B',
        'E Minor': '9A', 'G Major': '9B',
        'B Minor': '10A', 'D Major': '10B',
        'F# Minor': '11A', 'A Major': '11B',
        'C# Minor': '12A', 'E Major': '12B'
    }
    return mapping.get(key_str, "6A")

def is_harmonically_compatible(key1, key2):
    """Checks Camelot wheel adjacency."""
    c1, c2 = get_camelot_key(key1), get_camelot_key(key2)
    if "Unknown" in (c1, c2):
        return True
    n1, m1, n2, m2 = int(c1[:-1]), c1[-1], int(c2[:-1]), c2[-1]
    if n1 == n2:
        return True
    if m1 == m2 and (abs(n1 - n2) in (1, 11)):
        return True
    return False


def get_native_bpm(y, sr):
    """BPM detection using autocorrelation on kick envelope."""
    y_mono = np.mean(y, axis=0) if y.ndim == 2 else y
    b, a = get_butter_coeffs(150.0, sr, btype='lowpass')
    y_kick = np.abs(apply_iir_filter(y_mono, b, a))
    hop = 512
    y_ds = y_kick[::hop]
    corr = fast_correlate(y_ds, y_ds)
    corr = corr[len(corr)//2:]
    min_lag = int((60/170) * sr / hop)
    max_lag = int((60/120) * sr / hop)
    peaks = corr[min_lag:max_lag]
    if len(peaks) == 0:
        return 145.0, y, sr
    best_lag = np.argmax(peaks) + min_lag
    final_bpm = 60.0 / (best_lag * hop / sr)
    return float(final_bpm), y, sr


def get_energy_profile(y, sr):
    """Calculates RMS energy."""
    return np.mean(np.abs(y))


def detect_phrases(y, sr):
    """Detects structural boundaries using energy novelty."""
    y_mono = np.mean(y, axis=0) if y.ndim == 2 else y
    hop = int(sr * 2)
    energy = [np.mean(np.abs(y_mono[i:i+hop])) for i in range(0, len(y_mono), hop)]
    diff = np.abs(np.diff(energy))
    threshold = np.mean(diff) * 2
    breaks = np.where(diff > threshold)[0]
    return (breaks * hop * 1000 / sr).astype(int)


def analyze_geometry(segment, sr, target_bpm, beats_per_bar, transition_bars):
    """
    Returns (beat_times_ms, theoretical_ms_trans, first_beat_ms, last_beat_ms).
    Uses pure-numpy kick detection — no scipy/librosa dependency.
    """
    from .utils import pydub_to_ndarray
    y = pydub_to_ndarray(segment)
    y_mono = np.mean(y, axis=0) if y.ndim == 2 else y

    # Kick detection via low-pass filter
    b, a = get_butter_coeffs(150.0, sr, btype='lowpass')
    y_kick = np.abs(apply_iir_filter(y_mono, b, a))

    ms_per_beat = 60000.0 / target_bpm
    ms_per_bar = ms_per_beat * beats_per_bar
    ms_per_transition = ms_per_bar * transition_bars

    # First Kick: search first 10 seconds
    search_start = int(sr * 10)
    window_s = y_kick[:search_start]
    first_kick_sample = np.argmax(window_s) if len(window_s) > 0 else 0
    first_beat_ms = int(first_kick_sample * 1000 / sr)

    # Last Kick: search last 20 seconds
    search_end = int(sr * 20)
    window_e = y_kick[-search_end:] if len(y_kick) > search_end else y_kick
    last_kick_rel = np.argmax(window_e) if len(window_e) > 0 else 0
    last_kick_sample = len(y_kick) - (len(window_e) - last_kick_rel)
    last_beat_ms = int(last_kick_sample * 1000 / sr)

    # Generate beat grid anchored to first kick
    beat_times_ms = np.arange(first_beat_ms, len(y_mono)*1000/sr, ms_per_beat).astype(int)

    return beat_times_ms, int(ms_per_transition), int(first_beat_ms), int(last_beat_ms)


def calculate_dynamic_transition(outro_y, intro_y, sr, target_bpm, beats_per_bar):
    """Analyzes phrase structure to determine optimal transition length."""
    outro_ph = detect_phrases(outro_y, sr)
    intro_ph = detect_phrases(intro_y, sr)
    ms_per_bar = (60000.0 / target_bpm) * beats_per_bar
    if len(outro_ph) > 5 or len(intro_ph) > 5:
        return 8
    elif len(outro_ph) < 2 and len(intro_ph) < 2:
        return 32
    else:
        return 16


def identify_loopable_phrase(y, sr, bpm, beats_per_bar=4):
    """Finds a high-energy bar for looping (tail extension)."""
    ms_per_beat = 60000.0 / bpm
    samples_per_bar = int(sr * (ms_per_beat * beats_per_bar / 1000.0))
    # Check last 30 seconds for a high-energy bar
    window = y[:, -int(sr*30):] if y.ndim == 2 else y[-int(sr*30):]
    step = samples_per_bar
    max_e, best_chunk = 0, None
    for i in range(0, (window.shape[1] if window.ndim == 2 else len(window)) - step, step):
        chunk = window[:, i:i+step] if window.ndim == 2 else window[i:i+step]
        e = np.mean(np.abs(chunk))
        if e > max_e:
            max_e, best_chunk = e, chunk
    if best_chunk is not None:
        return best_chunk
    return y[:, -samples_per_bar:] if y.ndim == 2 else y[-samples_per_bar:]


def find_sync_offset(outro_y, intro_y, sr, bpm):
    """
    Finds sample-accurate sync offset using cross-correlation on kick envelopes.
    Pure numpy — no librosa/scipy dependency.
    """
    # Mono conversion
    o_m = np.mean(outro_y, axis=0) if outro_y.ndim == 2 else outro_y
    i_m = np.mean(intro_y, axis=0) if intro_y.ndim == 2 else intro_y

    # Kick isolation via low-pass
    b, a = get_butter_coeffs(150.0, sr, btype='lowpass')
    o_kick = np.abs(apply_iir_filter(o_m, b, a))
    i_kick = np.abs(apply_iir_filter(i_m, b, a))

    # Energy gate: skip if too quiet
    if np.max(i_kick) < 0.05:
        return 0

    win = int(sr * (60/bpm) * 8)
    win = min(win, len(o_kick), len(i_kick))
    if win < 100:
        return 0

    # Cross-correlate kick envelopes
    corr = fast_correlate(o_kick[:win], i_kick[:win])
    center = win - 1
    ms_per_beat = 60000.0 / bpm
    search = int(sr * (60/bpm) * 0.5)
    start = max(0, center - search)
    end = min(len(corr), center + search)
    slice_c = corr[start:end]
    if len(slice_c) == 0:
        return 0

    lag = np.argmax(slice_c) - (len(slice_c) // 2)

    # Confidence check
    peak_val = np.max(slice_c)
    avg_val = np.mean(np.abs(slice_c))
    if peak_val < avg_val * 1.25:
        return 0

    return int(lag * 1000 / sr)


def get_genre_archetype(y, sr, bpm=None):
    """Genre classification using spectral centroid heuristic."""
    # Simple centroid estimation via FFT
    if y.ndim == 2:
        y = np.mean(y, axis=0)
    fft = np.abs(np.fft.rfft(y[:min(len(y), sr*30)]))
    freqs = np.fft.rfftfreq(len(y[:min(len(y), sr*30)]), 1.0/sr)
    centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)
    if centroid > 3000:
        return "High-Energy", "Standard Psytrance"
    elif centroid > 1500:
        return "Techno", "Progressive"
    return "Ambient", "Chillout"


def extract_spectral_terrain(y, sr):
    """Stub: Returns empty terrain data."""
    return []
