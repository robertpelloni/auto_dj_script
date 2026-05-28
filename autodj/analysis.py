"""Lightweight Audio Analysis Module (v9.4.0 - Robust BPM Detection).

Uses librosa beat tracking for precise BPM detection and beat position finding,
with pure-numpy fallbacks when librosa is unavailable.

Key improvements:
- get_native_bpm: uses librosa beat tracker + median IBI measurement
  instead of simple autocorrelation on kick envelope
- analyze_geometry: walks beats from center outward to find first/last beats
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

    # Pre-compute FFT bin to chroma bin mapping
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
    """BPM detection using librosa beat tracking + median IBI refinement.

    Method:
    1. Compute spectral flux onset envelope via librosa
    2. Beat-track using dynamic programming (robust to noise)
    3. Measure median inter-beat interval from tracked beats
    4. Remove outliers (>20% off median) and re-measure for precision
    5. Cross-validate with autocorrelation for octave error correction

    This approach is far more reliable than simple autocorrelation on a
    kick envelope, which frequently locks onto half/double tempo.
    """
    try:
        import librosa

        y_mono = np.mean(y, axis=0) if y.ndim == 2 else y

        # Use up to 5 minutes for analysis (balance speed vs accuracy)
        n_samples = min(len(y_mono), sr * 300)
        y_seg = y_mono[:n_samples]

        hop = 512
        onset_env = librosa.onset.onset_strength(y=y_seg, sr=sr, hop_length=hop)

        # Beat tracking with dynamic programming
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop,
            start_bpm=145.0, tightness=100.0
        )
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)

        # Convert to times and measure actual inter-beat intervals
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

        if len(beat_times) >= 5:
            ibis = np.diff(beat_times)
            median_ibi = np.median(ibis)

            # Remove outliers (>20% off median)
            good_mask = np.abs(ibis - median_ibi) / median_ibi < 0.20
            good = ibis[good_mask]

            if len(good) > 3:
                precise_ibi = np.mean(good)
            else:
                precise_ibi = median_ibi

            bpm = 60.0 / precise_ibi

            # Cross-validate with autocorrelation to catch octave errors
            acf = np.correlate(onset_env - np.mean(onset_env),
                               onset_env - np.mean(onset_env), mode='full')
            acf = acf[len(acf) // 2:]
            if acf[0] > 0:
                acf = acf / acf[0]

            min_lag = max(1, int(60 * sr / (hop * 200)))
            max_lag = min(int(60 * sr / (hop * 80)), len(acf) - 2)
            if max_lag > min_lag:
                search = acf[min_lag:max_lag + 1]
                acf_peak_lag = np.argmax(search) + min_lag
                acf_bpm = 60.0 * sr / (hop * acf_peak_lag)

                # If beat-track and ACF disagree by ~2x, use ACF
                ratio = bpm / acf_bpm
                if 0.4 < ratio < 0.6:
                    bpm = acf_bpm
                elif 1.6 < ratio < 2.4:
                    bpm = acf_bpm
        else:
            pass  # keep bpm from precise_ibi

        # Final sanity: clamp to reasonable range
        if bpm < 80:
            bpm *= 2
        elif bpm > 200:
            bpm /= 2

        return float(bpm), y, sr

    except ImportError:
        # Fallback: original autocorrelation method (no librosa)
        y_mono = np.mean(y, axis=0) if y.ndim == 2 else y
        b, a = get_butter_coeffs(150.0, sr, btype='lowpass')
        y_kick = np.abs(apply_iir_filter(y_mono, b, a))
        hop = 512
        y_ds = y_kick[::hop]
        corr = fast_correlate(y_ds, y_ds)
        corr = corr[len(corr) // 2:]
        min_lag = int((60 / 180) * sr / hop)
        max_lag = int((60 / 100) * sr / hop)
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
    energy = [np.mean(np.abs(y_mono[i:i + hop])) for i in range(0, len(y_mono), hop)]
    diff = np.abs(np.diff(energy))
    threshold = np.mean(diff) * 2
    breaks = np.where(diff > threshold)[0]
    return (breaks * hop * 1000 / sr).astype(int)


def analyze_geometry(segment, sr, target_bpm, beats_per_bar, transition_bars):
    """Returns (beat_times_ms, theoretical_ms_trans, first_beat_ms, last_beat_ms).

    Uses librosa beat tracking for precise beat position detection,
    then walks beats from the center outward to find first/last beats reliably.

    Method: "Follow the beats to the beginning and end"
    1. Find beats in the middle of the track (where they are most reliable)
    2. Measure the average inter-beat interval
    3. Walk backward from the first detected beat to find the true FIRST beat
    4. Walk forward from the last detected beat to find the true LAST beat
    5. Snap first/last beats to nearest kick drum peak for sample accuracy
    """
    from .utils import pydub_to_ndarray
    y = pydub_to_ndarray(segment)
    y_mono = np.mean(y, axis=0) if y.ndim == 2 else y

    ms_per_beat = 60000.0 / target_bpm
    ms_per_bar = ms_per_beat * beats_per_bar
    ms_per_transition = ms_per_bar * transition_bars

    try:
        import librosa

        hop = 512
        onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr, hop_length=hop)

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop,
            start_bpm=target_bpm, tightness=100.0
        )
        beat_times_s = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
        beat_times_ms = (beat_times_s * 1000).astype(int)

        # Snap each beat to the nearest kick drum PEAK (not onset).
        # This ensures that beat alignment between tracks aligns the
        # perceptual "beat" (the kick peak) rather than the onset,
        # eliminating galloping caused by different onset-to-peak delays.
        try:
            from scipy.signal import butter as _butter, lfilter as _lfilter, find_peaks as _fp
            _b, _a = _butter(2, 80.0 / (0.5 * sr), btype='low')
            _kick_env = np.abs(_lfilter(_b, _a, y_mono))
            _kick_env = _kick_env / (np.max(_kick_env) + 1e-10)
            _min_dist = int(0.6 * 60 / target_bpm * sr)
            _kick_peaks, _ = _fp(_kick_env, height=0.2, distance=_min_dist)
            _kick_peak_ms = (_kick_peaks * 1000.0 / sr).astype(int)
            
            if len(_kick_peak_ms) > 10:
                _snapped = []
                _max_snap = 30  # max snap distance in ms
                for bt in beat_times_ms:
                    _dists = np.abs(_kick_peak_ms - bt)
                    _nearest_idx = np.argmin(_dists)
                    if _dists[_nearest_idx] <= _max_snap:
                        _snapped.append(int(_kick_peak_ms[_nearest_idx]))
                    else:
                        _snapped.append(bt)  # keep original if no nearby peak
                beat_times_ms = np.array(_snapped)
        except Exception:
            pass  # if peak detection fails, keep onset-based positions

        if len(beat_times_ms) >= 5:
            # Measure the actual beat interval from tracked beats
            ibis_ms = np.diff(beat_times_ms)
            median_ibi_ms = float(np.median(ibis_ms))

            # Walk backward from first detected beat to find the true FIRST beat
            first_detected_ms = int(beat_times_ms[0])
            # Only walk backward if the first beat is within 4 bars of track start
            # If it is further (long ambient intro), just use the detected position
            max_walkback_ms = int(median_ibi_ms * beats_per_bar * 4)
            if first_detected_ms > max_walkback_ms:
                # Long intro: the first beat IS the first detected beat
                first_beat_ms = first_detected_ms
            elif first_detected_ms > median_ibi_ms:
                n_beats_before = int(first_detected_ms / median_ibi_ms)
                first_beat_ms = first_detected_ms - int(n_beats_before * median_ibi_ms)
                # Keep the grid-calculated position (onset-aligned)
                # Do NOT snap to kick peak - librosa onsets are already onset-aligned
                # and snapping would shift by ~20ms (attack->peak), creating
                # an inconsistency with the beat array which uses onset positions.
            else:
                first_beat_ms = first_detected_ms

            # Walk forward from last detected beat to find the true LAST beat
            last_detected_ms = int(beat_times_ms[-1])
            total_ms = int(len(y_mono) * 1000 / sr)
            remaining_ms = total_ms - last_detected_ms
            # Only walk forward if within 4 bars of track end
            max_walkfwd_ms = int(median_ibi_ms * beats_per_bar * 4)
            if remaining_ms > max_walkfwd_ms:
                # Long outro: the last beat IS the last detected beat
                last_beat_ms = last_detected_ms
            elif remaining_ms > median_ibi_ms * 0.5:
                n_beats_after = int(remaining_ms / median_ibi_ms)
                last_beat_ms = last_detected_ms + int(n_beats_after * median_ibi_ms)
                # Keep grid position (onset-aligned, consistent with beat array)
            else:
                last_beat_ms = last_detected_ms

            return beat_times_ms, int(ms_per_transition), int(first_beat_ms), int(last_beat_ms)

    except ImportError:
        pass

    # Fallback: original kick envelope method
    b, a = get_butter_coeffs(150.0, sr, btype='lowpass')
    y_kick = np.abs(apply_iir_filter(y_mono, b, a))

    search_start = int(sr * 10)
    window_s = y_kick[:search_start]
    first_kick_sample = np.argmax(window_s) if len(window_s) > 0 else 0
    first_beat_ms = int(first_kick_sample * 1000 / sr)

    search_end = int(sr * 20)
    window_e = y_kick[-search_end:] if len(y_kick) > search_end else y_kick
    last_kick_rel = np.argmax(window_e) if len(window_e) > 0 else 0
    last_kick_sample = len(y_kick) - (len(window_e) - last_kick_rel)
    last_beat_ms = int(last_kick_sample * 1000 / sr)

    beat_times_ms = np.arange(first_beat_ms, len(y_mono) * 1000 / sr, ms_per_beat).astype(int)
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
    window = y[:, -int(sr * 30):] if y.ndim == 2 else y[-int(sr * 30):]
    step = samples_per_bar
    max_e, best_chunk = 0, None
    for i in range(0, (window.shape[1] if window.ndim == 2 else len(window)) - step, step):
        chunk = window[:, i:i + step] if window.ndim == 2 else window[i:i + step]
        e = np.mean(np.abs(chunk))
        if e > max_e:
            max_e, best_chunk = e, chunk
    if best_chunk is not None:
        return best_chunk
    return y[:, -samples_per_bar:] if y.ndim == 2 else y[-samples_per_bar:]


def find_sync_offset(outro_y, intro_y, sr, bpm):
    """Finds sample-accurate sync offset by aligning kick drum peaks.
    
    Instead of cross-correlating envelopes (which aligns overall musical
    patterns), this detects individual kick drum peaks and finds the
    offset that best aligns them. This ensures the loudest part of each
    kick lands at the same time, eliminating galloping/flamming.
    
    Method:
    1. Lowpass at 60Hz to isolate just the kick drum body
    2. Detect peaks in both tracks' kick envelopes
    3. Try different offsets and count how many peaks align within 10ms
    4. Return the offset with the most aligned peaks
    """
    from scipy.signal import butter, lfilter, find_peaks as sp_find_peaks
    
    o_m = np.mean(outro_y, axis=0) if outro_y.ndim == 2 else outro_y
    i_m = np.mean(intro_y, axis=0) if intro_y.ndim == 2 else intro_y

    # Use a very low cutoff (60Hz) to isolate just the kick drum
    # This avoids confusion from basslines and mid-range content
    try:
        b, a = butter(2, 60.0 / (0.5 * sr), btype='low')
    except Exception:
        b, a = get_butter_coeffs(80.0, sr, btype='lowpass')
    o_kick = np.abs(lfilter(b, a, o_m))
    i_kick = np.abs(lfilter(b, a, i_m))

    # Normalize
    o_max = np.max(o_kick)
    i_max = np.max(i_kick)
    if o_max < 1e-6 or i_max < 1e-6:
        return 0
    o_kick = o_kick / o_max
    i_kick = i_kick / i_max

    # Find kick peaks in both tracks
    min_dist = int(0.7 * 60 / bpm * sr)  # 70% of a beat period
    o_peaks, _ = sp_find_peaks(o_kick, height=0.15, distance=min_dist)
    i_peaks, _ = sp_find_peaks(i_kick, height=0.15, distance=min_dist)

    if len(o_peaks) < 4 or len(i_peaks) < 4:
        # Fallback: use cross-correlation if not enough peaks
        win = min(len(o_kick), len(i_kick), int(sr * (60 / bpm) * 8))
        if win < 100:
            return 0
        corr = fast_correlate(o_kick[:win], i_kick[:win])
        center = win - 1
        search = int(sr * (60 / bpm) * 1.0)
        start = max(0, center - search)
        end = min(len(corr), center + search)
        slice_c = corr[start:end]
        if len(slice_c) == 0:
            return 0
        lag = np.argmax(slice_c) - (len(slice_c) // 2)
        peak_val = np.max(slice_c)
        avg_val = np.mean(np.abs(slice_c))
        if peak_val < avg_val * 1.5:
            return 0
        if abs(lag) > search * 0.9:
            return 0
        return int(lag * 1000 / sr)

    # Use only the middle portion of peaks (most reliable)
    # Skip first/last 10% (filter settling, breakdowns)
    n_skip_o = max(2, len(o_peaks) // 10)
    n_skip_i = max(2, len(i_peaks) // 10)
    o_peaks = o_peaks[n_skip_o:max(n_skip_o+1, len(o_peaks)-n_skip_o)]
    i_peaks = i_peaks[n_skip_i:max(n_skip_i+1, len(i_peaks)-n_skip_i)]

    if len(o_peaks) < 3 or len(i_peaks) < 3:
        return 0

    # Search for the offset that aligns the most peaks
    ms_per_beat = 60.0 / bpm * 1000
    max_offset_ms = ms_per_beat  # search +/-1 beat
    tolerance = int(0.010 * sr)  # 10ms tolerance for "aligned"
    
    best_offset = 0
    best_count = 0
    
    # Coarse search: 5ms steps
    for offset_ms in np.arange(-max_offset_ms, max_offset_ms, 5.0):
        offset_samples = int(offset_ms * sr / 1000)
        shifted = i_peaks + offset_samples
        count = 0
        for sp in shifted:
            if np.any(np.abs(o_peaks - sp) <= tolerance):
                count += 1
        if count > best_count:
            best_count = count
            best_offset = offset_ms

    # Fine search: 1ms steps around the best
    for offset_ms in np.arange(best_offset - 10, best_offset + 10, 1.0):
        offset_samples = int(offset_ms * sr / 1000)
        shifted = i_peaks + offset_samples
        count = 0
        for sp in shifted:
            if np.any(np.abs(o_peaks - sp) <= tolerance):
                count += 1
        if count > best_count:
            best_count = count
            best_offset = offset_ms

    # Only apply if enough peaks align (at least 30% of incoming peaks)
    min_aligned = max(3, len(i_peaks) * 0.3)
    if best_count < min_aligned:
        return 0

    return int(best_offset)


def get_genre_archetype(y, sr, bpm=None):
    """Genre classification using spectral centroid + BPM heuristics."""
    if y.ndim == 2:
        y = np.mean(y, axis=0)
    fft = np.abs(np.fft.rfft(y[:min(len(y), sr * 30)]))
    freqs = np.fft.rfftfreq(len(y[:min(len(y), sr * 30)]), 1.0 / sr)
    centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)

    if bpm and bpm >= 170:
        return "Hi-Tech", "Fast psytrance (170+ BPM)"
    elif bpm and bpm >= 145:
        if centroid > 4000:
            return "Full-On", "High-energy psytrance"
        elif centroid > 2800:
            return "Full-On", "Standard psytrance"
        else:
            return "Progressive", "Groovy psytrance"
    elif bpm and bpm >= 130:
        if centroid > 3500:
            return "Progressive", "Energetic progressive"
        elif centroid > 2000:
            return "Progressive", "Standard progressive"
        else:
            return "Minimal", "Minimal progressive"
    elif bpm and bpm >= 115:
        if centroid > 3000:
            return "Progressive", "Mid-tempo psytrance"
        else:
            return "Downtempo", "Chillout psytrance"
    else:
        if centroid > 2000:
            return "Downtempo", "Ambient psytrance"
    return "Ambient", "Chillout"


def extract_spectral_terrain(y, sr):
    """Stub: Returns empty terrain data."""
    return []
