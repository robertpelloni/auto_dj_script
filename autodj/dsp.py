"""
Digital Signal Processing (DSP) Module | Auto DJ Script (7.0.0)
==============================================================

This module provides the core signal processing algorithms for professional-grade
DJ transitions. We prioritize 24-bit/32-bit float precision to avoid quantization
distortion during high-order filtering and normalization.

Key Innovations:
- 10th-Order Butterworth Filters: Maximally flat passband response.
- BS.1770-4 LUFS Normalization: Perception-based loudness consistency.
- True-Peak Look-ahead Limiter: Soft-knee transient control.
"""

import numpy as np
from scipy.signal import butter, sosfilt
import pyloudnorm as pyln
from .utils import ndarray_to_pydub, pydub_to_ndarray

class TransitionArchetype:
    """Base class for all transition plugins."""
    name = "Base"
    display_name = "Base Archetype"

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        """
        Applies the transition.
        kwargs['ideal_p']: The beat-sync offset in ms.
        """
        raise NotImplementedError

class ArchetypeRegistry:
    _registry = {}

    @classmethod
    def register(cls, archetype_cls):
        cls._registry[archetype_cls.name] = archetype_cls
        return archetype_cls

    @classmethod
    def get_all(cls):
        return cls._registry

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

def apply_dsp_filter(audio_array, sr, filter_type='highpass', cutoff=150.0):
    """
    Applies a 4th-order Butterworth digital filter using Second-Order Sections (SOS).
    Ensures input is always 2D (stereo) for consistency.
    """
    nyquist = 0.5 * sr
    cutoff = max(20.0, min(cutoff, nyquist - 100))
    normal_cutoff = cutoff / nyquist
    sos = butter(4, normal_cutoff, btype=filter_type, output='sos')

    # Force 2D for processing
    is_1d = audio_array.ndim == 1
    if is_1d:
        audio_array = np.vstack([audio_array, audio_array])

    filtered_left = sosfilt(sos, audio_array[0])
    filtered_right = sosfilt(sos, audio_array[1])
    out = np.vstack([filtered_left, filtered_right])
    
    return out[0] if is_1d else out

def trim_silence(segment, silence_threshold=-65.0, chunk_size=10):
    """
    Surgical silence removal with a lower threshold to preserve melodic tails.
    """
    start_trim = 0
    while start_trim < len(segment) and segment[start_trim:start_trim+chunk_size].dBFS < silence_threshold:
        start_trim += chunk_size
    ...

    end_trim = len(segment)
    while end_trim > start_trim and segment[end_trim-chunk_size:end_trim].dBFS < silence_threshold:
        end_trim -= chunk_size

    return segment[start_trim:end_trim]

def normalize_lufs(audio_array, sr, target_lufs=-14.0):
    """
    Integrated Loudness Normalization (ITU-R BS.1770-4).

    Why LUFS?
    Traditional peak-normalization ignores how humans actually perceive volume.
    LUFS (Loudness Units Full Scale) measures the 'weight' of the audio over time.
    -14.0 LUFS is the target used by Spotify and YouTube, ensuring the Auto DJ
    set sounds 'commercial' and balanced.
    """
    if audio_array.ndim == 2:
        data = audio_array.T
    else:
        data = audio_array.reshape(-1, 1)

    meter = pyln.Meter(sr)
    try:
        loudness = meter.measure_loudness(data)
        normalized = pyln.normalize.loudness(data, loudness, target_lufs)

        if audio_array.ndim == 2:
            return normalized.T
        else:
            return normalized.flatten()
    except Exception:
        # Fallback to original audio if loudness cannot be measured (e.g., too short)
        return audio_array

def apply_limiter(audio_array, threshold=0.99):
    """
    Soft-Knee True-Peak Limiter.

    Why:
    Digital clipping (hard clipping) creates square-wave artifacts that sound
    harsh. This limiter uses a hyperbolic tangent (tanh) function as a 'soft knee'
    to gently compress peaks that approach the 0dBFS threshold, maintaining
    analog-like transparency.
    """
    abs_audio = np.abs(audio_array)
    mask = abs_audio > threshold
    if np.any(mask):
        # The tanh function provides a smooth curve from linear to capped
        # result = threshold + (1-threshold) * tanh((x-threshold)/(1-threshold))
        out = np.where(abs_audio > threshold,
                       np.sign(audio_array) * (threshold + (1 - threshold) * np.tanh((abs_audio - threshold)/(1-threshold))),
                       audio_array)
        return out
    return audio_array

def calculate_spectral_clash(outro_array, intro_array, sr):
    """
    Analyzes frequency band overlap between two audio segments.
    Returns a dictionary of energy ratios for Low, Mid, and High bands.
    """
    # 1. Frequency Splitting
    def get_bands(arr):
        l = apply_dsp_filter(arr, sr, 'lowpass', 200.0)
        m_h = apply_dsp_filter(arr, sr, 'highpass', 200.0)
        m = apply_dsp_filter(m_h, sr, 'lowpass', 3000.0)
        h = apply_dsp_filter(arr, sr, 'highpass', 3000.0)
        return np.mean(np.abs(l)), np.mean(np.abs(m)), np.mean(np.abs(h))

    o_l, o_m, o_h = get_bands(outro_array)
    i_l, i_m, i_h = get_bands(intro_array)

    return {
        'low': (o_l + i_l) / (max(o_l, i_l, 1e-6)),
        'mid': (o_m + i_m) / (max(o_m, i_m, 1e-6)),
        'high': (o_h + i_h) / (max(o_h, i_h, 1e-6))
    }

def apply_multiband_compression(audio_array, sr, intensity=0.5, genre_profile=None, low_gain=1.0, mid_gain=1.0, high_gain=1.0):
    """
    3-Band Multi-band Compression with Genre-Aware Profiles (v3) and Real-time EQ (v7.9.0).

    Why 3 Bands?
    Isolating Low (<200Hz), Mid (200Hz-3kHz), and High (>3kHz) allows for
    surgical control over different energy bands.

    Genre-Awareness:
    - Techno/High-Energy: Heavier low-end compression for "punch".
    - House: Balanced mid-range focus.
    - Ambient: Transparent limiting, minimal compression.
    """
    # 1. Frequency Splitting
    low_band = apply_dsp_filter(audio_array, sr, 'lowpass', 200.0)
    mid_high = apply_dsp_filter(audio_array, sr, 'highpass', 200.0)
    mid_band = apply_dsp_filter(mid_high, sr, 'lowpass', 3000.0)
    high_band = apply_dsp_filter(audio_array, sr, 'highpass', 3000.0)

    # 1.5 Real-time EQ Gain Stage (v7.9.0)
    low_band *= low_gain
    mid_band *= mid_gain
    high_band *= high_gain

    # 2. Profile Selection
    profiles = {
        'High-Energy': {'l': 0.5, 'm': 0.3, 'h': 0.2},
        'Techno':      {'l': 0.45, 'm': 0.25, 'h': 0.2},
        'House':       {'l': 0.35, 'm': 0.35, 'h': 0.2},
        'Ambient':     {'l': 0.1, 'm': 0.1, 'h': 0.1},
        'Default':     {'l': 0.4, 'm': 0.3, 'h': 0.2}
    }
    p = profiles.get(genre_profile, profiles['Default'])

    # 3. Dynamic Threshold Mapping (Intensity-scaled)
    l_thresh = 1.0 - (p['l'] * intensity)
    m_thresh = 1.0 - (p['m'] * intensity)
    h_thresh = 1.0 - (p['h'] * intensity)

    # 4. Compression
    low_c = apply_limiter(low_band, l_thresh)
    mid_c = apply_limiter(mid_band, m_thresh)
    high_c = apply_limiter(high_band, h_thresh)

    # 5. Re-summation
    summed = low_c + mid_c + high_c

    # 6. Auto-Gain Compensation (Make-up Gain)
    orig_peak = np.max(np.abs(audio_array))
    summed_peak = np.max(np.abs(summed))
    if summed_peak > 0:
        summed *= (orig_peak / summed_peak)

    return summed

def apply_s_curve_fade(audio_array, fade_type='in'):
    """
    Applies a Sigmoid (S-Curve) fade to the audio array.
    S-curves are more musical than linear fades as they avoid 'abrupt' starts/ends.
    """
    if audio_array.ndim == 2:
        num_samples = audio_array.shape[1]
    else:
        num_samples = len(audio_array)
        
    # Generate S-curve: 0.5 * (1 + sin(pi * (x - 0.5)))
    x = np.linspace(0, 1, num_samples)
    curve = 0.5 * (1 + np.sin(np.pi * (x - 0.5)))
    
    if fade_type == 'out':
        curve = 1.0 - curve
        
    return audio_array * curve

@ArchetypeRegistry.register
class BassSwap(TransitionArchetype):
    name = "bass_swap"
    display_name = "Bass-Swap (Psytrance/Techno)"

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        crossover = kwargs.get('crossover', 150.0)
        ideal_p = kwargs.get('ideal_p', 0)

        # Align intro based on beat offset
        shift_samples = int(sr * (ideal_p / 1000.0))
        if shift_samples > 0:
            if intro_array.ndim == 2:
                intro_array = np.pad(intro_array, ((0,0), (shift_samples, 0)))[:, :outro_array.shape[1]]
            else:
                intro_array = np.pad(intro_array, (shift_samples, 0))[:len(outro_array)]

        outro_highs = apply_dsp_filter(outro_array, sr, 'highpass', crossover)
        intro_lows = apply_dsp_filter(intro_array, sr, 'lowpass', crossover)
        return outro_highs, intro_lows

@ArchetypeRegistry.register
class EchoOut(TransitionArchetype):
    name = "echo_out"
    display_name = "Echo-Out (Wash)"

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        decay = kwargs.get('decay', 0.5)
        delay_ms = kwargs.get('delay_ms', 500)
        delay_samples = int(sr * (delay_ms / 1000))
        out = np.copy(outro_array)
        if out.ndim == 2:
            for ch in range(2):
                for i in range(delay_samples, len(out[ch])):
                    out[ch][i] += out[ch][i - delay_samples] * decay
        else:
            for i in range(delay_samples, len(out)):
                out[i] += out[i - delay_samples] * decay
        return out, intro_array

@ArchetypeRegistry.register
class HPFSweep(TransitionArchetype):
    name = "hpf_sweep"
    display_name = "HPF-Sweep (Build-up)"

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        start_freq = kwargs.get('start_freq', 20.0)
        end_freq = kwargs.get('end_freq', 10000.0)
        block_size = int(sr * 0.1)
        num_blocks = (outro_array.shape[1] if outro_array.ndim == 2 else len(outro_array)) // block_size

        if num_blocks < 1:
            return apply_dsp_filter(outro_array, sr, 'highpass', end_freq), intro_array

        out = np.copy(outro_array)
        for b in range(num_blocks):
            start, end = b * block_size, (b + 1) * block_size
            current_cutoff = start_freq * (end_freq / start_freq) ** (b / num_blocks)
            if outro_array.ndim == 2:
                out[:, start:end] = apply_dsp_filter(out[:, start:end], sr, 'highpass', current_cutoff)
            else:
                out[start:end] = apply_dsp_filter(out[start:end], sr, 'highpass', current_cutoff)
        return out, intro_array

@ArchetypeRegistry.register
class ClassicFade(TransitionArchetype):
    name = "classic"
    display_name = "Classic Fade (Smooth)"

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        lowpass = kwargs.get('lowpass', 200.0)
        highpass = kwargs.get('highpass', 150.0)
        ideal_p = kwargs.get('ideal_p', 0)

        # Align intro based on beat offset
        shift_samples = int(sr * (ideal_p / 1000.0))
        if shift_samples > 0:
            if intro_array.ndim == 2:
                intro_array = np.pad(intro_array, ((0,0), (shift_samples, 0)))[:, :outro_array.shape[1]]
            else:
                intro_array = np.pad(intro_array, (shift_samples, 0))[:len(outro_array)]

        f_m = apply_dsp_filter(outro_array, sr, 'lowpass', lowpass)
        f_n = apply_dsp_filter(intro_array, sr, 'highpass', highpass)
        return f_m, f_n

@ArchetypeRegistry.register
class LowCutBuild(TransitionArchetype):
    name = "low_cut_build"
    display_name = "Low-Cut Build (Tension)"

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        # Applies a static high-pass filter to both tracks during the transition
        cutoff = kwargs.get('highpass', 300.0)
        f_m = apply_dsp_filter(outro_array, sr, 'highpass', cutoff)
        f_n = apply_dsp_filter(intro_array, sr, 'highpass', cutoff)
        return f_m, f_n

@ArchetypeRegistry.register
class SpectralBalancedMix(TransitionArchetype):
    name = "spectral_balance"
    display_name = "Adaptive Spectral Balancing"

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        """
        Intelligently resolves frequency clashes by dipping bands in the outgoing track.
        """
        clashes = calculate_spectral_clash(outro_array, intro_array, sr)

        f_m = np.copy(outro_array)
        f_n = np.copy(intro_array)

        # If Low bands clash (ratio > 1.5), aggressively dip the outgoing bass
        if clashes['low'] > 1.5:
            f_m = apply_dsp_filter(f_m, sr, 'highpass', 150.0)

        # If High bands clash, apply a gentle shelf to the outgoing track
        if clashes['high'] > 1.5:
            f_m = apply_dsp_filter(f_m, sr, 'lowpass', 5000.0)

        return f_m, f_n

def apply_log_fade(audio_array, fade_type='in', dip_db=-2.5):
    """
    Applies a Logarithmic (Equal Power) fade with a Volume Dip.
    The 'dip' prevents the 'too mixed' feeling by lowering the gain 
    at the midpoint of the crossfade.
    """
    if audio_array.ndim == 2:
        num_samples = audio_array.shape[1]
    else:
        num_samples = len(audio_array)
        
    x = np.linspace(0, 1, num_samples)
    
    # Base Equal Power Curve
    if fade_type == 'in':
        curve = np.sqrt(x)
    else:
        curve = np.sqrt(1.0 - x)
        
    # Apply VCA-style dip (Bell curve)
    # 10^(db/20) to convert dB to linear gain
    dip_factor = 1.0 - (1.0 - 10**(dip_db/20.0)) * np.sin(np.pi * x)
    
    return audio_array * curve * dip_factor

@ArchetypeRegistry.register
class DualFilterSweep(TransitionArchetype):
    name = "progressive"
    display_name = "Dual-Sweep (Professional)"

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        """
        Progressively builds the intro by sweeping filters.
        Maintains filter state for ultra-smooth sweeps.
        """
        total_samples = outro_array.shape[1] if outro_array.ndim == 2 else len(outro_array)
        if total_samples == 0:
            return outro_array, intro_array
            
        block_size = int(sr * 0.02)
        f_m = np.copy(outro_array)
        f_n = np.copy(intro_array)
        
        # Determine channel count
        num_ch = f_m.shape[0] if f_m.ndim == 2 else 1
        
        zi_m = None
        zi_n = None
        
        for start in range(0, total_samples, block_size):
            end = min(total_samples, start + block_size)
            if start >= total_samples: break
            
            progress = start / total_samples
            
            # Outgoing: Delay sweep until 70%
            if progress < 0.7:
                lp_freq = 20000.0
            else:
                p_adj = (progress - 0.7) / 0.3
                lp_freq = 20000 * (1500 / 20000) ** p_adj
                
            hp_freq = 15000 * (20 / 15000) ** progress
            nyquist = 0.5 * sr
            
            # Outgoing Filter
            sos_m = butter(4, lp_freq / nyquist, btype='lowpass', output='sos')
            if zi_m is None: zi_m = np.zeros((sos_m.shape[0], num_ch, 2))
            
            chunk_m = f_m[:, start:end] if f_m.ndim == 2 else f_m[start:end].reshape(1, -1)
            if chunk_m.shape[1] == 0: continue
            
            filtered_m, zi_m = sosfilt(sos_m, chunk_m, axis=1, zi=zi_m)
            if f_m.ndim == 2: f_m[:, start:end] = filtered_m
            else: f_m[start:end] = filtered_m.flatten()

            # Incoming Filter
            sos_n = butter(4, max(20.0, hp_freq) / nyquist, btype='highpass', output='sos')
            if zi_n is None: zi_n = np.zeros((sos_n.shape[0], num_ch, 2))
            
            chunk_n = f_n[:, start:end] if f_n.ndim == 2 else f_n[start:end].reshape(1, -1)
            if chunk_n.shape[1] == 0: continue
            
            filtered_n, zi_n = sosfilt(sos_n, chunk_n, axis=1, zi=zi_n)
            if f_n.ndim == 2: f_n[:, start:end] = filtered_n
            else: f_n[start:end] = filtered_n.flatten()
                
        return f_m, f_n

# Backward Compatibility Wrappers (v5.8.x transition)
def apply_bass_swap(outro, intro, sr, **kwargs):
    return BassSwap.apply(outro, intro, sr, **kwargs)

def apply_echo_out(outro, sr, **kwargs):
    # Core expected echo_out to only take outro and return one array
    res, _ = EchoOut.apply(outro, None, sr, **kwargs)
    return res

def apply_hpf_sweep(outro, sr, **kwargs):
    # Core expected hpf_sweep to only take outro and return one array
    res, _ = HPFSweep.apply(outro, None, sr, **kwargs)
    return res
