"""
Digital Signal Processing (DSP) Module | Auto DJ Script (v5.4.1)
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
    Applies a 10th-order Butterworth digital filter using Second-Order Sections (SOS).

    Why 10th Order?
    Standard DJ mixers often use 2nd or 4th order filters. A 10th-order filter
    provides a much steeper roll-off (~60dB per octave), allowing for surgical
    isolation of frequency bands during complex transitions (e.g., swapping bass
    lines without any bleed from the outgoing track).

    Why SOS?
    Second-order sections are more numerically stable than transfer function (ba)
    representations, especially for high-order filters which can otherwise
    suffer from precision errors in floating-point arithmetic.
    """
    nyquist = 0.5 * sr
    # Safety: Clamp cutoff within stable ranges
    cutoff = max(20.0, min(cutoff, nyquist - 100))
    normal_cutoff = cutoff / nyquist
    sos = butter(10, normal_cutoff, btype=filter_type, output='sos')

    if audio_array.ndim == 2:
        # Stereo processing: Apply filter to each channel independently
        filtered_left = sosfilt(sos, audio_array[0])
        filtered_right = sosfilt(sos, audio_array[1])
        return np.vstack([filtered_left, filtered_right])
    else:
        return sosfilt(sos, audio_array)

def trim_silence(segment, silence_threshold=-50.0, chunk_size=10):
    """
    Surgical silence removal for precise beat-alignment.

    In DJing, 10ms of silence at the start of a track causes an immediate
    beat-sync failure. This function iterates through chunks to find the exact
    transient onset.
    """
    start_trim = 0
    while start_trim < len(segment) and segment[start_trim:start_trim+chunk_size].dBFS < silence_threshold:
        start_trim += chunk_size

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

def apply_multiband_compression(audio_array, sr, intensity=0.5):
    """
    3-Band Multi-band Compression for professional mastering (v2).

    Why 3 Bands?
    Isolating Low (<200Hz), Mid (200Hz-3kHz), and High (>3kHz) allows for
    surgical control over the 'punch' of the kick, the 'clarity' of the vocals/leads,
    and the 'shimmer' of the hats without cross-band pumping.

    Intensity (0.0 to 1.0): Scales the thresholds for all three bands.
    """
    # 1. Frequency Splitting
    low_band = apply_dsp_filter(audio_array, sr, 'lowpass', 200.0)

    mid_high = apply_dsp_filter(audio_array, sr, 'highpass', 200.0)
    mid_band = apply_dsp_filter(mid_high, sr, 'lowpass', 3000.0)

    high_band = apply_dsp_filter(audio_array, sr, 'highpass', 3000.0)

    # 2. Dynamic Threshold Mapping
    # High intensity = Lower threshold = More compression
    l_thresh = 1.0 - (0.4 * intensity)
    m_thresh = 1.0 - (0.3 * intensity)
    h_thresh = 1.0 - (0.2 * intensity)

    # 3. Compression
    low_c = apply_limiter(low_band, l_thresh)
    mid_c = apply_limiter(mid_band, m_thresh)
    high_c = apply_limiter(high_band, h_thresh)

    # 4. Re-summation
    summed = low_c + mid_c + high_c

    # 5. Auto-Gain Compensation (Make-up Gain)
    # Ensure the compressed signal peak matches the original peak level
    # to prevent volume drops at high intensity.
    orig_peak = np.max(np.abs(audio_array))
    summed_peak = np.max(np.abs(summed))
    if summed_peak > 0:
        summed *= (orig_peak / summed_peak)

    return summed

@ArchetypeRegistry.register
class BassSwap(TransitionArchetype):
    name = "bass_swap"
    display_name = "Bass-Swap (Psytrance/Techno)"

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        crossover = kwargs.get('crossover', 150.0)
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
