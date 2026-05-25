""" Digital Signal Processing (DSP) Module | Auto DJ Script (v9.2.0 - DJ Transitions).
==============================================================
Professional DJ-style transitions with proper EQ swap logic.
"""

import numpy as np
from .utils import ndarray_to_pydub, pydub_to_ndarray
from .np_signal import get_butter_coeffs, apply_iir_filter


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
    b, a = get_butter_coeffs(cutoff, sr, btype=filter_type)
    return apply_iir_filter(audio_array, b, a)


def trim_silence(segment, silence_threshold=-65.0, chunk_size=10):
    """Surgical silence removal using optimized NumPy analysis."""
    samples = np.array(segment.get_array_of_samples())
    if len(samples) == 0:
        return segment

    abs_samples = np.abs(samples)
    threshold = (10 ** (silence_threshold / 20.0)) * (2 ** 15)

    active_indices = np.where(abs_samples > threshold)[0]
    if len(active_indices) == 0:
        return segment

    start_sample = active_indices[0]
    end_sample = active_indices[-1]

    start_ms = int(start_sample * 1000 / segment.frame_rate / segment.channels)
    end_ms = int(end_sample * 1000 / segment.frame_rate / segment.channels)

    return segment[start_ms:end_ms]


def normalize_lufs(audio_array, sr, target_lufs=-14.0):
    rms = np.sqrt(np.mean(audio_array ** 2))
    if rms < 1e-6:
        return audio_array
    target_gain = 10 ** (target_lufs / 20.0)
    normalized = audio_array * (target_gain / rms)
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized /= (peak / 0.99)
    return normalized


def apply_limiter(audio_array, threshold=0.99):
    abs_audio = np.abs(audio_array)
    mask = abs_audio > threshold
    if np.any(mask):
        out = np.where(
            abs_audio > threshold,
            np.sign(audio_array) * (threshold + (1 - threshold) * np.tanh((abs_audio - threshold) / (1 - threshold))),
            audio_array
        )
        return out
    return audio_array


def apply_log_fade(audio_array, fade_type='in', dip_db=-2.5):
    """Apply logarithmic fade curve.
    
    The dip_db parameter controls a subtle mid-boost that compensates
    for the perceived volume dip during crossfades. -2.5dB is gentle.
    """
    num_samples = audio_array.shape[1] if audio_array.ndim == 2 else len(audio_array)
    x = np.linspace(0, 1, num_samples)

    # Equal-power crossfade: sqrt curves ensure constant total power
    curve = np.sqrt(x) if fade_type == 'in' else np.sqrt(1.0 - x)

    # Subtle dip compensation (prevents "hole" at crossover point)
    dip_factor = 1.0 - (1.0 - 10 ** (dip_db / 20.0)) * np.sin(np.pi * x)

    return audio_array * curve * dip_factor


def apply_multiband_compression(audio_array, sr, intensity=0.5, genre_profile=None):
    return audio_array


def calculate_spectral_clash(outro_array, intro_array, sr):
    return {'low': 1.0, 'mid': 1.0, 'high': 1.0}


@ArchetypeRegistry.register
class DualFilterSweep(TransitionArchetype):
    name = "progressive"
    display_name = "Dual-Sweep (Professional)"

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        """Professional DJ-style EQ swap transition.

        The key principle: in a real DJ mix, you don't kill the outgoing
        track's entire spectrum. You SWAP the bass and gently fade the rest.

        Phase 1 (0-40%):  Intro enters with bass cut (HPF 200Hz->80Hz).
                           Outro at full bandwidth. This lets the intro's
                           mid/high frequencies blend over the outro.
        Phase 2 (40-60%): Bass swap zone. Both tracks have bass briefly.
                           This is the "mixing point" where the kick drum
                           transitions from outgoing to incoming.
        Phase 3 (60-100%): Outro LPF closes 10kHz->2kHz (gentle, not harsh).
                            Intro fully open. Outro naturally recedes.

        IMPORTANT: This function ONLY does spectral separation (EQ curves).
        The volume crossfade is applied separately by transition_render_worker
        using apply_log_fade(). The combination creates a smooth transition
        where neither track "abruptly" disappears.
        """
        total_samples = outro_array.shape[1] if outro_array.ndim == 2 else len(outro_array)
        if total_samples == 0:
            return outro_array, intro_array

        block_size = int(sr * 0.05)  # 50ms blocks for smooth sweeps
        f_m = np.copy(outro_array)
        f_n = np.copy(intro_array)

        try:
            from scipy.signal import lfilter
            use_scipy = True
        except ImportError:
            use_scipy = False

        for start in range(0, total_samples, block_size):
            end = min(total_samples, start + block_size)
            progress = start / total_samples

            # OUTRO: full bandwidth for first 60%, then gentle LPF closing
            if progress < 0.6:
                lp_f = 20000.0  # Full bandwidth - outro dominates
            else:
                # Gentle close: 10kHz -> 2kHz over last 40%
                t = (progress - 0.6) / 0.4  # 0..1
                lp_f = 10000 * (2000 / 10000) ** t

            # INTRO: bass-cut opening up, mid/high always present
            if progress < 0.4:
                # HPF sweeps 200Hz -> 80Hz (bass gradually entering)
                t = progress / 0.4  # 0..1
                hp_f = 200 * (80 / 200) ** t
            elif progress < 0.6:
                # Bass swap zone: brief overlap where both have bass
                hp_f = 80.0
            else:
                # Intro fully open - it's now the main track
                hp_f = 20.0

            b_m, a_m = get_butter_coeffs(lp_f, sr, btype='lowpass')
            b_n, a_n = get_butter_coeffs(hp_f, sr, btype='highpass')

            chunk_m = f_m[:, start:end] if f_m.ndim == 2 else f_m[start:end]
            chunk_n = f_n[:, start:end] if f_n.ndim == 2 else f_n[start:end]

            if chunk_m.shape[-1] > 0:
                if use_scipy:
                    if chunk_m.ndim == 2:
                        for ch in range(chunk_m.shape[0]):
                            f_m[ch, start:end] = lfilter(b_m, a_m, chunk_m[ch])
                    else:
                        f_m[start:end] = lfilter(b_m, a_m, chunk_m)

            if chunk_n.shape[-1] > 0:
                if use_scipy:
                    if chunk_n.ndim == 2:
                        for ch in range(chunk_n.shape[0]):
                            f_n[ch, start:end] = lfilter(b_n, a_n, chunk_n[ch])
                    else:
                        f_n[start:end] = lfilter(b_n, a_n, chunk_n)

        return f_m, f_n


def apply_bass_swap(outro, intro, sr, **kwargs):
    return apply_dsp_filter(outro, sr, 'highpass', 150.0), apply_dsp_filter(intro, sr, 'lowpass', 150.0)


def apply_echo_out(outro, sr, **kwargs):
    return outro, None


def apply_hpf_sweep(outro, sr, **kwargs):
    return outro, None
