"""Digital Signal Processing (DSP) Module | Auto DJ Script (v9.3.0 - Clean Transitions).
==============================================================
Professional DJ-style transitions that sound like a real DJ mixer.

Design philosophy:
- Volume crossfade does 90% of the mixing work
- Bass swap handles the kick drum transition
- No aggressive filter sweeps — those are an effect, not a transition
- A real DJ mixer has 3-band EQ (HI/MID/LO) and volume faders, nothing more
"""

import numpy as np
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


def apply_dsp_filter(audio_array, sr, filter_type="highpass", cutoff=150.0):
    b, a = get_butter_coeffs(cutoff, sr, btype=filter_type)
    return apply_iir_filter(audio_array, b, a)


def trim_silence(segment, silence_threshold=-65.0, chunk_size=10):
    """Surgical silence removal using optimized NumPy analysis."""
    samples = np.array(segment.get_array_of_samples())
    if len(samples) == 0:
        return segment

    abs_samples = np.abs(samples)
    threshold = (10 ** (silence_threshold / 20.0)) * (2**15)

    active_indices = np.where(abs_samples > threshold)[0]
    if len(active_indices) == 0:
        return segment

    start_sample = active_indices[0]
    end_sample = active_indices[-1]

    start_ms = int(start_sample * 1000 / segment.frame_rate / segment.channels)
    end_ms = int(end_sample * 1000 / segment.frame_rate / segment.channels)

    return segment[start_ms:end_ms]


def normalize_lufs(audio_array, sr, target_lufs=-14.0):
    """Normalize audio to target LUFS using K-weighted RMS.
    Uses a simplified ITU-R BS.1770 model: high-shelf + high-pass
    filtering before RMS measurement for perceptually accurate loudness.
    Processes in chunks to handle large files without OOM.
    """
    if audio_array.ndim == 2:
        # BS.1770: sum channel powers then convert to loudness
        ch_powers = []
        for ch in range(audio_array.shape[0]):
            ch_loudness = _measure_loudness_chunked(audio_array[ch], sr)
            ch_powers.append(10 ** (ch_loudness / 10.0))
        total_power = sum(ch_powers)
        if total_power < 1e-20:
            return audio_array
        # -0.691dB channel sum correction per BS.1770 for stereo
        loudness = 10 * np.log10(total_power) - 0.691
    else:
        loudness = _measure_loudness_chunked(audio_array, sr)

    if loudness < -70 or np.isinf(loudness):
        return audio_array

    gain_db = target_lufs - loudness
    gain_lin = 10 ** (gain_db / 20.0)

    # Clamp gain to avoid extreme adjustments
    gain_lin = float(np.clip(gain_lin, 0.1, 10.0))

    # Apply gain in-place to avoid copying the entire array
    audio_array = audio_array * gain_lin
    peak = float(np.max(np.abs(audio_array)))
    if peak > 0.99:
        audio_array *= 0.99 / peak

    return audio_array


def _measure_loudness_chunked(audio, sr, chunk_seconds=120):
    """Measure perceptual loudness using chunked processing to avoid OOM.
    Processes audio through K-weighting filters in chunks with state carryover,
    accumulating the weighted sum-of-squares for accurate RMS.
    """
    from scipy.signal import bilinear, sosfilt, sosfilt_zi

    # Get K-weighting filter coefficients as second-order sections
    # Stage 1: high-shelf boost (~+4dB at 4kHz) - BS.1770 pre-filter
    if sr == 44100:
        # Direct SOS coefficients for the shelf filter
        b_shelf = [1.53512485, -2.69161304, 1.19839265]
        a_shelf = [1.0, -1.69004453, 0.73249962]
    else:
        b_shelf = [1.0]
        a_shelf = [1.0]

    # Stage 2: high-pass at ~60Hz
    try:
        b_hp, a_hp = bilinear([0, 1], [1, 1 / (2 * np.pi * 60)], sr=sr)
    except Exception:
        b_hp = [1.0]
        a_hp = [1.0]

    # Convert to SOS format for numerical stability and state tracking
    # Simple 2nd-order filters -> single SOS section each
    sos_shelf = np.array(
        [
            [
                b_shelf[0],
                b_shelf[1] if len(b_shelf) > 1 else 0,
                b_shelf[2] if len(b_shelf) > 2 else 0,
                a_shelf[0],
                a_shelf[1] if len(a_shelf) > 1 else 0,
                a_shelf[2] if len(a_shelf) > 2 else 0,
            ]
        ]
    )
    sos_hp = np.array(
        [
            [
                b_hp[0],
                b_hp[1] if len(b_hp) > 1 else 0,
                b_hp[2] if len(b_hp) > 2 else 0,
                a_hp[0],
                a_hp[1] if len(a_hp) > 1 else 0,
                a_hp[2] if len(a_hp) > 2 else 0,
            ]
        ]
    )

    chunk_size = int(chunk_seconds * sr)
    total_samples = len(audio)
    weighted_sum_sq = 0.0
    total_weighted = 0

    # Initialize filter states
    zi_shelf = sosfilt_zi(sos_shelf) * 0  # zero initial state
    zi_hp = sosfilt_zi(sos_hp) * 0

    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        chunk = audio[start:end]

        # Apply shelf filter with state carryover
        filtered, zi_shelf = sosfilt(sos_shelf, chunk, zi=zi_shelf)
        # Apply HP filter with state carryover
        filtered, zi_hp = sosfilt(sos_hp, filtered, zi=zi_hp)

        # Accumulate weighted sum of squares
        weighted_sum_sq += float(np.sum(filtered**2))
        total_weighted += end - start

    if total_weighted == 0 or weighted_sum_sq < 1e-20:
        return -70.0

    rms = np.sqrt(weighted_sum_sq / total_weighted)
    return 20 * np.log10(float(rms))


def apply_limiter(audio_array, threshold=0.99):
    abs_audio = np.abs(audio_array)
    mask = abs_audio > threshold
    if np.any(mask):
        out = np.where(
            abs_audio > threshold,
            np.sign(audio_array)
            * (
                threshold
                + (1 - threshold) * np.tanh((abs_audio - threshold) / (1 - threshold))
            ),
            audio_array,
        )
        return out
    return audio_array


def apply_multiband_compression(audio_array, sr, intensity=0.5, genre_profile=None):
    return audio_array


def calculate_spectral_clash(outro_array, intro_array, sr):
    return {"low": 1.0, "mid": 1.0, "high": 1.0}


def _apply_bass_swap_transition(outro_array, intro_array, sr, **kwargs):
    """The core DJ transition: bass swap + volume crossfade.

    This is how a real DJ mixes on a standard 2-channel mixer:
    1. Intro starts playing with its LOW EQ cut (bass at -inf)
    2. As the mix progresses, the intro's bass is brought in
    3. Simultaneously the outro's bass is cut
    4. The volume fader crossfades the mid/high smoothly
    5. The "bass swap" moment is where the kick drum changes over

    NO filter sweeps. NO resonant sweeps. Just EQ and faders.
    """
    out_len = outro_array.shape[1] if outro_array.ndim == 2 else len(outro_array)
    in_len = intro_array.shape[1] if intro_array.ndim == 2 else len(intro_array)
    if out_len == 0 or in_len == 0:
        return np.copy(outro_array), np.copy(intro_array)

    # --- SPECTRAL: bass management only (single filter call, no per-block) ---

    # Build separate gain curves for outro and intro (they may differ in length)
    x_out = np.linspace(0, 1, out_len)
    outro_bass_curve = np.ones(out_len)
    mask = x_out > 0.4
    outro_bass_curve[mask] = 0.5 * (1 + np.cos(np.pi * (x_out[mask] - 0.4) / 0.4))
    outro_bass_curve[x_out > 0.8] = 0.02

    x_in = np.linspace(0, 1, in_len)
    intro_bass_curve = np.zeros(in_len)
    mask = x_in > 0.3
    intro_bass_curve[mask] = 0.5 * (1 - np.cos(np.pi * (x_in[mask] - 0.3) / 0.4))
    intro_bass_curve[x_in > 0.7] = 1.0

    # Apply bass management using zero-phase subtraction crossover.
    # Uses filtfilt (zero-phase) instead of lfilter to avoid the +2.1dB
    # boost around 150-200Hz caused by lfilter's group delay misaligning
    # the subtracted bass with the original signal.
    # Formula: output = audio - bass * (1 - curve)
    # When curve=1.0: output = audio (no change, exact)
    # When curve=0.0: output = audio - bass (highpass effect)
    try:
        from scipy.signal import filtfilt, butter

        b_lo, a_lo = butter(2, 150.0 / (0.5 * sr), btype="low")

        # Extract bass and subtract with gain curve
        f_m = np.zeros_like(outro_array)
        f_n = np.zeros_like(intro_array)

        if outro_array.ndim == 2:
            for ch in range(outro_array.shape[0]):
                bass = filtfilt(b_lo, a_lo, outro_array[ch])
                f_m[ch] = outro_array[ch] - bass * (1 - outro_bass_curve)
        else:
            bass = filtfilt(b_lo, a_lo, outro_array)
            f_m = outro_array - bass * (1 - outro_bass_curve)

        if intro_array.ndim == 2:
            for ch in range(intro_array.shape[0]):
                bass = filtfilt(b_lo, a_lo, intro_array[ch])
                f_n[ch] = intro_array[ch] - bass * (1 - intro_bass_curve)
        else:
            bass = filtfilt(b_lo, a_lo, intro_array)
            f_n = intro_array - bass * (1 - intro_bass_curve)

        return f_m, f_n

    except ImportError:
        # No scipy: just return unmodified -- the volume crossfade will do all the work
        return np.copy(outro_array), np.copy(intro_array)


def apply_bass_swap(outro, intro, sr, **kwargs):
    return apply_dsp_filter(outro, sr, "highpass", 150.0), apply_dsp_filter(
        intro, sr, "lowpass", 150.0
    )


def apply_echo_out(outro, sr, **kwargs):
    return outro, None


def apply_hpf_sweep(outro, sr, **kwargs):
    return outro, None
