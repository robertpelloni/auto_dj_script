"""Digital Signal Processing (DSP) Module | Auto DJ Script.
==============================================================
Professional DJ-style transitions, 3-band dynamics mastering, and transition archetype plugins.

Design philosophy:
- Default `progressive` archetype uses phase-coherent zero-phase (`filtfilt`) bass subtraction
  paired with equal-power volume crossfading (what a real 2-channel DJ mixer does).
- Additional registered `TransitionArchetype` plugins (`spectral_balance`, `classic`,
  `echo_out`, `hpf_sweep`, `low_cut_build`) give creative control when requested via CLI/UI.
- Multiband compression uses Linkwitz-style zero-phase 3-band splitting (`<200 Hz`, `200-3500 Hz`,
  `>3500 Hz`) with soft-knee envelope gain reduction and genre-aware thresholds.
"""

import numpy as np
from .np_signal import get_butter_coeffs, apply_iir_filter


class TransitionArchetype:
    """Base class for all transition plugins."""

    name = "Base"
    display_name = "Base Archetype"
    description = ""

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
    """Surgical silence removal using vectorized NumPy threshold scan."""
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
    """Normalize audio to target LUFS using K-weighted RMS (ITU-R BS.1770 approximation).
    Processes in chunks to handle multi-hour master streams without OOM.
    """
    if audio_array.ndim == 2:
        ch_powers = []
        for ch in range(audio_array.shape[0]):
            ch_loudness = _measure_loudness_chunked(audio_array[ch], sr)
            ch_powers.append(10 ** (ch_loudness / 10.0))
        total_power = sum(ch_powers)
        if total_power < 1e-20:
            return audio_array
        loudness = 10 * np.log10(total_power) - 0.691
    else:
        loudness = _measure_loudness_chunked(audio_array, sr)

    if loudness < -70 or np.isinf(loudness):
        return audio_array

    gain_db = target_lufs - loudness
    gain_lin = float(np.clip(10 ** (gain_db / 20.0), 0.1, 10.0))

    audio_array = audio_array * gain_lin
    peak = float(np.max(np.abs(audio_array)))
    if peak > 0.99:
        audio_array *= 0.99 / peak

    return audio_array


def _measure_loudness_chunked(audio, sr, chunk_seconds=120):
    """Measure perceptual loudness using chunked K-weighting filters with state carryover."""
    from scipy.signal import bilinear, sosfilt, sosfilt_zi

    if sr == 44100:
        b_shelf = [1.53512485, -2.69161304, 1.19839265]
        a_shelf = [1.0, -1.69004453, 0.73249962]
    else:
        b_shelf = [1.0]
        a_shelf = [1.0]

    try:
        b_hp, a_hp = bilinear([0, 1], [1, 1 / (2 * np.pi * 60)], sr=sr)
    except Exception:
        b_hp = [1.0]
        a_hp = [1.0]

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

    zi_shelf = sosfilt_zi(sos_shelf) * 0
    zi_hp = sosfilt_zi(sos_hp) * 0

    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        chunk = audio[start:end]
        filtered, zi_shelf = sosfilt(sos_shelf, chunk, zi=zi_shelf)
        filtered, zi_hp = sosfilt(sos_hp, filtered, zi=zi_hp)
        weighted_sum_sq += float(np.sum(filtered**2))
        total_weighted += end - start

    if total_weighted == 0 or weighted_sum_sq < 1e-20:
        return -70.0

    rms = np.sqrt(weighted_sum_sq / total_weighted)
    return 20 * np.log10(float(rms))


def apply_limiter(audio_array, threshold=0.99):
    """Soft-clip hyperbolic tangent peak limiter preventing inter-sample clipping."""
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


def _compress_band_1d(band: np.ndarray, sr: int, threshold_db: float, ratio: float, makeup_db: float) -> np.ndarray:
    """Applies smooth RMS-envelope soft-knee compression to a 1D band signal."""
    if band.size == 0 or ratio <= 1.0:
        return band
    from scipy.signal import lfilter

    # ~25ms envelope smoothing window prevents waveform-rate distortion on sub-bass
    win = max(1, int(sr * 0.025))
    alpha = 1.0 - np.exp(-1.0 / float(win))
    env_sq = lfilter([alpha], [1.0, -(1.0 - alpha)], band.astype(np.float64) ** 2)
    env_db = 10.0 * np.log10(np.maximum(env_sq, 1e-12))

    over_db = np.maximum(0.0, env_db - threshold_db)
    gain_red_db = -over_db * (1.0 - 1.0 / ratio) + makeup_db
    gain_lin = (10.0 ** (gain_red_db / 20.0)).astype(band.dtype, copy=False)
    return band * gain_lin


def apply_multiband_compression(audio_array, sr, intensity=0.5, genre_profile=None):
    """3-band zero-phase dynamics mastering (`<200 Hz`, `200-3500 Hz`, `>3500 Hz`).

    Why zero-phase subtraction split:
    Summing independent minimum-phase Butterworth bandpasses causes phase cancellation notches
    at the crossover frequencies. Extracting `low` and `low+mid` with zero-phase `filtfilt`
    and deriving the complementary bands algebraically guarantees exact unity reconstruction
    (`low + mid + high == audio_array` to floating-point precision) when `intensity == 0`.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity <= 0.01 or audio_array.size < 32:
        return audio_array

    profiles = {
        "Ambient": {"thresh": (-16.0, -18.0, -20.0), "ratio": (1.6, 1.4, 1.4), "makeup": (0.5, 0.4, 0.3)},
        "Downtempo": {"thresh": (-16.0, -18.0, -20.0), "ratio": (1.8, 1.6, 1.5), "makeup": (0.8, 0.6, 0.5)},
        "Minimal": {"thresh": (-18.0, -19.0, -21.0), "ratio": (2.5, 2.0, 1.8), "makeup": (1.2, 0.9, 0.7)},
        "Progressive": {"thresh": (-18.0, -20.0, -22.0), "ratio": (2.8, 2.2, 2.0), "makeup": (1.4, 1.0, 0.8)},
        "Full-On": {"thresh": (-19.0, -21.0, -22.0), "ratio": (3.2, 2.5, 2.2), "makeup": (1.6, 1.2, 1.0)},
        "Hi-Tech": {"thresh": (-20.0, -21.0, -23.0), "ratio": (3.5, 2.8, 2.4), "makeup": (1.8, 1.3, 1.1)},
    }
    prof = profiles.get(str(genre_profile or ""), profiles["Progressive"])

    try:
        from scipy.signal import butter, filtfilt

        nyq = 0.5 * float(sr)
        f_lo = min(200.0, nyq * 0.4)
        f_hi = min(3500.0, nyq * 0.85)
        b_lo, a_lo = butter(2, f_lo / nyq, btype="low")
        b_lm, a_lm = butter(2, f_hi / nyq, btype="low")

        def _proc_1d(sig: np.ndarray) -> np.ndarray:
            low = filtfilt(b_lo, a_lo, sig)
            low_mid = filtfilt(b_lm, a_lm, sig)
            mid = low_mid - low
            high = sig - low_mid

            c_low = _compress_band_1d(low, sr, prof["thresh"][0], 1.0 + (prof["ratio"][0] - 1.0) * intensity, prof["makeup"][0] * intensity)
            c_mid = _compress_band_1d(mid, sr, prof["thresh"][1], 1.0 + (prof["ratio"][1] - 1.0) * intensity, prof["makeup"][1] * intensity)
            c_high = _compress_band_1d(high, sr, prof["thresh"][2], 1.0 + (prof["ratio"][2] - 1.0) * intensity, prof["makeup"][2] * intensity)
            wet = c_low + c_mid + c_high
            return (sig * (1.0 - intensity) + wet * intensity).astype(sig.dtype, copy=False)

        if audio_array.ndim == 2:
            out = np.empty_like(audio_array)
            for ch in range(audio_array.shape[0]):
                out[ch] = _proc_1d(audio_array[ch])
            return apply_limiter(out)
        return apply_limiter(_proc_1d(audio_array))
    except Exception:
        return audio_array


def calculate_spectral_clash(outro_array, intro_array, sr):
    """Analyzes frequency band overlap between two audio segments (Low, Mid, High)."""

    def _get_bands(arr):
        low = apply_dsp_filter(arr, sr, "lowpass", 200.0)
        mid_high = apply_dsp_filter(arr, sr, "highpass", 200.0)
        mid = apply_dsp_filter(mid_high, sr, "lowpass", 3000.0)
        high = apply_dsp_filter(arr, sr, "highpass", 3000.0)
        return (
            float(np.mean(np.abs(low))),
            float(np.mean(np.abs(mid))),
            float(np.mean(np.abs(high))),
        )

    o_l, o_m, o_h = _get_bands(outro_array)
    i_l, i_m, i_h = _get_bands(intro_array)

    return {
        "low": (o_l + i_l) / max(o_l, i_l, 1e-6),
        "mid": (o_m + i_m) / max(o_m, i_m, 1e-6),
        "high": (o_h + i_h) / max(o_h, i_h, 1e-6),
    }


def _zero_phase_filter(arr: np.ndarray, sr: int, cutoff: float, btype: str = "low", order: int = 2) -> np.ndarray:
    """Applies zero-phase Butterworth filtering along time axis for mono or stereo arrays."""
    try:
        from scipy.signal import butter, filtfilt

        nyq = 0.5 * float(sr)
        norm_c = float(np.clip(cutoff / nyq, 0.005, 0.98))
        b, a = butter(order, norm_c, btype=btype)
        if arr.ndim == 2:
            out = np.zeros_like(arr)
            for ch in range(arr.shape[0]):
                out[ch] = filtfilt(b, a, arr[ch])
            return out
        return filtfilt(b, a, arr)
    except Exception:
        return apply_dsp_filter(arr, sr, "lowpass" if btype == "low" else "highpass", cutoff)


def _apply_bass_swap_transition(outro_array, intro_array, sr, **kwargs):
    """The core DJ transition: zero-phase bass swap + clean mid/high preservation."""
    out_len = outro_array.shape[1] if outro_array.ndim == 2 else len(outro_array)
    in_len = intro_array.shape[1] if intro_array.ndim == 2 else len(intro_array)
    if out_len == 0 or in_len == 0:
        return np.copy(outro_array), np.copy(intro_array)

    bass_style = kwargs.get("bass_transition", "swap")
    cutoff = float(kwargs.get("highpass", 150.0) or 150.0)

    x_out = np.linspace(0, 1, out_len)
    x_in = np.linspace(0, 1, in_len)

    if bass_style == "fade":
        outro_bass_curve = np.cos(np.pi * x_out / 2.0)
        intro_bass_curve = np.sin(np.pi * x_in / 2.0)
    else:
        outro_bass_curve = np.ones(out_len)
        mask_o = x_out > 0.4
        outro_bass_curve[mask_o] = 0.5 * (1 + np.cos(np.pi * (x_out[mask_o] - 0.4) / 0.4))
        outro_bass_curve[x_out > 0.8] = 0.02

        intro_bass_curve = np.zeros(in_len)
        mask_i = x_in > 0.3
        intro_bass_curve[mask_i] = 0.5 * (1 - np.cos(np.pi * (x_in[mask_i] - 0.3) / 0.4))
        intro_bass_curve[x_in > 0.7] = 1.0

    o_bass = _zero_phase_filter(outro_array, sr, cutoff, btype="low", order=2)
    i_bass = _zero_phase_filter(intro_array, sr, cutoff, btype="low", order=2)

    if outro_array.ndim == 2:
        f_m = outro_array - o_bass * (1.0 - outro_bass_curve[np.newaxis, :])
    else:
        f_m = outro_array - o_bass * (1.0 - outro_bass_curve)

    if intro_array.ndim == 2:
        f_n = intro_array - i_bass * (1.0 - intro_bass_curve[np.newaxis, :])
    else:
        f_n = intro_array - i_bass * (1.0 - intro_bass_curve)

    return f_m, f_n


def apply_bass_swap(outro, intro, sr, **kwargs):
    return apply_dsp_filter(outro, sr, "highpass", 150.0), apply_dsp_filter(
        intro, sr, "lowpass", 150.0
    )


def apply_echo_out(outro, sr, bpm=145.0, feedback=0.42, taps=4, **kwargs):
    """Applies a tempo-synced dotted-eighth dub delay wash over the second half of `outro`."""
    n = outro.shape[1] if outro.ndim == 2 else len(outro)
    if n == 0:
        return np.copy(outro), None

    # Dotted-eighth delay (3/4 of a beat) creates classic rhythmic psychedelic tail
    delay_samples = max(1, int((60.0 / max(bpm, 60.0)) * 0.75 * sr))
    wet = np.zeros_like(outro, dtype=np.float64)
    high_outro = _zero_phase_filter(outro, sr, 280.0, btype="high", order=2)

    for tap in range(1, taps + 1):
        shift = delay_samples * tap
        if shift >= n:
            break
        gain = feedback**tap
        if outro.ndim == 2:
            wet[:, shift:] += high_outro[:, : n - shift] * gain
        else:
            wet[shift:] += high_outro[: n - shift] * gain

    ramp = np.linspace(0.0, 0.85, n)
    if outro.ndim == 2:
        mixed = outro + wet * ramp[np.newaxis, :]
    else:
        mixed = outro + wet * ramp
    return apply_limiter(mixed.astype(outro.dtype, copy=False)), None


def apply_hpf_sweep(outro, sr, **kwargs):
    """Progressive high-pass wash on `outro` using smooth multi-band subtraction."""
    n = outro.shape[1] if outro.ndim == 2 else len(outro)
    if n == 0:
        return np.copy(outro), None

    low_band = _zero_phase_filter(outro, sr, 250.0, btype="low", order=2)
    mid_band = _zero_phase_filter(outro, sr, 1200.0, btype="low", order=2) - low_band
    x = np.linspace(0.0, 1.0, n)
    low_cut = np.clip(x * 1.5, 0.0, 1.0)
    mid_cut = np.clip((x - 0.35) * 1.4, 0.0, 0.85)

    if outro.ndim == 2:
        swept = outro - low_band * low_cut[np.newaxis, :] - mid_band * mid_cut[np.newaxis, :]
    else:
        swept = outro - low_band * low_cut - mid_band * mid_cut
    return swept.astype(outro.dtype, copy=False), None


@ArchetypeRegistry.register
class ProgressiveBassSwap(TransitionArchetype):
    """Default club-grade zero-phase DJ bass swap / fade transition."""

    name = "progressive"
    display_name = "Progressive DJ Bass Swap"
    description = "Zero-phase sub-150Hz bass handoff with clean mid/high equal-power crossfade."

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        return _apply_bass_swap_transition(outro_array, intro_array, sr, **kwargs)


@ArchetypeRegistry.register
class SpectralBalancedMix(TransitionArchetype):
    """Adaptive Spectral Balancing archetype resolving band-specific clashes."""

    name = "spectral_balance"
    display_name = "Adaptive Spectral Balancing"
    description = "Measures Low/Mid/High band overlap and dynamically carves clashing frequencies."

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        clashes = calculate_spectral_clash(outro_array, intro_array, sr)

        f_m = np.copy(outro_array)
        f_n = np.copy(intro_array)

        if clashes["low"] > 1.5:
            f_m = _zero_phase_filter(f_m, sr, 150.0, btype="high", order=4)

        if clashes["high"] > 1.5:
            f_m = _zero_phase_filter(f_m, sr, min(5000.0, 0.45 * sr), btype="low", order=4)

        return f_m, f_n


@ArchetypeRegistry.register
class ClassicCrossfade(TransitionArchetype):
    """Unfiltered equal-power blend for ambient, chillout, or beatless transitions."""

    name = "classic"
    display_name = "Classic Unfiltered Blend"
    description = "Pure equal-power full-spectrum crossfade without EQ carving (ideal for ambient/chill)."

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        return np.copy(outro_array), np.copy(intro_array)


@ArchetypeRegistry.register
class EchoOutTransition(TransitionArchetype):
    """Tempo-synced dub delay tail on outgoing track + clean bass-managed incoming entry."""

    name = "echo_out"
    display_name = "Dub Delay / Echo Out"
    description = "Adds a dotted-eighth high-passed echo tail to the outgoing track during bass handoff."

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        f_m, f_n = _apply_bass_swap_transition(outro_array, intro_array, sr, **kwargs)
        f_m, _ = apply_echo_out(f_m, sr, bpm=float(kwargs.get("bpm", 145.0) or 145.0))
        return f_m, f_n


@ArchetypeRegistry.register
class HPFSweepTransition(TransitionArchetype):
    """Progressive high-pass wash on outgoing track while incoming drops in."""

    name = "hpf_sweep"
    display_name = "High-Pass Wash Out"
    description = "Gradually strips sub-bass and low-mids from the outgoing track for airy lift transitions."

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        swept_out, _ = apply_hpf_sweep(outro_array, sr)
        _, f_n = _apply_bass_swap_transition(outro_array, intro_array, sr, **kwargs)
        return swept_out, f_n


@ArchetypeRegistry.register
class LowCutBuildTransition(TransitionArchetype):
    """Cuts both sub-basses briefly before dropping incoming full-range bass at 65% mark."""

    name = "low_cut_build"
    display_name = "Low-Cut Tension Drop"
    description = " Pulls low-end out in the middle of the phrase to build tension before dropping the incoming kick."

    @staticmethod
    def apply(outro_array, intro_array, sr, **kwargs):
        out_len = outro_array.shape[1] if outro_array.ndim == 2 else len(outro_array)
        in_len = intro_array.shape[1] if intro_array.ndim == 2 else len(intro_array)
        if out_len == 0 or in_len == 0:
            return np.copy(outro_array), np.copy(intro_array)

        o_bass = _zero_phase_filter(outro_array, sr, 160.0, btype="low", order=2)
        i_bass = _zero_phase_filter(intro_array, sr, 160.0, btype="low", order=2)

        xo = np.linspace(0.0, 1.0, out_len)
        xi = np.linspace(0.0, 1.0, in_len)
        o_curve = np.clip(1.0 - xo * 2.2, 0.0, 1.0)
        i_curve = np.clip((xi - 0.62) * 5.0, 0.0, 1.0)

        f_m = outro_array - o_bass * (1.0 - o_curve[np.newaxis, :] if outro_array.ndim == 2 else (1.0 - o_curve))
        f_n = intro_array - i_bass * (1.0 - i_curve[np.newaxis, :] if intro_array.ndim == 2 else (1.0 - i_curve))
        return f_m, f_n
