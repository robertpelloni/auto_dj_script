import numpy as np
import pytest
from autodj.analysis import is_harmonically_compatible, get_camelot_key

def test_camelot_mapping():
    assert get_camelot_key('A Minor') == '8A'
    assert get_camelot_key('C Major') == '8B'
    assert get_camelot_key('Non-existent') == 'Unknown'

def test_harmonic_compatibility():
    # Same key
    assert is_harmonically_compatible('A Minor', 'A Minor') is True
    # Relative major/minor
    assert is_harmonically_compatible('A Minor', 'C Major') is True
    # Adjacent on wheel (8A -> 9A)
    assert is_harmonically_compatible('A Minor', 'E Minor') is True
    # Adjacent on wheel (8A -> 7A)
    assert is_harmonically_compatible('A Minor', 'D Minor') is True
    # Far away
    assert is_harmonically_compatible('A Minor', 'D# Minor') is False

def test_wrap_around():
    # 12A (C# Minor) to 1A (G# Minor)
    assert is_harmonically_compatible('C# Minor', 'G# Minor') is True

def test_dynamic_transition_logic():
    from autodj.analysis import calculate_dynamic_transition
    sr = 44100
    bpm = 120
    # Mock some audio arrays
    y = np.zeros(sr * 5)

    # Low activity (all zeros) -> 32 bars
    res = calculate_dynamic_transition(y, y, sr, bpm, 4)
    assert res == 32

def test_genre_archetype_v3():
    from autodj.analysis import get_genre_archetype
    sr = 22050
    # Generate some white noise
    y = np.random.uniform(-1, 1, sr * 5)

    # High energy profile (high centroid from white noise)
    genre, rationale = get_genre_archetype(y, sr, bpm=145)
    assert genre in ('Full-On', 'High-Energy')
    assert any(k in rationale.lower() for k in ("high-energy", "high spectral centroid", "psytrance"))

    # Low BPM / low energy
    y_low = np.sin(2 * np.pi * 440 * np.linspace(0, 5, sr * 5))
    genre_low, rationale_low = get_genre_archetype(y_low, sr, bpm=90)
    assert genre_low == 'Ambient'


def test_kick_interval_bpm_no_octave_error():
    from autodj.analysis import get_native_bpm
    sr = 22050
    duration = 16.0
    target_bpm = 145.0
    y = np.zeros(int(sr * duration), dtype=np.float32)
    beat_samples = int(round(sr * 60.0 / target_bpm))
    kick_len = int(sr * 0.08)
    t_kick = np.arange(kick_len) / sr
    kick = (np.sin(2 * np.pi * 65.0 * t_kick) * np.exp(-t_kick * 35.0)).astype(np.float32)
    for start in range(0, len(y) - kick_len, beat_samples):
        y[start : start + kick_len] += kick

    bpm, _, _ = get_native_bpm(y, sr)
    assert abs(bpm - target_bpm) < 2.0
