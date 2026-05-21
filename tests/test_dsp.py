import numpy as np
from autodj.dsp import apply_dsp_filter

def test_filter_shape():
    sr = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sr * duration))
    # Mono signal
    y = np.sin(2 * np.pi * 440 * t)

    y_filtered = apply_dsp_filter(y, sr, 'highpass', 100)
    assert y.shape == y_filtered.shape

    # Stereo signal
    y_stereo = np.vstack([y, y])
    y_filtered_stereo = apply_dsp_filter(y_stereo, sr, 'lowpass', 1000)
    assert y_stereo.shape == y_filtered_stereo.shape

def test_multiband_compression_profiles():
    from autodj.dsp import apply_multiband_compression
    sr = 22050
    y = np.random.uniform(-0.5, 0.5, sr * 2)

    # Test Default profile
    res_def = apply_multiband_compression(y, sr, intensity=1.0)
    assert res_def.shape == y.shape

    # Test Ambient profile (should be less compressed / higher peak if makeup gain is off,
    # but here we check it runs without error)
    res_amb = apply_multiband_compression(y, sr, intensity=1.0, genre_profile='Ambient')
    assert res_amb.shape == y.shape
