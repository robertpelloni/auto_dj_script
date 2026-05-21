import numpy as np
import pytest
from autodj.dsp import calculate_spectral_clash, ArchetypeRegistry

def test_spectral_clash_detection():
    sr = 22050
    # Low frequency sine
    t = np.linspace(0, 1, sr)
    low_sine = np.sin(2 * np.pi * 50 * t)
    # High frequency sine
    high_sine = np.sin(2 * np.pi * 10000 * t)

    # Clash between two low sines
    clash = calculate_spectral_clash(low_sine, low_sine, sr)
    assert clash['low'] > 1.5

    # No clash between low and high
    no_clash = calculate_spectral_clash(low_sine, high_sine, sr)
    assert no_clash['low'] < 1.5
    assert no_clash['high'] < 1.5

def test_spectral_balanced_archetype():
    plugin = ArchetypeRegistry.get('spectral_balance')
    assert plugin is not None

    sr = 22050
    t = np.linspace(0, 1, sr)
    low_sine = np.sin(2 * np.pi * 50 * t)

    # Apply to clashing bass
    f_m, f_n = plugin.apply(low_sine, low_sine, sr)

    # Outgoing track should have been high-passed (bass dipped)
    # Mean of absolute values should be significantly lower
    assert np.mean(np.abs(f_m)) < np.mean(np.abs(low_sine)) * 0.1
