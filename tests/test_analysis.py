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
