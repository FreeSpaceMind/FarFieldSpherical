"""
Shared fixtures and constants for FarFieldSpherical test suite.
"""
import os
import pytest

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

CUT_FILE = os.path.join(TEST_DATA_DIR, 'example.cut')
SPH_FILE = os.path.join(TEST_DATA_DIR, 'example.sph')

requires_cut = pytest.mark.skipif(
    not os.path.exists(CUT_FILE),
    reason='example.cut not found in tests/data/'
)
requires_sph = pytest.mark.skipif(
    not os.path.exists(SPH_FILE),
    reason='example.sph not found in tests/data/'
)

try:
    import swe  # noqa: F401
    SWE_AVAILABLE = True
except ImportError:
    SWE_AVAILABLE = False

requires_swe = pytest.mark.skipif(not SWE_AVAILABLE, reason='swe package not installed')

# Constants matching the example data files (sourced from spherical_wave_expansion tests)
FREQ_8GHZ = 8.0e9
N_THETA = 761        # theta points per phi cut in example.cut (-180 to 180, ~0.474° step)
N_PHI = 37           # phi cuts per frequency in example.cut
N_FREQS = 9          # number of frequency blocks in example.cut
THETA_START = -180.0
THETA_END = 180.0
