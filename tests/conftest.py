"""
Shared fixtures and constants for FarFieldSpherical test suite.
"""
import os
import pytest

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# The example .cut and .sph files are versioned in the spherical_wave_expansion
# repository (tests/example.cut, tests/example.sph), which the SWE-dependent
# tests need installed anyway. They are looked for, in order, in tests/data/
# here, in the directory named by FARFIELD_TEST_DATA, and in a sibling checkout
# of spherical_wave_expansion. X5_horn_1.ffd is a measured file that is not
# versioned anywhere; its tests run only when a copy is placed in tests/data/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIRS = [TEST_DATA_DIR]
if os.environ.get('FARFIELD_TEST_DATA'):
    _DATA_DIRS.append(os.environ['FARFIELD_TEST_DATA'])
_DATA_DIRS.append(os.path.join(os.path.dirname(_REPO_ROOT), 'spherical_wave_expansion', 'tests'))


def find_data_file(name):
    """The first existing copy of a test data file, or its tests/data/ path."""
    for directory in _DATA_DIRS:
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(TEST_DATA_DIR, name)


CUT_FILE = find_data_file('example.cut')
SPH_FILE = find_data_file('example.sph')
FFD_FILE = find_data_file('X5_horn_1.ffd')

requires_cut = pytest.mark.skipif(
    not os.path.exists(CUT_FILE),
    reason='example.cut not found (tests/data/, FARFIELD_TEST_DATA or ../spherical_wave_expansion/tests)'
)
requires_sph = pytest.mark.skipif(
    not os.path.exists(SPH_FILE),
    reason='example.sph not found (tests/data/, FARFIELD_TEST_DATA or ../spherical_wave_expansion/tests)'
)
requires_ffd = pytest.mark.skipif(
    not os.path.exists(FFD_FILE),
    reason='X5_horn_1.ffd not found in tests/data/'
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
