"""
I/O roundtrip tests: read → write → read and compare field data.
"""
import os
import tempfile
import numpy as np
import pytest

from farfield_spherical.io.readers import read_cut
from farfield_spherical.io.npz_format import save_pattern_npz, load_pattern_npz
from .conftest import CUT_FILE, FREQ_8GHZ, requires_cut


@requires_cut
class TestNpzRoundtrip:
    @pytest.fixture(scope='class')
    def original(self):
        return read_cut(CUT_FILE, frequency_start=FREQ_8GHZ, frequency_end=FREQ_8GHZ)

    def _roundtrip(self, original):
        """Save to NPZ and reload, returning (loaded_pattern, metadata)."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            save_pattern_npz(original, path)
            loaded, metadata = load_pattern_npz(path)
        finally:
            os.unlink(path)
        return loaded, metadata

    def test_save_and_load_returns_pattern(self, original):
        from farfield_spherical import FarFieldSpherical
        loaded, _ = self._roundtrip(original)
        assert isinstance(loaded, FarFieldSpherical)

    def test_e_theta_preserved(self, original):
        loaded, _ = self._roundtrip(original)
        np.testing.assert_allclose(
            loaded.data.e_theta.values,
            original.data.e_theta.values,
            rtol=1e-5,
            atol=1e-30
        )

    def test_e_phi_preserved(self, original):
        loaded, _ = self._roundtrip(original)
        np.testing.assert_allclose(
            loaded.data.e_phi.values,
            original.data.e_phi.values,
            rtol=1e-5,
            atol=1e-30
        )

    def test_theta_angles_preserved(self, original):
        loaded, _ = self._roundtrip(original)
        np.testing.assert_allclose(loaded.theta_angles, original.theta_angles)

    def test_phi_angles_preserved(self, original):
        loaded, _ = self._roundtrip(original)
        np.testing.assert_allclose(loaded.phi_angles, original.phi_angles)

    def test_frequencies_preserved(self, original):
        loaded, _ = self._roundtrip(original)
        np.testing.assert_allclose(loaded.frequencies, original.frequencies)

    def test_polarization_preserved(self, original):
        loaded, _ = self._roundtrip(original)
        assert loaded.polarization == original.polarization

    def test_load_returns_metadata_dict(self, original):
        _, metadata = self._roundtrip(original)
        assert isinstance(metadata, dict)


@requires_cut
class TestCutWriteRoundtrip:
    """Write a pattern to a CUT file and read it back."""

    def test_write_and_read_back_field_magnitudes(self):
        original = read_cut(CUT_FILE, frequency_start=FREQ_8GHZ, frequency_end=FREQ_8GHZ)
        with tempfile.NamedTemporaryFile(suffix='.cut', delete=False) as f:
            path = f.name
        try:
            original.write_cut(path)
            loaded = read_cut(path, frequency_start=FREQ_8GHZ, frequency_end=FREQ_8GHZ)
            np.testing.assert_allclose(
                np.abs(loaded.data.e_theta.values),
                np.abs(original.data.e_theta.values),
                rtol=1e-4,
                atol=1e-30
            )
        finally:
            os.unlink(path)

    def test_write_and_read_back_dimensions(self):
        original = read_cut(CUT_FILE, frequency_start=FREQ_8GHZ, frequency_end=FREQ_8GHZ)
        with tempfile.NamedTemporaryFile(suffix='.cut', delete=False) as f:
            path = f.name
        try:
            original.write_cut(path)
            loaded = read_cut(path, frequency_start=FREQ_8GHZ, frequency_end=FREQ_8GHZ)
            assert loaded.data.e_theta.shape == original.data.e_theta.shape
        finally:
            os.unlink(path)
