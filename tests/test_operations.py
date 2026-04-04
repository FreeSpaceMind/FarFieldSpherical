"""
Tests for basic pattern operations on a loaded pattern.
"""
import numpy as np
import pytest

from farfield_spherical.io.readers import read_cut
from .conftest import CUT_FILE, FREQ_8GHZ, requires_cut


@requires_cut
class TestPolarizationAssignment:
    @pytest.fixture
    def pattern(self):
        return read_cut(CUT_FILE, frequency_start=FREQ_8GHZ, frequency_end=FREQ_8GHZ)

    def test_initial_polarization(self, pattern):
        assert pattern.polarization == 'x'

    def test_co_pol_shape_matches_e_theta(self, pattern):
        assert pattern.data.e_co.shape == pattern.data.e_theta.shape

    def test_cx_pol_shape_matches_e_phi(self, pattern):
        assert pattern.data.e_cx.shape == pattern.data.e_phi.shape

    def test_change_to_rhcp(self, pattern):
        pattern.change_polarization('rhcp')
        assert pattern.polarization == 'rhcp'
        assert np.any(np.abs(pattern.data.e_co.values) > 0)

    def test_change_to_theta(self, pattern):
        pattern.change_polarization('theta')
        assert pattern.polarization == 'theta'
        # e_co should equal e_theta for theta polarization
        np.testing.assert_allclose(
            pattern.data.e_co.values,
            pattern.data.e_theta.values
        )

    def test_polarization_roundtrip(self, pattern):
        """Change polarization and change back — e_theta/e_phi should be preserved."""
        original_e_theta = pattern.data.e_theta.values.copy()
        original_e_phi = pattern.data.e_phi.values.copy()

        pattern.change_polarization('rhcp')
        pattern.change_polarization('x')

        np.testing.assert_allclose(
            pattern.data.e_theta.values, original_e_theta, atol=1e-10
        )
        np.testing.assert_allclose(
            pattern.data.e_phi.values, original_e_phi, atol=1e-10
        )


@requires_cut
class TestGainDb:
    @pytest.fixture
    def pattern(self):
        return read_cut(CUT_FILE, frequency_start=FREQ_8GHZ, frequency_end=FREQ_8GHZ)

    def test_returns_xarray(self, pattern):
        import xarray as xr
        gain = pattern.get_gain_db()
        assert isinstance(gain, xr.DataArray)

    def test_shape_matches_field(self, pattern):
        gain = pattern.get_gain_db()
        assert gain.shape == pattern.data.e_theta.shape

    def test_values_are_finite_where_nonzero(self, pattern):
        gain = pattern.get_gain_db()
        # At least half of gain values should be finite (pattern has signal content)
        assert np.sum(np.isfinite(gain.values)) > gain.values.size / 2

    def test_co_pol_gain_returns_xarray(self, pattern):
        import xarray as xr
        gain_co = pattern.get_gain_db(component='e_co')
        assert isinstance(gain_co, xr.DataArray)
        assert gain_co.shape == pattern.data.e_theta.shape


@requires_cut
class TestPhaseExtraction:
    @pytest.fixture
    def pattern(self):
        return read_cut(CUT_FILE, frequency_start=FREQ_8GHZ, frequency_end=FREQ_8GHZ)

    def test_get_phase_returns_xarray(self, pattern):
        import xarray as xr
        phase = pattern.get_phase()
        assert isinstance(phase, xr.DataArray)

    def test_phase_shape(self, pattern):
        phase = pattern.get_phase()
        assert phase.shape == pattern.data.e_theta.shape

    def test_phase_values_in_degrees(self, pattern):
        # get_phase() returns degrees in range (-180, 180]
        phase = pattern.get_phase()
        assert phase.values.min() >= -180.0 - 1e-6
        assert phase.values.max() <= 180.0 + 1e-6


@requires_cut
class TestRotateStub:
    def test_rotate_raises_not_implemented(self):
        pattern = read_cut(CUT_FILE, frequency_start=FREQ_8GHZ, frequency_end=FREQ_8GHZ)
        with pytest.raises(NotImplementedError):
            pattern.rotate(0, 0, 0)
