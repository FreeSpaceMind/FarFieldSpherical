"""
Tests for reading GRASP CUT format files.
"""
import numpy as np
import pytest

from farfield_spherical.io.readers import read_cut
from .conftest import CUT_FILE, FREQ_8GHZ, N_THETA, N_PHI, N_FREQS, THETA_START, THETA_END, requires_cut


@requires_cut
class TestReadCut:
    @pytest.fixture(scope='class')
    def pattern(self):
        # frequency_start=frequency_end assigns all blocks the same frequency label
        return read_cut(CUT_FILE, frequency_start=FREQ_8GHZ, frequency_end=FREQ_8GHZ)

    def test_returns_farfield_object(self, pattern):
        from farfield_spherical import FarFieldSpherical
        assert isinstance(pattern, FarFieldSpherical)

    def test_frequency_count(self, pattern):
        assert len(pattern.frequencies) == N_FREQS

    def test_theta_shape(self, pattern):
        assert len(pattern.theta_angles) == N_THETA

    def test_phi_shape(self, pattern):
        assert len(pattern.phi_angles) == N_PHI

    def test_theta_range(self, pattern):
        assert pattern.theta_angles[0] == pytest.approx(THETA_START, abs=0.5)
        assert pattern.theta_angles[-1] == pytest.approx(THETA_END, abs=0.5)

    def test_data_shape(self, pattern):
        # Shape is (frequency, theta, phi)
        assert pattern.data.e_theta.shape == (N_FREQS, N_THETA, N_PHI)
        assert pattern.data.e_phi.shape == (N_FREQS, N_THETA, N_PHI)

    def test_field_dtype_is_complex(self, pattern):
        assert np.iscomplexobj(pattern.data.e_theta.values)
        assert np.iscomplexobj(pattern.data.e_phi.values)

    def test_polarization_is_x(self, pattern):
        # example.cut uses ICOMP=3 (X/Y Ludwig-3 linear), so co-pol = X
        assert pattern.polarization == 'x'

    def test_fields_not_all_zero(self, pattern):
        assert np.any(np.abs(pattern.data.e_theta.values) > 0)

    def test_phi_angles_ascending(self, pattern):
        phi = pattern.phi_angles
        assert np.all(np.diff(phi) > 0)

    def test_theta_angles_ascending(self, pattern):
        theta = pattern.theta_angles
        assert np.all(np.diff(theta) > 0)

    def test_phi_angles_in_range(self, pattern):
        phi = pattern.phi_angles
        assert phi[0] >= 0.0
        assert phi[-1] < 360.0
