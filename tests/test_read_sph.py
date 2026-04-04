"""
Tests for reading TICRA SPH (spherical wave expansion) files.
"""
import numpy as np
import pytest

from .conftest import SPH_FILE, FREQ_8GHZ, requires_sph, requires_swe


@requires_sph
@requires_swe
class TestReadTicraSph:
    @pytest.fixture(scope='class')
    def pattern(self):
        from farfield_spherical import FarFieldSpherical
        return FarFieldSpherical.from_ticra_sph(SPH_FILE, frequency=FREQ_8GHZ)

    def test_returns_farfield_object(self, pattern):
        from farfield_spherical import FarFieldSpherical
        assert isinstance(pattern, FarFieldSpherical)

    def test_has_one_frequency(self, pattern):
        assert len(pattern.frequencies) == 1

    def test_frequency_is_8ghz(self, pattern):
        assert pattern.frequencies[0] == pytest.approx(FREQ_8GHZ, rel=1e-4)

    def test_data_shape_consistent(self, pattern):
        n_freq, n_theta, n_phi = pattern.data.e_theta.shape
        assert n_freq == len(pattern.frequencies)
        assert n_theta == len(pattern.theta_angles)
        assert n_phi == len(pattern.phi_angles)

    def test_field_dtype_is_complex(self, pattern):
        assert np.iscomplexobj(pattern.data.e_theta.values)
        assert np.iscomplexobj(pattern.data.e_phi.values)

    def test_fields_not_all_zero(self, pattern):
        assert np.any(np.abs(pattern.data.e_theta.values) > 0)

    def test_has_swe_attribute(self, pattern):
        assert hasattr(pattern, 'swe')
        assert isinstance(pattern.swe, dict)
        assert len(pattern.swe) >= 1


@requires_sph
@requires_swe
class TestReadTicraSphWithCustomAngles:
    """Test from_ticra_sph with explicit theta/phi angle arrays."""

    def test_custom_theta_phi(self):
        from farfield_spherical import FarFieldSpherical
        theta = np.linspace(0, 180, 37)
        phi = np.arange(0, 360, 10.0)
        pattern = FarFieldSpherical.from_ticra_sph(
            SPH_FILE, frequency=FREQ_8GHZ, theta_angles=theta, phi_angles=phi
        )
        assert pattern.data.e_theta.shape[1] == len(theta)
        assert pattern.data.e_theta.shape[2] == len(phi)
