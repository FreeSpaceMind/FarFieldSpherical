"""
Tests for FarFieldSpherical.rotate using an analytic dipole field.

For a short dipole with moment d the far field is E(r) = d - (d.r) r. Rotating
the antenna by R gives a dipole with moment R d, so the rotated pattern has a
closed form against which the interpolated result can be checked.
"""
import logging

import numpy as np
import pytest

from farfield_spherical import FarFieldSpherical
from farfield_spherical.pattern_operations import _rotation_matrix
from .test_coordinate_transforms import analytic_fields, D

FREQS = np.array([8e9, 10e9])
TH_FINE = np.arange(0, 181, 1.0)
PHI_FINE = np.arange(0, 360, 2.0)


def make(theta_deg, phi_deg, moments=D):
    e_theta, e_phi = analytic_fields(theta_deg, phi_deg, moments)
    return FarFieldSpherical(np.asarray(theta_deg, float), np.asarray(phi_deg, float),
                             FREQS, e_theta, e_phi, polarization='x')


def assert_matches_rotated_dipole(pattern, alpha, beta, gamma, atol):
    R = _rotation_matrix(alpha, beta, gamma)
    e_theta, e_phi = analytic_fields(pattern.theta_angles, pattern.phi_angles, D @ R.T)
    np.testing.assert_allclose(pattern.data.e_theta.values, e_theta, atol=atol)
    np.testing.assert_allclose(pattern.data.e_phi.values, e_phi, atol=atol)


class TestRotate:
    def test_identity_is_exact(self):
        p = make(TH_FINE, PHI_FINE)
        before = p.data.e_theta.values.copy()
        p.rotate(0, 0, 0)
        np.testing.assert_array_equal(p.data.e_theta.values, before)

    @pytest.mark.parametrize('angles', [(30, 0, 0), (0, 45, 0), (0, 0, 60), (20, -35, 50)])
    def test_matches_analytic_rotation(self, angles):
        p = make(TH_FINE, PHI_FINE)
        p.rotate(*angles)
        # Linear interpolation on a 1 x 2 degree grid of a unit-amplitude field
        assert_matches_rotated_dipole(p, *angles, atol=2e-3)

    def test_cubic_is_more_accurate(self):
        p_lin = make(TH_FINE, PHI_FINE)
        p_cub = make(TH_FINE, PHI_FINE)
        p_lin.rotate(20, -35, 50)
        p_cub.rotate(20, -35, 50, method='cubic')
        R = _rotation_matrix(20, -35, 50)
        ref, _ = analytic_fields(TH_FINE, PHI_FINE, D @ R.T)
        err_lin = np.abs(p_lin.data.e_theta.values - ref).max()
        err_cub = np.abs(p_cub.data.e_theta.values - ref).max()
        assert err_cub < err_lin
        assert err_cub < 1e-4

    def test_roll_by_grid_step_equals_phi_shift(self):
        gamma = 90.0
        p_rot = make(TH_FINE, PHI_FINE)
        p_shift = make(TH_FINE, PHI_FINE)
        p_rot.rotate(0, 0, gamma)
        p_shift.shift_phi_origin(gamma)
        np.testing.assert_allclose(p_rot.data.e_theta.values, p_shift.data.e_theta.values, atol=1e-5)
        np.testing.assert_allclose(p_rot.data.e_phi.values, p_shift.data.e_phi.values, atol=1e-5)

    def test_inverse_rotation_restores(self):
        p = make(TH_FINE, PHI_FINE)
        before_theta = p.data.e_theta.values.copy()
        p.rotate(0, 30, 0)
        p.rotate(0, -30, 0)
        np.testing.assert_allclose(p.data.e_theta.values, before_theta, atol=4e-3)

    def test_boresight_lands_where_documented(self):
        """A narrow beam about +z moves to theta_0, phi_0 given in the docstring."""
        th = np.radians(TH_FINE)[:, None]
        beam = np.exp(-(th / np.radians(12.0)) ** 2) * np.ones((1, len(PHI_FINE)))
        e_theta = np.stack([beam, beam])
        e_phi = np.zeros_like(e_theta)
        for alpha, beta in [(30, 0), (0, 30), (25, -40)]:
            p = FarFieldSpherical(TH_FINE, PHI_FINE, FREQS, e_theta, e_phi, polarization='x')
            p.rotate(alpha, beta, 0)
            mag = np.abs(p.data.e_theta.values[0]) ** 2 + np.abs(p.data.e_phi.values[0]) ** 2
            i, j = np.unravel_index(np.argmax(mag), mag.shape)
            a, b = np.radians(alpha), np.radians(beta)
            theta0 = np.degrees(np.arccos(np.cos(a) * np.cos(b)))
            phi0 = np.degrees(np.arctan2(-np.sin(b), -np.sin(a) * np.cos(b))) % 360
            assert abs(TH_FINE[i] - theta0) <= 1.0, (alpha, beta, TH_FINE[i], theta0)
            assert abs((PHI_FINE[j] - phi0 + 180) % 360 - 180) <= 2.0, (alpha, beta, PHI_FINE[j], phi0)

    def test_central_format_input(self):
        """Central input stays central and matches the analytic field on its own grid."""
        p = make(np.arange(-180, 181, 1.0), np.arange(0, 180, 2.0))
        p.rotate(20, -35, 50)
        assert p.theta_angles.min() < 0
        assert p.phi_angles.max() < 180
        assert_matches_rotated_dipole(p, 20, -35, 50, atol=2e-3)

    def test_ffd_style_phi_grid_preserved(self):
        """phi -180..180 with duplicated endpoint keeps its grid after rotation."""
        phi = np.arange(-180, 181, 2.0)
        p = make(TH_FINE, phi)
        p.rotate(0, 20, 0)
        np.testing.assert_array_equal(p.phi_angles, phi)
        assert_matches_rotated_dipole(p, 0, 20, 0, atol=2e-3)

    def test_partial_sphere_warns_and_zero_fills(self, caplog):
        p = make(np.arange(0, 91, 1.0), PHI_FINE)
        with caplog.at_level(logging.WARNING):
            p.rotate(30, 0, 0)
        assert 'outside' in caplog.text
        # Directions that rotate to the back hemisphere have no data
        assert np.any(p.data.e_theta.values == 0)

    def test_polarization_and_metadata(self):
        p = make(TH_FINE, PHI_FINE)
        p.rotate(10, 0, 0)
        assert p.polarization == 'x'
        assert np.all(np.isfinite(p.data.e_co.values))
        op = p.metadata['operations'][-1]
        assert op['type'] == 'rotate' and op['alpha'] == 10.0

    def test_isometric_rotation_consistent_with_matrix(self):
        u, v, w = 0.2, -0.4, 0.8
        from farfield_spherical import isometric_rotation
        out = np.array(isometric_rotation(u, v, w, 20, -35, 50))
        np.testing.assert_allclose(out, _rotation_matrix(20, -35, 50) @ np.array([u, v, w]))
