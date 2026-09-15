"""
Tests for shift_theta_origin (measurement correction) using the analytic
dipole field from test_coordinate_transforms.

In central format every phi cut is a closed great circle, so a theta-origin
shift is a periodic resampling of each cut: new(theta) = old(theta + offset).
A sided pattern must give the same answer as the equivalent central pattern,
on its own grid.
"""
import logging

import numpy as np
import pytest

from farfield_spherical import FarFieldSpherical
from .test_coordinate_transforms import analytic_fields, make, assert_grid, FREQS

TH_C = np.arange(-180, 181, 1.0)
PHI_C = np.arange(0, 180, 10.0)
TH_S = np.arange(0, 181, 1.0)
PHI_S = np.arange(-180, 181, 10.0)   # .ffd style, duplicated endpoint


def rolled_reference(theta, phi, offset):
    """Analytic field sampled at theta + offset on a central grid: the exact answer."""
    return analytic_fields(theta + offset, phi)


class TestCentral:
    def test_grid_multiple_is_exact_roll(self):
        p = make(TH_C, PHI_C)
        p.shift_theta_origin(7.0)
        e_theta, e_phi = rolled_reference(TH_C, PHI_C, 7.0)
        np.testing.assert_allclose(p.data.e_theta.values, e_theta, atol=1e-5)
        np.testing.assert_allclose(p.data.e_phi.values, e_phi, atol=1e-5)

    def test_fractional_shift_interpolates(self):
        p = make(TH_C, PHI_C)
        p.shift_theta_origin(2.4)
        e_theta, e_phi = rolled_reference(TH_C, PHI_C, 2.4)
        # Amplitude/phase interpolation has a cusp at field zero crossings,
        # which bounds the error at a few 1e-3 on a unit-amplitude field.
        np.testing.assert_allclose(p.data.e_theta.values, e_theta, atol=3e-3)
        np.testing.assert_allclose(p.data.e_phi.values, e_phi, atol=3e-3)

    def test_wraps_periodically(self):
        """Data shifted past theta = -180 reappears at +180: nothing is lost."""
        p = make(TH_C, PHI_C)
        p.shift_theta_origin(10.0)
        e_theta, _ = rolled_reference(TH_C, PHI_C, 10.0)
        # the last 10 samples come from the other end of the cut
        np.testing.assert_allclose(p.data.e_theta.values[:, -11:, :], e_theta[:, -11:, :], atol=1e-5)

    def test_shift_by_360_is_identity(self):
        p = make(TH_C, PHI_C)
        before = p.data.e_theta.values.copy()
        p.shift_theta_origin(360.0)
        np.testing.assert_allclose(p.data.e_theta.values, before, atol=1e-5)

    def test_round_trip(self):
        p = make(TH_C, PHI_C)
        before = p.data.e_theta.values.copy()
        p.shift_theta_origin(3.3)
        p.shift_theta_origin(-3.3)
        np.testing.assert_allclose(p.data.e_theta.values, before, atol=6e-3)

    def test_half_circle_cut_extends_ends(self):
        """theta -90..90 is not a closed circle: ends are extended, not wrapped."""
        th = np.arange(-90, 91, 1.0)
        p = make(th, PHI_C)
        p.shift_theta_origin(5.0)
        e_theta, _ = analytic_fields(th + 5.0, PHI_C)
        # interior matches the shifted field ...
        np.testing.assert_allclose(p.data.e_theta.values[:, :-6, :], e_theta[:, :-6, :], atol=2e-4)
        # ... and the last samples hold the end value
        end = p.data.e_theta.values[:, -1, :]
        np.testing.assert_allclose(p.data.e_theta.values[:, -3, :], end, atol=1e-5)


class TestSided:
    def test_matches_central_on_own_grid(self):
        """Sided .ffd-style pattern shifted == central pattern shifted, grid preserved."""
        sided = make(TH_S, PHI_S)
        sided.shift_theta_origin(4.0)
        assert_grid(sided, TH_S, PHI_S)

        central = make(TH_C, np.arange(0, 180, 10.0))
        central.shift_theta_origin(4.0)
        central.transform_coordinates('sided')
        # compare on the sided pattern's own grid (phi mod 360)
        for j, pc in enumerate(PHI_S):
            k = int(np.argmin(np.abs(central.phi_angles - np.mod(pc, 360.0))))
            np.testing.assert_allclose(sided.data.e_theta.values[:, :, j],
                                       central.data.e_theta.values[:, :, k], atol=1e-5)
            np.testing.assert_allclose(sided.data.e_phi.values[:, :, j],
                                       central.data.e_phi.values[:, :, k], atol=1e-5)

    def test_peak_moves_to_opposite_cut(self):
        """A beam at boresight shifted by +5 deg peaks at theta = 5 on the phi + 180 cuts."""
        th, ph = TH_S, np.arange(0, 360, 10.0)
        beam = np.exp(-(np.radians(th)[:, None] / np.radians(12.0)) ** 2) * np.ones((1, len(ph)))
        p = FarFieldSpherical(th, ph, FREQS[:1], beam[None], np.zeros((1, len(th), len(ph))),
                              polarization='x')
        p.shift_theta_origin(5.0)
        mag = np.abs(p.data.e_co.values[0])
        for j, pc in enumerate(ph):
            peak_theta = th[np.argmax(mag[:, j])]
            assert peak_theta == (5.0 if pc >= 180 else 0.0), (pc, peak_theta)

    def test_no_data_lost_past_boresight(self):
        """The old failure mode: in sided format the data pushed past theta = 0 vanished."""
        p = make(TH_S, PHI_S)
        p.shift_theta_origin(6.0)
        e_theta, _ = analytic_fields(TH_C + 6.0, np.arange(0, 180, 10.0))
        # central row for sided theta = 0 at phi + 180 is theta = -0 -> old(6)
        # Compare the phi = 180 sided cut, theta 0..20, with the central phi = 0 cut at -theta
        j = int(np.where(PHI_S == 180)[0][0])
        got = p.data.e_theta.values[:, :21, j]
        want = -e_theta[:, 180:159:-1, 0]
        np.testing.assert_allclose(got, want, atol=2e-4)

    def test_theta_not_starting_at_zero_warns(self, caplog):
        p = make(np.arange(10, 181, 1.0), np.arange(0, 360, 10.0))
        with caplog.at_level(logging.WARNING):
            p.shift_theta_origin(2.0)
        assert 'end-value extension' in caplog.text

    def test_metadata_and_polarization(self):
        p = make(TH_S, PHI_S)
        p.shift_theta_origin(1.0)
        assert p.metadata['operations'][-1] == {'type': 'shift_theta_origin', 'theta_offset': 1.0}
        assert p.polarization == 'x'
        assert np.all(np.isfinite(p.data.e_co.values))
