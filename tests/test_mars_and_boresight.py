"""
Tests for apply_mars and normalize_at_boresight.

apply_mars is a cylindrical-mode (Fourier in theta) low-pass on each closed
phi cut. A smooth analytic field whose modes lie below the cutoff must pass
through unchanged, and a high-order ripple added to it must be removed.

normalize_at_boresight removes a per-cut complex gain. Applying a known set
of cut-to-cut errors and normalizing must recover the original pattern up to
one global complex constant.
"""
import logging

import numpy as np
import pytest

from farfield_spherical import FarFieldSpherical
from farfield_spherical.polarization import polarization_xy2tp
from farfield_spherical.utilities import lightspeed
from .test_coordinate_transforms import analytic_fields, make, assert_grid, FREQS

TH_C = np.arange(-180, 181, 2.0)          # duplicated +/-180 endpoint
TH_C_OPEN = np.arange(-180, 180, 2.0)     # no duplicate
TH_S = np.arange(0, 181, 2.0)
PHI_C = np.arange(0, 180, 15.0)
PHI_S = np.arange(-180, 181, 15.0)


def extent_for_order(n_max, freq=FREQS[0]):
    """Radial extent giving floor(k D) == n_max at `freq`."""
    k = 2 * np.pi * freq / lightspeed
    return (n_max + 0.5) / k


def with_ripple(pattern, order=30, amplitude=0.1):
    """Add an azimuthally uniform cos(order * theta) ripple to e_theta."""
    p = pattern.copy()
    th = np.radians(p.theta_angles)[None, :, None]
    p.data['e_theta'].values = (p.data.e_theta.values + amplitude * np.cos(order * th)).astype(np.complex64)
    p.assign_polarization(p.polarization)
    return p


class TestApplyMars:
    def test_smooth_field_passes_through(self):
        """The dipole field has modes of order <= 2, far below the cutoff."""
        p = make(TH_C, PHI_C)
        before_theta = p.data.e_theta.values.copy()
        before_phi = p.data.e_phi.values.copy()
        p.apply_mars(extent_for_order(10))
        np.testing.assert_allclose(p.data.e_theta.values, before_theta, atol=2e-6)
        np.testing.assert_allclose(p.data.e_phi.values, before_phi, atol=2e-6)

    def test_open_grid_passes_through(self):
        """-180..178 with no duplicated endpoint is a complete circle too."""
        p = make(TH_C_OPEN, PHI_C)
        before = p.data.e_theta.values.copy()
        p.apply_mars(extent_for_order(10))
        np.testing.assert_allclose(p.data.e_theta.values, before, atol=2e-6)

    def test_ripple_is_removed(self):
        """A cos(30 theta) ripple lies above a cutoff of 10 and must vanish."""
        clean = make(TH_C, PHI_C)
        p = with_ripple(clean, order=30, amplitude=0.1)
        assert np.abs(p.data.e_theta.values - clean.data.e_theta.values).max() > 0.09
        p.apply_mars(extent_for_order(10))
        np.testing.assert_allclose(p.data.e_theta.values, clean.data.e_theta.values, atol=2e-5)
        np.testing.assert_allclose(p.data.e_phi.values, clean.data.e_phi.values, atol=2e-5)

    def test_ripple_below_cutoff_is_kept(self):
        clean = make(TH_C, PHI_C)
        p = with_ripple(clean, order=30, amplitude=0.1)
        before = p.data.e_theta.values.copy()
        p.apply_mars(extent_for_order(40))
        np.testing.assert_allclose(p.data.e_theta.values, before, atol=2e-5)

    def test_amplitude_is_preserved(self):
        """The old normalization halved the peak on the wrong span."""
        p = make(TH_C, PHI_C)
        peak_before = np.abs(p.data.e_theta.values).max()
        p.apply_mars(extent_for_order(10))
        assert np.abs(p.data.e_theta.values).max() == pytest.approx(peak_before, rel=1e-5)

    def test_sided_input_matches_central(self):
        """Sided .ffd-style input is filtered on a central view and mapped back."""
        central = with_ripple(make(TH_C, PHI_C), order=30, amplitude=0.1)
        sided = central.copy()
        sided.transform_coordinates('sided')
        sided_grid_theta, sided_grid_phi = sided.theta_angles.copy(), sided.phi_angles.copy()

        central.apply_mars(extent_for_order(10))
        sided.apply_mars(extent_for_order(10))
        assert_grid(sided, sided_grid_theta, sided_grid_phi)

        central.transform_coordinates('sided')
        np.testing.assert_allclose(sided.data.e_theta.values, central.data.e_theta.values, atol=2e-5)
        np.testing.assert_allclose(sided.data.e_phi.values, central.data.e_phi.values, atol=2e-5)

    def test_hemisphere_raises(self):
        """theta -90..90 is not a closed circle."""
        p = make(np.arange(-90, 91, 2.0), PHI_C)
        with pytest.raises(ValueError, match='360'):
            p.apply_mars(extent_for_order(10))

    def test_sided_not_starting_at_zero_raises(self):
        p = make(np.arange(10, 181, 2.0), np.arange(0, 360, 15.0))
        with pytest.raises(ValueError, match='closed circle'):
            p.apply_mars(extent_for_order(10))

    def test_cutoff_above_sampling_warns_and_is_identity(self, caplog):
        p = make(TH_C, PHI_C)
        before = p.data.e_theta.values.copy()
        with caplog.at_level(logging.WARNING, logger='farfield_spherical.farfield_operations'):
            p.apply_mars(extent_for_order(500))
        assert 'removes nothing' in caplog.text
        np.testing.assert_allclose(p.data.e_theta.values, before, atol=2e-6)

    def test_negative_extent_raises(self):
        with pytest.raises(ValueError, match='positive'):
            make(TH_C, PHI_C).apply_mars(-1.0)


def linear_pattern(theta, phi, cross_level=1e-3, phase_deg=0.0):
    """x-polarized beam with a weak, noisy cross-pol and a chosen boresight phase."""
    rng = np.random.default_rng(0)
    th = np.radians(theta)[:, None]
    e_co = np.exp(-(th / np.radians(30.0)) ** 2) * np.ones((1, len(phi))) * np.exp(1j * np.radians(phase_deg))
    e_cx = cross_level * (rng.standard_normal((len(theta), len(phi)))
                          + 1j * rng.standard_normal((len(theta), len(phi))))
    e_co = np.stack([e_co] * len(FREQS))      # (freq, theta, phi)
    e_cx = np.stack([e_cx] * len(FREQS))
    e_theta, e_phi = polarization_xy2tp(phi, e_co, e_cx)
    return FarFieldSpherical(theta, phi, FREQS, e_theta, e_phi, polarization='x')


def apply_cut_errors(pattern, gains):
    """Multiply every sample of cut j by the complex gain gains[j]."""
    p = pattern.copy()
    p.data['e_theta'].values = (p.data.e_theta.values * gains[None, None, :]).astype(np.complex64)
    p.data['e_phi'].values = (p.data.e_phi.values * gains[None, None, :]).astype(np.complex64)
    p.assign_polarization(p.polarization)
    return p


class TestNormalizeAtBoresight:
    def test_cut_errors_are_removed(self):
        """Per-cut complex gains are undone, up to one global constant."""
        clean = linear_pattern(TH_S, np.arange(0, 360, 15.0))
        rng = np.random.default_rng(1)
        gains = (1 + 0.2 * rng.standard_normal(24)) * np.exp(1j * np.radians(10 * rng.standard_normal(24)))
        p = apply_cut_errors(clean, gains)
        p.normalize_at_boresight()

        co = p.data.e_co.values[0]
        boresight = co[0, :]
        # every cut now shares the boresight value
        np.testing.assert_allclose(boresight, boresight[0], rtol=1e-5, atol=1e-6)
        # and the pattern shape is restored up to that shared constant
        scale = boresight[0] / clean.data.e_co.values[0, 0, 0]
        np.testing.assert_allclose(co, clean.data.e_co.values[0] * scale, atol=1e-5)

    def test_phases_straddling_180_degrees(self):
        """A linear median of wrapped angles put the reference near 0 and
        rotated the whole pattern by 180 degrees."""
        clean = linear_pattern(TH_S, np.arange(0, 360, 15.0), phase_deg=179.0)
        rng = np.random.default_rng(2)
        gains = np.exp(1j * np.radians(3 * rng.standard_normal(24)))   # +/- a few degrees
        p = apply_cut_errors(clean, gains)
        p.normalize_at_boresight()
        phase = np.degrees(np.angle(p.data.e_co.values[0, 0, 0]))
        assert abs((phase - 179.0 + 180) % 360 - 180) < 2.0, phase

    def test_weak_cross_pol_is_not_amplified(self):
        """Correcting the cross-pol by its own noisy boresight value applied an
        arbitrary large gain to the whole cut."""
        clean = linear_pattern(TH_S, np.arange(0, 360, 15.0), cross_level=1e-3)
        rng = np.random.default_rng(3)
        gains = (1 + 0.1 * rng.standard_normal(24)) * np.exp(1j * np.radians(5 * rng.standard_normal(24)))
        p = apply_cut_errors(clean, gains)
        cx_before = np.abs(p.data.e_cx.values).max()
        p.normalize_at_boresight()
        cx_after = np.abs(p.data.e_cx.values).max()
        assert cx_after < 3 * cx_before, (cx_before, cx_after)

    def test_dual_pol_components_corrected_independently(self):
        """When both components are significant each keeps its own correction."""
        theta, phi = TH_S, np.arange(0, 360, 15.0)
        th = np.radians(theta)[:, None]
        beam = np.exp(-(th / np.radians(30.0)) ** 2) * np.ones((1, len(phi)))
        e_co = np.stack([beam] * len(FREQS))
        e_cx = np.stack([0.7 * beam * np.exp(1j * 0.4)] * len(FREQS))
        e_theta, e_phi = polarization_xy2tp(phi, e_co, e_cx)
        clean = FarFieldSpherical(theta, phi, FREQS, e_theta, e_phi, polarization='x')
        rng = np.random.default_rng(4)
        gains = (1 + 0.2 * rng.standard_normal(24)) * np.exp(1j * np.radians(10 * rng.standard_normal(24)))
        p = apply_cut_errors(clean, gains)
        p.normalize_at_boresight()
        for name in ('e_co', 'e_cx'):
            boresight = p.data[name].values[0, 0, :]
            np.testing.assert_allclose(boresight, boresight[0], rtol=1e-5, atol=1e-6)

    def test_central_format(self):
        clean = linear_pattern(TH_C, PHI_C)
        rng = np.random.default_rng(5)
        gains = (1 + 0.2 * rng.standard_normal(12)) * np.exp(1j * np.radians(10 * rng.standard_normal(12)))
        p = apply_cut_errors(clean, gains)
        p.normalize_at_boresight()
        i0 = int(np.argmin(np.abs(p.theta_angles)))
        boresight = p.data.e_co.values[0, i0, :]
        np.testing.assert_allclose(boresight, boresight[0], rtol=1e-5, atol=1e-6)

    def test_records_operation(self):
        p = linear_pattern(TH_S, np.arange(0, 360, 15.0))
        p.normalize_at_boresight()
        assert p.metadata['operations'][-1] == {'type': 'normalize_at_boresight'}
