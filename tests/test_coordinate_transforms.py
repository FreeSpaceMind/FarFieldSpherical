"""
Tests for transform_coordinates and mirror_pattern using an analytic vector field.

The field of an electrically short dipole with complex moment d,

    E(r_hat) = d - (d . r_hat) r_hat

is smooth over the whole sphere, so it can be sampled on any (theta, phi)
layout, including negative theta, using the analytic theta_hat / phi_hat
basis. A correct format transform must reproduce the same analytic field on
the output grid exactly (to float32 precision), which checks cut pairing, the
sign flip across boresight, and the shared boresight sample all at once.
"""
import logging

import numpy as np
import pytest

from farfield_spherical import FarFieldSpherical
from farfield_spherical.analysis import detect_coordinate_format

FREQS = np.array([8e9, 10e9])
D = np.array([[1.0 + 0.3j, 0.5 - 0.2j, 0.25 + 0.1j],
              [0.4 - 0.1j, 1.0 + 0.2j, -0.3 + 0.5j]])   # one moment per frequency


def basis(theta_deg, phi_deg):
    th = np.radians(np.asarray(theta_deg, dtype=float))[:, None]
    ph = np.radians(np.asarray(phi_deg, dtype=float))[None, :]
    ones = np.ones_like(th * ph)
    r_hat = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th) * ones], -1)
    th_hat = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th) * ones], -1)
    ph_hat = np.stack([-np.sin(ph) * ones, np.cos(ph) * ones, np.zeros_like(ones)], -1)
    return r_hat, th_hat, ph_hat


def analytic_fields(theta_deg, phi_deg, moments=D):
    """(e_theta, e_phi) shaped (freq, theta, phi) for the dipole field."""
    r_hat, th_hat, ph_hat = basis(theta_deg, phi_deg)
    e_theta, e_phi = [], []
    for d in moments:
        e_cart = d[None, None, :] - np.einsum('tpc,c->tp', r_hat, d)[..., None] * r_hat
        e_theta.append(np.einsum('tpc,tpc->tp', e_cart, th_hat))
        e_phi.append(np.einsum('tpc,tpc->tp', e_cart, ph_hat))
    return np.stack(e_theta), np.stack(e_phi)


def make(theta_deg, phi_deg):
    e_theta, e_phi = analytic_fields(theta_deg, phi_deg)
    return FarFieldSpherical(np.asarray(theta_deg, float), np.asarray(phi_deg, float),
                             FREQS, e_theta, e_phi, polarization='x')


def assert_matches_analytic(pattern, atol=2e-6):
    e_theta, e_phi = analytic_fields(pattern.theta_angles, pattern.phi_angles)
    np.testing.assert_allclose(pattern.data.e_theta.values, e_theta, atol=atol)
    np.testing.assert_allclose(pattern.data.e_phi.values, e_phi, atol=atol)


def assert_grid(pattern, theta, phi):
    np.testing.assert_allclose(pattern.theta_angles, theta, atol=1e-9)
    np.testing.assert_allclose(pattern.phi_angles, phi, atol=1e-9)


TH_SIDED = np.arange(0, 181, 2.0)
TH_CENTRAL = np.arange(-180, 181, 2.0)
PHI_SIDED = np.arange(0, 360, 15.0)
PHI_CENTRAL = np.arange(0, 180, 15.0)


class TestSidedInputs:
    @pytest.mark.parametrize('phi', [
        np.arange(-180, 181, 15.0),     # HFSS .ffd style, duplicated endpoint
        np.arange(0, 361, 15.0),        # duplicated endpoint at 360
        np.arange(0, 360, 15.0),        # clean
        np.arange(-180, 180, 15.0),     # clean, negative start
    ], ids=['pm180dup', '0-360dup', '0-345', 'pm180'])
    def test_to_central(self, phi):
        p = make(TH_SIDED, phi)
        p.transform_coordinates('central')
        assert_grid(p, TH_CENTRAL, PHI_CENTRAL)
        assert_matches_analytic(p)
        assert detect_coordinate_format(p) == 'central'

    @pytest.mark.parametrize('phi', [np.arange(-180, 181, 15.0), np.arange(0, 361, 15.0)],
                             ids=['pm180dup', '0-360dup'])
    def test_to_sided_normalises_phi(self, phi):
        """Already sided: phi is wrapped to [0, 360), sorted and deduplicated."""
        p = make(TH_SIDED, phi)
        p.transform_coordinates('sided')
        assert_grid(p, TH_SIDED, PHI_SIDED)
        assert_matches_analytic(p)

    def test_round_trip(self):
        p = make(TH_SIDED, np.arange(-180, 181, 15.0))
        p.transform_coordinates('central')
        p.transform_coordinates('sided')
        assert_grid(p, TH_SIDED, PHI_SIDED)
        assert_matches_analytic(p)

    def test_phi_step_not_dividing_180(self, caplog):
        """phi = 0, 50, ..., 350: no cut has a partner at phi + 180.
        Every cut still ends up in the output, on its own central phi."""
        phi = np.arange(0, 360, 50.0)
        p = make(TH_SIDED, phi)
        with caplog.at_level(logging.WARNING):
            p.transform_coordinates('central')
        assert 'no source data' in caplog.text
        expected_phi = np.sort(np.concatenate([phi[phi < 180], phi[phi >= 180] - 180]))
        assert_grid(p, TH_CENTRAL, expected_phi)
        # Wherever data was placed it is the analytic field; elsewhere it is zero.
        e_theta, e_phi = analytic_fields(p.theta_angles, p.phi_angles)
        got = p.data.e_theta.values
        filled = np.abs(got) > 0
        np.testing.assert_allclose(got[filled], e_theta[filled], atol=2e-6)
        # Direct cuts have the theta >= 0 half, folded cuts the theta <= 0 half
        n_pos = len(TH_SIDED)
        for j, pc in enumerate(p.phi_angles):
            if pc in phi:
                assert filled[0, n_pos - 1:, j].all() and not filled[0, :n_pos - 1, j].any()
            else:
                assert filled[0, :n_pos, j].all() and not filled[0, n_pos:, j].any()

    def test_hemisphere_to_central_and_back(self):
        p = make(np.arange(0, 91, 2.0), PHI_SIDED)
        p.transform_coordinates('central')
        assert_grid(p, np.arange(-90, 91, 2.0), PHI_CENTRAL)
        assert_matches_analytic(p)
        p.transform_coordinates('sided')
        assert_grid(p, np.arange(0, 91, 2.0), PHI_SIDED)
        assert_matches_analytic(p)

    def test_theta_not_starting_at_zero_raises(self):
        p = make(np.arange(10, 181, 2.0), PHI_SIDED)
        with pytest.raises(ValueError, match='start at 0'):
            p.transform_coordinates('central')

    def test_half_plane_phi(self, caplog):
        """phi 0..165 only: negative theta half is unknown and zero-filled."""
        p = make(TH_SIDED, PHI_CENTRAL)
        with caplog.at_level(logging.WARNING):
            p.transform_coordinates('central')
        assert 'no source data' in caplog.text
        assert_grid(p, TH_CENTRAL, PHI_CENTRAL)
        n_pos = len(TH_SIDED)
        e_theta, _ = analytic_fields(p.theta_angles, p.phi_angles)
        np.testing.assert_allclose(p.data.e_theta.values[:, n_pos - 1:, :],
                                   e_theta[:, n_pos - 1:, :], atol=2e-6)
        assert np.all(p.data.e_theta.values[:, :n_pos - 1, :] == 0)

    def test_bad_format_raises(self):
        with pytest.raises(ValueError):
            make(TH_SIDED, PHI_SIDED).transform_coordinates('polar')


class TestCentralInputs:
    @pytest.mark.parametrize('phi', [
        np.arange(0, 180, 15.0),        # canonical
        np.arange(0, 181, 15.0),        # 180 included (redundant with phi=0, theta<0)
        np.arange(-90, 90, 15.0),       # measurement style, negative phi
        np.arange(0, 360, 15.0),        # full phi with +/- theta (every direction twice)
    ], ids=['0-165', '0-180', 'pm90', '0-345'])
    def test_to_sided(self, phi):
        p = make(TH_CENTRAL, phi)
        p.transform_coordinates('sided')
        assert_grid(p, TH_SIDED, PHI_SIDED)
        assert_matches_analytic(p)
        assert detect_coordinate_format(p) == 'sided'

    @pytest.mark.parametrize('phi', [
        np.arange(0, 181, 15.0),
        np.arange(-90, 90, 15.0),
        np.arange(0, 360, 15.0),
    ], ids=['0-180', 'pm90', '0-345'])
    def test_to_central_canonicalises(self, phi):
        """Already central: cuts outside 0 <= phi < 180 are folded back in."""
        p = make(TH_CENTRAL, phi)
        p.transform_coordinates('central')
        assert_grid(p, TH_CENTRAL, PHI_CENTRAL)
        assert_matches_analytic(p)

    def test_canonical_central_is_unchanged(self):
        p = make(TH_CENTRAL, PHI_CENTRAL)
        before = p.data.e_theta.values.copy()
        p.transform_coordinates('central')
        assert_grid(p, TH_CENTRAL, PHI_CENTRAL)
        np.testing.assert_array_equal(p.data.e_theta.values, before)

    def test_round_trip(self):
        p = make(TH_CENTRAL, PHI_CENTRAL)
        p.transform_coordinates('sided')
        p.transform_coordinates('central')
        assert_grid(p, TH_CENTRAL, PHI_CENTRAL)
        assert_matches_analytic(p)

    def test_boresight_row_is_continuous(self):
        """The theta = 0 sample of the phi + 180 cuts must carry the sign flip.
        In Ludwig-3 co-pol this shows up as phase continuity through boresight."""
        p = make(TH_CENTRAL, PHI_CENTRAL)
        p.transform_coordinates('sided')
        e_co = p.data.e_co.values                      # (freq, theta, phi)
        # For each cut, the phase step from theta = 0 to theta = 2 deg is small
        step = np.angle(e_co[:, 1, :] * np.conj(e_co[:, 0, :]))
        assert np.all(np.abs(step) < np.radians(10))

    def test_asymmetric_theta_range(self, caplog):
        """theta -90..180: the phi + 180 half only exists up to theta = 90."""
        p = make(np.arange(-90, 181, 2.0), PHI_CENTRAL)
        with caplog.at_level(logging.WARNING):
            p.transform_coordinates('sided')
        assert 'no source data' in caplog.text
        assert_grid(p, TH_SIDED, PHI_SIDED)
        e_theta, _ = analytic_fields(p.theta_angles, p.phi_angles)
        got = p.data.e_theta.values
        beyond = p.theta_angles > 90
        # Direct cuts (phi < 180) are complete; mirrored cuts stop at 90 deg
        np.testing.assert_allclose(got[:, :, :12], e_theta[:, :, :12], atol=2e-6)
        np.testing.assert_allclose(got[:, ~beyond, 12:], e_theta[:, ~beyond, 12:], atol=2e-6)
        assert np.all(got[:, beyond, 12:] == 0)

    def test_half_step_theta_grid_without_zero(self):
        """theta = -179..179 step 2 has no theta = 0 sample."""
        th = np.arange(-179, 180, 2.0)
        p = make(th, PHI_CENTRAL)
        p.transform_coordinates('sided')
        assert_grid(p, np.arange(1, 180, 2.0), PHI_SIDED)
        assert_matches_analytic(p)


class TestBookkeeping:
    def test_polarization_recomputed(self):
        p = make(TH_CENTRAL, PHI_CENTRAL)
        p.transform_coordinates('sided')
        from farfield_spherical.polarization import polarization_tp2xy
        e_x, _ = polarization_tp2xy(p.phi_angles, p.data.e_theta.values, p.data.e_phi.values)
        np.testing.assert_allclose(p.data.e_co.values, e_x, atol=1e-6)

    def test_metadata_records_ranges(self):
        p = make(TH_SIDED, np.arange(-180, 181, 15.0))
        p.transform_coordinates('central')
        op = p.metadata['operations'][-1]
        assert op['type'] == 'transform_coordinates'
        assert op['old_phi_range'] == [-180.0, 180.0]
        assert op['new_theta_range'] == [-180.0, 180.0]
        assert op['new_phi_range'] == [0.0, 165.0]

    def test_source_pattern_untouched_by_copy(self):
        p = make(TH_SIDED, np.arange(-180, 181, 15.0))
        q = p.copy()
        q.transform_coordinates('central')
        assert_grid(p, TH_SIDED, np.arange(-180, 181, 15.0))


class TestMirrorPattern:
    def test_mirror_fills_negative_theta(self):
        p = make(TH_CENTRAL, PHI_CENTRAL)
        e_theta_ref = p.data.e_theta.values.copy()
        e_phi_ref = p.data.e_phi.values.copy()
        neg = p.theta_angles < 0
        p.data['e_theta'].values[:, neg, :] = 0
        p.data['e_phi'].values[:, neg, :] = 0
        p.mirror_pattern()
        pos = p.theta_angles > 0
        # Documented behaviour: E_theta negated, E_phi copied, mirrored about theta = 0
        np.testing.assert_allclose(p.data.e_theta.values[:, neg, :],
                                   -e_theta_ref[:, pos, :][:, ::-1, :], atol=1e-6)
        np.testing.assert_allclose(p.data.e_phi.values[:, neg, :],
                                   e_phi_ref[:, pos, :][:, ::-1, :], atol=1e-6)
        np.testing.assert_allclose(p.data.e_theta.values[:, pos, :], e_theta_ref[:, pos, :])

    def test_mirror_requires_central(self):
        with pytest.raises(ValueError, match='central'):
            make(TH_SIDED, PHI_SIDED).mirror_pattern()

    def test_mirror_requires_zero(self):
        with pytest.raises(ValueError, match='theta=0'):
            make(np.arange(-179, 180, 2.0), PHI_CENTRAL).mirror_pattern()
