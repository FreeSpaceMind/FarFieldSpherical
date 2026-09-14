"""
Tests for feed cross-polarization metrics (farfield_spherical.crosspol).

Two groups:

* Synthetic patterns with closed-form co/cross fields, so every metric can be
  checked against an analytic value and the error paths can be exercised
  without data files.
* Regression against the Guppy horn HFSS export (X5_horn_1.ffd). These are
  skipped when the file is not present in tests/data/.
"""
import logging

import numpy as np
import pytest

from farfield_spherical import (
    FarFieldSpherical,
    read_ffd,
    integrated_xpd,
    azimuthal_modes,
    n0_crosspol_level,
    point_xpd,
    edge_taper,
    crosspol_report,
    check_requirements,
)
from farfield_spherical.polarization import polarization_xy2tp
from .conftest import FFD_FILE, requires_ffd


# ---------------------------------------------------------------------------
# Synthetic pattern helpers
# ---------------------------------------------------------------------------

THETA_E = 35.0
CO_WIDTH_DEG = 40.0     # co-pol Gaussian width
CX_AMP = 1e-2           # cos(2 phi) cross-pol amplitude
N0_AMP = 3e-4           # azimuthally symmetric cross-pol amplitude
FREQS = np.array([8e9, 12e9])
# Per-frequency scale factors for (co, cos2phi cross, n=0 cross)
SCALES = [(1.0, 1.0, 1.0), (0.9, 2.0, 0.5)]


def _fields(theta_deg, phi_deg):
    """Closed-form co/cross fields on a (theta, phi) grid, shaped (freq, theta, phi)."""
    TH, PH = np.meshgrid(np.deg2rad(theta_deg), np.deg2rad(phi_deg), indexing='ij')
    co_env = np.exp(-(TH / np.deg2rad(CO_WIDTH_DEG)) ** 2)
    cx_env = CX_AMP * np.sin(TH) ** 2
    n0_env = N0_AMP * np.cos(TH)
    e_co = np.stack([sa * co_env for sa, _, _ in SCALES])
    e_cx = np.stack([sb * cx_env * np.cos(2 * PH) + sc * n0_env for _, sb, sc in SCALES])
    return e_co, e_cx


def make_pattern(theta_deg, phi_deg, polarization='x'):
    """Build a FarFieldSpherical whose Ludwig-3 x co/cross fields are the closed forms."""
    e_co, e_cx = _fields(theta_deg, phi_deg)
    e_th, e_ph = polarization_xy2tp(phi_deg, e_co, e_cx)
    return FarFieldSpherical(theta_deg, phi_deg, FREQS, e_th, e_ph, polarization=polarization)


@pytest.fixture
def sided_pattern():
    """Sided format, phi -180..180 with duplicated endpoint, like an HFSS .ffd."""
    return make_pattern(np.arange(0, 181, 1.0), np.arange(-180, 181, 15.0))


def analytic_metrics(theta_e=THETA_E, dtheta=1.0, dphi=15.0):
    """Reference values using the same rectangular quadrature on a deduplicated grid."""
    thc = np.deg2rad(np.arange(0, theta_e + 1e-9, dtheta))
    phc = np.deg2rad(np.arange(0, 360, dphi))
    e_co, e_cx = _fields(np.rad2deg(thc), np.rad2deg(phc))
    w = (np.sin(thc) * np.deg2rad(dtheta) * np.deg2rad(dphi))[None, :, None]
    p_co = (np.abs(e_co) ** 2 * w).sum(axis=(1, 2))
    p_cx = (np.abs(e_cx) ** 2 * w).sum(axis=(1, 2))
    co_pk = np.array([sa for sa, _, _ in SCALES])
    n0_pk = np.array([sc * N0_AMP for _, _, sc in SCALES])
    return {
        'xpd_int_db': 10 * np.log10(p_co / p_cx),
        'n0_level_db': 20 * np.log10(co_pk / n0_pk),
        'xpd_worst_db': (20 * np.log10(np.abs(e_co) / np.abs(e_cx))).min(axis=(1, 2)),
        'xpol_peak_db': 20 * np.log10(co_pk / np.abs(e_cx).max(axis=(1, 2))),
        'edge_taper_db': np.full(len(SCALES),
                                 20 * np.log10(np.exp(-(np.deg2rad(theta_e) / np.deg2rad(CO_WIDTH_DEG)) ** 2))),
    }


# ---------------------------------------------------------------------------
# Synthetic: values
# ---------------------------------------------------------------------------

class TestSyntheticValues:
    def test_report_matches_analytic(self, sided_pattern):
        report = crosspol_report(sided_pattern, THETA_E)
        ref = analytic_metrics()
        for name, expected in ref.items():
            np.testing.assert_allclose(report[name].values, expected, atol=1e-3,
                                       err_msg=name)

    def test_individual_functions_agree_with_report(self, sided_pattern):
        report = crosspol_report(sided_pattern, THETA_E)
        np.testing.assert_allclose(integrated_xpd(sided_pattern, THETA_E).values,
                                   report['xpd_int_db'].values)
        np.testing.assert_allclose(n0_crosspol_level(sided_pattern, THETA_E).values,
                                   report['n0_level_db'].values)
        pt = point_xpd(sided_pattern, THETA_E)
        np.testing.assert_allclose(pt['xpd_worst_db'].values, report['xpd_worst_db'].values)
        np.testing.assert_allclose(pt['xpol_peak_db'].values, report['xpol_peak_db'].values)
        np.testing.assert_allclose(edge_taper(sided_pattern, THETA_E).values,
                                   report['edge_taper_db'].values)

    def test_method_wrapper(self, sided_pattern):
        a = sided_pattern.crosspol_report(THETA_E, n_max=4)
        b = crosspol_report(sided_pattern, THETA_E, n_max=4)
        assert a.equals(b)
        assert a.attrs['n_max'] == 4

    def test_mode_spectrum_is_pure_cos2phi_plus_dc(self, sided_pattern):
        modes = azimuthal_modes(sided_pattern, THETA_E, n_max=6)
        rel = modes['mode_power_rel_db'].values
        # n = 2 carries essentially all the power (n = 0 term is 30 dB down)
        assert np.all(rel[:, 2] > -0.2)
        assert np.all(rel[:, 2] <= 1e-9)
        # everything except n = 0 and n = 2 is at the numerical floor
        for n in (1, 3, 4, 5, 6):
            assert np.all(rel[:, n] < -80), f"n={n}: {rel[:, n]}"
        # c_0 reproduces the n = 0 envelope
        c0 = np.abs(modes['c_n'].sel(n_signed=0).values)
        th = np.deg2rad(modes['theta'].values)
        for i, (_, _, sc) in enumerate(SCALES):
            np.testing.assert_allclose(c0[i], sc * N0_AMP * np.cos(th), rtol=1e-3)

    def test_mode_power_sums_to_total_cone_power(self, sided_pattern):
        """With n_max at Nyquist, mode powers sum to the cross-pol cone power (Parseval)."""
        modes = azimuthal_modes(sided_pattern, THETA_E, n_max=12)
        total_db = 10 * np.log10(modes['mode_power'].values.sum(axis=1))
        rel_sum_db = 10 * np.log10((10 ** (modes['mode_power_rel_db'].values / 10)).sum(axis=1))
        np.testing.assert_allclose(rel_sum_db, 0.0, atol=1e-9)
        assert np.all(np.isfinite(total_db))

    def test_co_component_decomposition(self, sided_pattern):
        modes = azimuthal_modes(sided_pattern, THETA_E, n_max=3, component='e_co')
        rel = modes['mode_power_rel_db'].values
        np.testing.assert_allclose(rel[:, 0], 0.0, atol=1e-9)
        assert np.all(rel[:, 1:] < -80)

    def test_pattern_not_mutated(self, sided_pattern):
        before_theta = sided_pattern.theta_angles.copy()
        before_phi = sided_pattern.phi_angles.copy()
        before_cx = sided_pattern.data.e_cx.values.copy()
        crosspol_report(sided_pattern, THETA_E)
        np.testing.assert_array_equal(sided_pattern.theta_angles, before_theta)
        np.testing.assert_array_equal(sided_pattern.phi_angles, before_phi)
        np.testing.assert_array_equal(sided_pattern.data.e_cx.values, before_cx)

    def test_phi_0_to_360_duplicate_endpoint(self, sided_pattern):
        """0..360 with duplicate endpoint gives the same result as -180..180."""
        alt = make_pattern(np.arange(0, 181, 1.0), np.arange(0, 361, 15.0))
        np.testing.assert_allclose(integrated_xpd(alt, THETA_E).values,
                                   integrated_xpd(sided_pattern, THETA_E).values, atol=1e-6)

    def test_phi_without_endpoint(self, sided_pattern):
        """0..345 (no duplicate) gives the same result."""
        alt = make_pattern(np.arange(0, 181, 1.0), np.arange(0, 360, 15.0))
        np.testing.assert_allclose(integrated_xpd(alt, THETA_E).values,
                                   integrated_xpd(sided_pattern, THETA_E).values, atol=1e-6)

    def test_central_format_matches_sided(self, sided_pattern):
        """Exercises transform_coordinates inside the cone preparation."""
        # Build the central pattern from the 0..360 layout: see the xfail below.
        src = make_pattern(np.arange(0, 181, 1.0), np.arange(0, 361, 15.0))
        central = src.copy()
        central.transform_coordinates('central')
        assert central.theta_angles.min() < 0
        assert central.phi_angles.max() < 180
        np.testing.assert_allclose(integrated_xpd(central, THETA_E).values,
                                   integrated_xpd(sided_pattern, THETA_E).values, atol=1e-6)
        # complex64 storage: the n = 0 term is 70 dB down, so the round trip
        # through the coordinate transform costs a few thousandths of a dB.
        np.testing.assert_allclose(n0_crosspol_level(central, THETA_E).values,
                                   n0_crosspol_level(sided_pattern, THETA_E).values, atol=1e-2)

    @pytest.mark.xfail(
        strict=True,
        reason="transform_coordinates('central') mis-aligns the phi >= 180 block when the "
               "input phi grid is -180..180 with a duplicated endpoint (both map to 180). "
               "Pre-existing in farfield_operations; not a crosspol issue.")
    def test_central_from_pm180_layout_matches_sided(self, sided_pattern):
        central = sided_pattern.copy()
        central.transform_coordinates('central')
        np.testing.assert_allclose(integrated_xpd(central, THETA_E).values,
                                   integrated_xpd(sided_pattern, THETA_E).values, atol=1e-6)

    def test_central_format_with_phi_180_inclusive(self, sided_pattern):
        """Central pattern with phi 0..180 inclusive produces an interior duplicate
        phi = 180 after transform; it must be deduplicated, not rejected."""
        theta_c = np.arange(-180, 181, 1.0)
        phi_c = np.arange(0, 181, 15.0)
        # Build directly in central format from the sided closed forms
        e_co_pos, e_cx_pos = _fields(np.abs(theta_c), phi_c)
        e_co_neg, e_cx_neg = _fields(np.abs(theta_c), phi_c + 180.0)
        neg = theta_c < 0
        e_co = np.where(neg[None, :, None], e_co_neg, e_co_pos)
        e_cx = np.where(neg[None, :, None], e_cx_neg, e_cx_pos)
        # Ludwig-3 x at (theta<0, phi) equals the field at (|theta|, phi+180);
        # spherical components flip sign across boresight.
        e_th, e_ph = polarization_xy2tp(phi_c, e_co, e_cx)
        e_th = np.where(neg[None, :, None], -e_th, e_th)
        e_ph = np.where(neg[None, :, None], -e_ph, e_ph)
        central = FarFieldSpherical(theta_c, phi_c, FREQS, e_th, e_ph, polarization='x')
        np.testing.assert_allclose(integrated_xpd(central, THETA_E).values,
                                   integrated_xpd(sided_pattern, THETA_E).values, atol=1e-3)

    def test_theta_e_between_samples(self, sided_pattern):
        """theta_e not on the grid: cone uses samples <= theta_e, edge taper uses nearest."""
        report = crosspol_report(sided_pattern, 35.4)
        ref = crosspol_report(sided_pattern, 35.0)
        np.testing.assert_allclose(report['xpd_int_db'].values, ref['xpd_int_db'].values)
        assert report.attrs['theta_actual_deg'] == 35.0

    def test_report_to_dataframe(self, sided_pattern):
        df = crosspol_report(sided_pattern, THETA_E).to_dataframe()
        assert 'xpd_int_db' in df.columns
        assert len(df) == len(FREQS) * 7


# ---------------------------------------------------------------------------
# Synthetic: requirement checking
# ---------------------------------------------------------------------------

class TestCheckRequirements:
    @pytest.fixture
    def report(self, sided_pattern):
        return crosspol_report(sided_pattern, THETA_E)

    def test_pass(self, report):
        r = check_requirements(report, xpd_int_min_db=20, n0_min_db=40)
        assert r.attrs['all_pass'] is True
        assert r['in_band'].values.all()
        assert r['xpd_int_pass'].values.all()
        assert r['n0_pass'].values.all()
        np.testing.assert_allclose(r['xpd_int_margin_db'].values,
                                   r['xpd_int_db'].values - 20)
        assert r.attrs['worst_xpd_int_in_band_db'] == pytest.approx(r['xpd_int_db'].values.min())

    def test_fail_out_of_band_excluded(self, report):
        # 12 GHz has the lower XPD_int; set the limit between the two values
        lo, hi = sorted(report['xpd_int_db'].values)
        limit = 0.5 * (lo + hi)
        r = check_requirements(report, xpd_int_min_db=limit)
        assert r.attrs['all_pass'] is False
        r = check_requirements(report, xpd_int_min_db=limit, bands_hz=[(7e9, 9e9)])
        assert r.attrs['all_pass'] is True
        np.testing.assert_array_equal(r['in_band'].values, [True, False])
        assert r.attrs['worst_xpd_int_in_band_db'] == pytest.approx(hi)

    def test_only_n0_requirement(self, report):
        r = check_requirements(report, n0_min_db=100)
        assert r.attrs['all_pass'] is False
        assert 'xpd_int_pass' not in r
        assert 'n0_pass' in r

    def test_no_requirements(self, report):
        r = check_requirements(report)
        assert r.attrs['all_pass'] is True
        assert 'xpd_int_pass' not in r and 'n0_pass' not in r

    def test_nothing_in_band(self, report):
        r = check_requirements(report, xpd_int_min_db=20, bands_hz=[(1e9, 2e9)])
        assert not r['in_band'].values.any()
        assert r.attrs['all_pass'] is False
        assert 'worst_xpd_int_in_band_db' not in r.attrs

    def test_input_not_mutated(self, report):
        check_requirements(report, xpd_int_min_db=20)
        assert 'in_band' not in report
        assert 'all_pass' not in report.attrs


# ---------------------------------------------------------------------------
# Synthetic: error paths
# ---------------------------------------------------------------------------

class TestErrors:
    def test_half_plane_phi_raises(self):
        p = make_pattern(np.arange(0, 181, 1.0), np.arange(0, 181, 15.0))
        with pytest.raises(ValueError, match='360'):
            integrated_xpd(p, THETA_E)

    def test_theta_e_beyond_range_raises(self):
        p = make_pattern(np.arange(0, 61, 1.0), np.arange(0, 360, 15.0))
        with pytest.raises(ValueError, match='exceeds'):
            integrated_xpd(p, 70.0)

    def test_theta_e_too_small_raises(self, sided_pattern):
        with pytest.raises(ValueError, match='Fewer than two'):
            integrated_xpd(sided_pattern, 0.5)

    def test_nonuniform_phi_raises(self):
        phi = np.array([0, 15, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330.0])
        p = make_pattern(np.arange(0, 181, 1.0), phi)
        with pytest.raises(ValueError, match='uniform phi'):
            integrated_xpd(p, THETA_E)

    def test_nonuniform_theta_in_cone_raises(self):
        theta = np.concatenate([np.arange(0, 20, 1.0), np.arange(20, 181, 2.0)])
        p = make_pattern(theta, np.arange(0, 360, 15.0))
        with pytest.raises(ValueError, match='uniform theta'):
            integrated_xpd(p, THETA_E)
        # but fine if the cone stays inside the uniform region
        integrated_xpd(p, 15.0)

    def test_n_max_above_nyquist_raises(self, sided_pattern):
        with pytest.raises(ValueError, match='Nyquist'):
            azimuthal_modes(sided_pattern, THETA_E, n_max=13)
        azimuthal_modes(sided_pattern, THETA_E, n_max=12)

    def test_bad_component_raises(self, sided_pattern):
        with pytest.raises(ValueError, match='component'):
            azimuthal_modes(sided_pattern, THETA_E, component='e_theta')

    def test_non_ludwig3_polarization_warns_but_runs(self, caplog):
        p = make_pattern(np.arange(0, 181, 1.0), np.arange(0, 360, 15.0), polarization='rhcp')
        with caplog.at_level(logging.WARNING, logger='farfield_spherical.crosspol'):
            report = crosspol_report(p, THETA_E)
        assert any('Ludwig-3' in rec.message for rec in caplog.records)
        assert np.all(np.isfinite(report['xpd_int_db'].values))
        assert report.attrs['polarization'] == 'rhcp'


# ---------------------------------------------------------------------------
# Regression against the Guppy horn export
# ---------------------------------------------------------------------------

# Reference values at theta_e = 35 deg, from the requirements document.
GUPPY_FREQS_GHZ = np.array([8, 9, 10, 11, 12, 13, 14, 15])
GUPPY_REF = {
    'edge_taper_db': [-10.98, -12.22, -12.91, -12.29, -11.24, -13.12, -13.37, -12.43],
    'xpd_int_db':    [45.92, 33.79, 27.39, 24.67, 21.55, 28.77, 31.73, 27.97],
    'xpd_worst_db':  [35.87, 23.32, 17.93, 17.33, 15.44, 22.60, 20.79, 16.36],
    'xpol_peak_db':  [44.68, 34.50, 28.56, 25.46, 21.47, 27.24, 30.05, 25.22],
    'n0_level_db':   [77.5, 77.8, 78.2, 74.2, 65.9, 70.7, 68.0, 65.2],
}
GUPPY_TOL = {'edge_taper_db': 0.05, 'xpd_int_db': 0.05, 'xpd_worst_db': 0.05,
             'xpol_peak_db': 0.05, 'n0_level_db': 0.3}
GUPPY_XPD_INT_25 = [47.07, 36.67, 29.83, 26.38, 22.72, 28.64, 32.22, 28.93]


@requires_ffd
class TestGuppyHorn:
    @pytest.fixture(scope='class')
    def pattern(self):
        return read_ffd(FFD_FILE)

    @pytest.fixture(scope='class')
    def report(self, pattern):
        return crosspol_report(pattern, 35.0)

    def test_grid_and_polarization(self, pattern):
        assert pattern.polarization == 'x'
        np.testing.assert_allclose(pattern.frequencies / 1e9, GUPPY_FREQS_GHZ)

    @pytest.mark.parametrize('name', list(GUPPY_REF))
    def test_reference_values_35deg(self, report, name):
        np.testing.assert_allclose(report[name].values, GUPPY_REF[name],
                                   atol=GUPPY_TOL[name], err_msg=name)

    def test_mode_spectrum(self, report):
        rel = report['mode_power_rel_db'].values
        np.testing.assert_allclose(rel[:, 2], 0.0, atol=0.1)
        others = np.delete(rel, 2, axis=1)
        assert np.all(others < -22), others.max()

    def test_xpd_int_25deg(self, pattern):
        np.testing.assert_allclose(integrated_xpd(pattern, 25.0).values,
                                   GUPPY_XPD_INT_25, atol=0.05)

    def test_cone_edge_insensitivity_11ghz(self, pattern):
        i = 3  # 11 GHz
        for te, ref in [(35.0, 24.67), (38.0, 24.45), (40.0, 24.32)]:
            assert integrated_xpd(pattern, te).values[i] == pytest.approx(ref, abs=0.05)

    def test_phi_decimation_insensitivity(self, pattern):
        p = pattern.copy()
        p.transform_coordinates('sided')
        theta = p.theta_angles
        phi = p.phi_angles
        e_th = p.data.e_theta.values
        e_ph = p.data.e_phi.values
        dec = FarFieldSpherical(theta, phi[::4], pattern.frequencies,
                                e_th[:, :, ::4], e_ph[:, :, ::4], polarization='x')
        full = integrated_xpd(pattern, 35.0).values
        coarse = integrated_xpd(dec, 35.0).values
        assert np.all(np.abs(full - coarse) < 0.06)

    def test_check_requirements(self, report):
        r = check_requirements(report, xpd_int_min_db=20, n0_min_db=40,
                               bands_hz=[(8e9, 11e9), (13e9, 15e9)])
        assert r.attrs['all_pass'] is True
        expected_in_band = GUPPY_FREQS_GHZ != 12
        np.testing.assert_array_equal(r['in_band'].values, expected_in_band)
        assert r.attrs['worst_xpd_int_in_band_db'] == pytest.approx(24.67, abs=0.05)
