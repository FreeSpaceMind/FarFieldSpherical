"""compare_patterns: the equivalent multipath level between two patterns."""
import numpy as np
import pytest

from farfield_spherical import compare_patterns
from .test_coordinate_transforms import make, FREQS

TH = np.arange(-180, 181, 2.0)
PHI = np.arange(0, 180, 30.0)


def _rebuilt(pattern, e_theta, e_phi):
    """A copy with new theta/phi fields and co/cx recomputed from them."""
    out = pattern.copy()
    out.data['e_theta'] = (('frequency', 'theta', 'phi'), e_theta)
    out.data['e_phi'] = (('frequency', 'theta', 'phi'), e_phi)
    out.assign_polarization(pattern.polarization)
    return out


def _from_co_cx(pattern, co, cx):
    from farfield_spherical.polarization import polarization_xy2tp
    e_th, e_ph = polarization_xy2tp(PHI, co, cx)
    return _rebuilt(pattern, e_th, e_ph)


def _scaled(pattern, factor):
    return _rebuilt(pattern, pattern.data.e_theta.values * factor, pattern.data.e_phi.values * factor)


def _with_error(pattern, level_db, sign=+1.0):
    """Add a constant stray amplitude ``level_db`` below the co-polar peak."""
    out = pattern.copy()
    co = out.data.e_co.values
    peak = np.abs(co).max(axis=(1, 2), keepdims=True)
    error = sign * peak * 10 ** (level_db / 20.0)
    # add the error along the co-polar unit phasor so amplitudes add directly
    phasor = np.exp(1j * np.angle(co))
    return _from_co_cx(pattern, co + error * phasor, out.data.e_cx.values)


class TestEMPL:
    def test_identical_patterns_have_no_multipath(self):
        a = make(TH, PHI)
        r = compare_patterns(a, a.copy())
        assert r.attrs['reference'].startswith('dB relative')
        assert set(r.data_vars) == {'empl', 'level_difference', 'empl_max', 'empl_rms'}
        assert r['empl'].shape == (len(FREQS), len(TH), len(PHI))
        assert np.all(np.isneginf(r['empl'].values) | np.isnan(r['empl'].values))
        assert np.all(np.isneginf(r['empl_max'].values))
        np.testing.assert_allclose(r['level_difference'].values[np.isfinite(r['level_difference'].values)], 0.0)

    def test_known_error_level_is_recovered(self):
        a = make(TH, PHI)
        b = _with_error(a, -40.0)
        r = compare_patterns(a, b, normalize=False)
        # a stray signal of -40 dB added once (not with opposite signs) reads
        # as an EMPL 6 dB lower: |A - B| = e, EMPL = 20 log10(e / 2 peak)
        np.testing.assert_allclose(r['empl_max'].values, -46.0, atol=0.05)
        np.testing.assert_allclose(r['empl_rms'].values, -46.0, atol=0.05)
        finite = np.isfinite(r['empl'].values)
        np.testing.assert_allclose(r['empl'].values[finite], -46.0, atol=0.05)

    def test_opposite_sign_errors_read_as_the_error_level(self):
        a = make(TH, PHI)
        plus, minus = _with_error(a, -30.0, +1.0), _with_error(a, -30.0, -1.0)
        r = compare_patterns(plus, minus, normalize=False)
        # the two measurements differ by 2e, the EMPL is e relative to the peak of pattern 1
        expected = -30.0 - 20 * np.log10(1 + 10 ** (-30 / 20))
        np.testing.assert_allclose(r['empl_max'].values, expected, atol=0.05)

    def test_normalization_removes_a_gain_offset(self):
        a = make(TH, PHI)
        b = _scaled(a, 10 ** (0.5 / 20))            # 0.5 dB hotter everywhere
        normalized = compare_patterns(a, b)
        raw = compare_patterns(a, b, normalize=False)
        assert np.all(normalized['empl_max'].values < -120)      # round-off only
        offset = 20 * np.log10((10 ** (0.5 / 20) - 1) / 2)
        np.testing.assert_allclose(raw['empl_max'].values, offset, atol=1e-4)
        finite = np.isfinite(raw['level_difference'].values)
        np.testing.assert_allclose(raw['level_difference'].values[finite], 0.5, atol=1e-4)

    def test_complex_difference_sees_phase_error(self):
        a = make(TH, PHI)
        b = a.copy()
        # a 10 degree phase twist across theta leaves the amplitudes untouched
        twist = np.exp(1j * np.radians(10.0 * TH / 180.0))[None, :, None]
        b = _rebuilt(a, a.data.e_theta.values * twist, a.data.e_phi.values * twist)
        amplitude = compare_patterns(a, b)
        complex_ = compare_patterns(a, b, complex_difference=True)
        assert np.all(amplitude['empl_max'].values < -120)       # round-off only
        assert np.all(np.isfinite(complex_['empl_max'].values))
        assert np.all(complex_['empl_max'].values > -40) and np.all(complex_['empl_max'].values < -10)

    def test_theta_max_limits_the_summary_only(self):
        a = make(TH, PHI)
        e_th = a.data.e_theta.values.copy()
        e_th[:, np.abs(TH) > 90, :] *= 2.0             # error only in the back hemisphere
        b = _rebuilt(a, e_th, a.data.e_phi.values)
        full = compare_patterns(a, b, component='e_theta', normalize=False)
        front = compare_patterns(a, b, component='e_theta', normalize=False, theta_max=90.0)
        assert front.attrs['theta_max'] == 90.0
        assert np.all(np.isfinite(full['empl_max'].values))
        assert np.all(np.isneginf(front['empl_max'].values) | (front['empl_max'].values < full['empl_max'].values - 20))
        np.testing.assert_array_equal(full['empl'].values, front['empl'].values)

    def test_cross_polar_is_relative_to_the_co_polar_peak(self):
        a = make(TH, PHI)
        b = _from_co_cx(a, a.data.e_co.values, a.data.e_cx.values * 1.5)
        r = compare_patterns(a, b, component='e_cx', normalize=False)
        peak_co = np.abs(a.data.e_co.values).max(axis=(1, 2))
        peak_cx = np.abs(a.data.e_cx.values).max(axis=(1, 2))
        expected = 20 * np.log10(0.5 * peak_cx / (2 * peak_co))
        np.testing.assert_allclose(r['empl_max'].values, expected, atol=1e-6)

    def test_polarization_and_grid_checks(self):
        a = make(TH, PHI)
        b = make(TH, PHI)
        b.change_polarization('rhcp')
        r = compare_patterns(a, b)
        assert b.polarization == 'rhcp'                       # not converted in place
        assert np.all(r['empl_max'].values < -60)              # same field, other basis
        with pytest.raises(ValueError, match='theta'):
            compare_patterns(a, make(np.arange(0, 181, 2.0), PHI))
        with pytest.raises(ValueError, match='component'):
            compare_patterns(a, a, component='e_x')
