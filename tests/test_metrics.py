"""Cut metrics: peak, beamwidths, nulls, sidelobe, values at angles, XPD."""
import numpy as np
import pytest

from farfield_spherical import (analyze_cut, beamwidth_at, boresight_xpd, cut_metrics,
                                pattern_metrics, value_at, METRIC_NAMES)
from farfield_spherical import FarFieldSpherical
from farfield_spherical.polarization import polarization_xy2tp


def synthetic_cut(hpbw=20.0, sidelobe_db=-18.0, sidelobe_theta=40.0, peak=12.0):
    theta = np.arange(-90, 91, 0.5)
    main = -3.0 * (theta / (hpbw / 2)) ** 2
    side = sidelobe_db - 3.0 * ((np.abs(theta) - sidelobe_theta) / 4.0) ** 2
    return theta, 10 * np.log10(10 ** (main / 10) + 10 ** (side / 10)) + peak


def beam_pattern(freqs=(8e9, 10e9), theta=None, phi=None, beam_deg=20.0, cross=0.01):
    theta = np.arange(0, 181, 2.0) if theta is None else theta
    phi = np.arange(0, 360, 30.0) if phi is None else phi
    th = np.radians(theta)[:, None]
    ph = np.radians(phi)[None, :]
    e_co = np.stack([np.exp(-(th / np.radians(beam_deg)) ** 2) * np.ones_like(ph) * (1 + 0.1 * i)
                     for i in range(len(freqs))])
    e_cx = np.stack([cross * np.sin(th) ** 2 * np.cos(2 * ph) for _ in freqs])
    e_theta, e_phi = polarization_xy2tp(phi, e_co, e_cx)
    return FarFieldSpherical(theta, phi, np.asarray(freqs, float), e_theta, e_phi, polarization='x')


class TestAnalyzeCut:
    def test_gaussian_with_sidelobe(self):
        theta, y = synthetic_cut()
        m = analyze_cut(theta, y)
        assert m.peak_value == pytest.approx(12.0, abs=1e-6) and m.peak_theta == 0.0
        assert m.hpbw == pytest.approx(20.0, abs=0.1)
        assert m.hp_left == pytest.approx(-10.0, abs=0.1)
        assert m.sidelobe_level == pytest.approx(-18.0, abs=0.1)
        assert abs(m.sidelobe_theta) == pytest.approx(40.0, abs=0.5)
        assert m.null_depth < -20 and not m.symmetric_assumed

    def test_several_levels(self):
        theta, y = synthetic_cut(hpbw=20.0)
        m = analyze_cut(theta, y, levels_db=(3.0, 10.0))
        # a Gaussian's width scales with sqrt(level)
        assert m.beamwidths[10.0].width == pytest.approx(20.0 * np.sqrt(10 / 3), abs=0.2)
        assert 'BW10dB' in m.summary() and 'HPBW 20.0°' in m.summary()

    def test_offset_invariance(self):
        theta, y = synthetic_cut()
        a, b = analyze_cut(theta, y), analyze_cut(theta, y - 30.0)
        assert a.hpbw == b.hpbw and a.sidelobe_level == pytest.approx(b.sidelobe_level)

    def test_sided_peak_at_boresight_is_mirrored(self):
        theta = np.arange(0, 91, 1.0)
        m = analyze_cut(theta, -3.0 * (theta / 10.0) ** 2)
        assert m.symmetric_assumed and m.hpbw == pytest.approx(20.0, abs=0.1)
        assert '(sym.)' in m.summary()

    def test_no_sidelobe_and_short_or_nan_traces(self):
        theta = np.arange(-60, 61, 1.0)
        m = analyze_cut(theta, -3.0 * (theta / 15.0) ** 2)
        assert m.hpbw == pytest.approx(30.0, abs=0.1)
        assert m.sidelobe_level is None and m.null_depth is None
        t, y = synthetic_cut()
        y = y.copy(); y[::7] = np.nan
        assert analyze_cut(t, y).hpbw == pytest.approx(20.0, abs=0.6)
        assert analyze_cut([0, 1], [1, 2]) is None


class TestHelpers:
    def test_beamwidth_at_and_value_at(self):
        theta, y = synthetic_cut()
        bw = beamwidth_at(theta, y, 6.0)
        assert bw.width == pytest.approx(20.0 * np.sqrt(2), abs=0.2)
        assert value_at(theta, y, 0.0) == pytest.approx(12.0, abs=1e-6)
        assert value_at(theta, y, 0.25) == pytest.approx(np.interp(0.25, theta, y))
        assert value_at(theta, y, 200.0) is None

    def test_boresight_xpd(self):
        theta = np.array([-2.0, -1.0, 0.0, 1.0])
        assert boresight_xpd(theta, [0, 0, 10.0, 0], [0, 0, -20.0, 0]) == pytest.approx(30.0)
        assert boresight_xpd(theta, [0, 0, 10.0, 0], [0, 0, -np.inf, 0]) is None


class TestPatternMetrics:
    def test_shapes_and_values(self):
        pattern = beam_pattern()
        out = pattern_metrics(pattern, [0.0, 90.0])
        assert set(out) == set(METRIC_NAMES)
        assert all(v.shape == (2, 2) for v in out.values())
        assert out['peak_gain'][1, 0] - out['peak_gain'][0, 0] == pytest.approx(20 * np.log10(1.1), abs=1e-3)
        assert out['squint'][0, 0] == 0.0
        assert np.isfinite(out['hpbw']).all()          # sided cut, mirrored
        assert not np.isinf(out['xpd_boresight']).any()

    def test_all_cuts_when_phi_is_none(self):
        pattern = beam_pattern()
        assert pattern_metrics(pattern)['peak_gain'].shape == (2, 12)

    def test_cut_metrics_picks_nearest(self):
        pattern = beam_pattern()
        m = cut_metrics(pattern, frequency=9.9e9, phi=31.0, levels_db=(3.0, 10.0))
        assert m.peak_value == pytest.approx(20 * np.log10(1.1), abs=1e-3)
        assert 10.0 in m.beamwidths
