"""
Regression tests for defects found in a review of the package.

Each test names the behaviour that was wrong before the fix, so that a
reintroduction is obvious from the failure message.
"""
import logging

import numpy as np
import pytest

from farfield_spherical import (
    FarFieldSpherical,
    average_patterns,
    calculate_directivity,
    difference_patterns,
)
from .test_coordinate_transforms import analytic_fields, make, FREQS

TH_S = np.arange(0, 181, 2.0)
PHI_S = np.arange(0, 360, 15.0)
TH_C = np.arange(-180, 181, 2.0)
PHI_C = np.arange(0, 180, 15.0)


def isotropic(theta, phi, freqs=FREQS[:1]):
    """Unit-intensity source: |e_theta|^2 + |e_phi|^2 == 1 everywhere."""
    e = np.ones((len(freqs), len(theta), len(phi)), dtype=complex) / np.sqrt(2)
    return FarFieldSpherical(theta, phi, freqs, e, e, polarization='x')


class TestPhaseCenter:
    def test_find_phase_center_runs(self):
        """calculate_directivity's module used lightspeed without importing it,
        so every phase-center call raised NameError."""
        p = make(TH_C, PHI_C)
        pc = p.find_phase_center(20.0, FREQS[0])
        assert np.asarray(pc).shape == (3,)
        assert np.all(np.isfinite(pc))


class TestSubsample:
    def test_result_is_usable(self):
        """subsample built the result with __new__, so _theta_grid was never
        set and the next property access raised AttributeError."""
        p = make(TH_S, PHI_S)
        q = p.subsample(theta_step=10.0)
        assert q.has_uniform_theta
        assert q.theta_angles.size > 1
        assert q.polarization == 'x'
        assert np.all(np.isfinite(q.data.e_co.values))

    def test_metadata_is_not_shared_with_source(self):
        p = make(TH_S, PHI_S)
        q = p.subsample(theta_step=10.0)
        n_before = len(p.metadata['operations'])
        q.metadata['operations'].append({'type': 'marker'})
        assert len(p.metadata['operations']) == n_before


class TestShiftPhiOrigin:
    def test_recomputes_ludwig3_components(self):
        """e_co/e_cx are referred to the fixed x/y axes, so relabelling phi
        invalidates them; they used to be reordered rather than recomputed."""
        p = make(TH_S, PHI_S)
        p.shift_phi_origin(90.0)
        expected = p.copy()
        expected.assign_polarization('x')
        np.testing.assert_allclose(p.data.e_co.values, expected.data.e_co.values, atol=1e-6)
        np.testing.assert_allclose(p.data.e_cx.values, expected.data.e_cx.values, atol=1e-6)

    def test_merges_wrapped_duplicate(self):
        """phi 0..360 inclusive shifted by 45 used to produce a duplicated
        phi coordinate, which breaks .sel and every later transform."""
        p = make(TH_S, np.arange(0, 361, 15.0))
        p.shift_phi_origin(45.0)
        phi = p.phi_angles
        assert len(np.unique(phi)) == len(phi)
        assert np.all(np.diff(phi) > 0)
        assert phi.min() >= 0 and phi.max() < 360

    def test_matches_a_roll_of_the_field(self):
        p = make(TH_S, PHI_S)
        before = p.data.e_theta.values.copy()
        p.shift_phi_origin(45.0)   # 3 grid steps
        np.testing.assert_allclose(p.data.e_theta.values, np.roll(before, 3, axis=2), atol=1e-6)

    def test_single_cut_is_relabelled(self):
        """A one-cut pattern used to be a silent no-op."""
        p = make(TH_S, np.array([10.0]))
        p.shift_phi_origin(30.0)
        assert p.phi_angles[0] == pytest.approx(40.0)


class TestDirectivity:
    @pytest.mark.parametrize('theta,phi,label', [
        (TH_S, PHI_S, 'sided'),
        (TH_C, PHI_C, 'central'),
        (TH_S, np.arange(-180, 181, 15.0), 'ffd-style duplicate endpoint'),
    ])
    def test_isotropic_is_0_db(self, theta, phi, label):
        """A complete central pattern used to read as 49% coverage and be
        'extrapolated', returning about -0.1 dB for an isotropic source."""
        d, _, _ = calculate_directivity(isotropic(theta, phi), frequency=FREQS[0])
        assert d == pytest.approx(0.0, abs=0.01), label

    def test_partial_directivity_never_exceeds_total(self):
        """The denominator used to be the selected component's own power, so
        a weak cross-pol could report more directivity than the total field."""
        p = make(TH_S, PHI_S)
        total = calculate_directivity(p, frequency=FREQS[0], component='total')[0]
        for component in ('e_co', 'e_cx', 'e_theta', 'e_phi'):
            partial = calculate_directivity(p, frequency=FREQS[0], component=component)[0]
            assert partial <= total + 1e-6, component

    def test_component_directivities_add_up_at_a_direction(self):
        """With a shared denominator, D_co + D_cx == D_total at any direction."""
        p = make(TH_S, PHI_S)
        kw = dict(frequency=FREQS[0], theta=20.0, phi=45.0)
        lin = {c: 10 ** (calculate_directivity(p, component=c, **kw) / 10)
               for c in ('total', 'e_co', 'e_cx')}
        assert lin['e_co'] + lin['e_cx'] == pytest.approx(lin['total'], rel=1e-6)

    def test_single_cut_raises(self):
        """A single phi cut gave dphi = 0, so measured power was 0 and the
        answer came entirely from the extrapolation heuristic."""
        p = make(TH_S, np.array([0.0]))
        with pytest.raises(ValueError, match='at least two'):
            calculate_directivity(p, frequency=FREQS[0])

    def test_theta_without_phi_raises(self):
        """Supplying only theta silently returned the 3-tuple peak result."""
        p = make(TH_S, PHI_S)
        with pytest.raises(ValueError, match='both theta and phi'):
            calculate_directivity(p, frequency=FREQS[0], theta=10.0)

    def test_redundant_coverage_warns(self, caplog):
        """Central theta with phi 0..360 samples every direction twice."""
        p = isotropic(TH_C, np.arange(0, 360, 15.0))
        with caplog.at_level(logging.WARNING, logger='farfield_spherical.analysis'):
            calculate_directivity(p, frequency=FREQS[0])
        assert 'covers the sphere' in caplog.text

    def test_directivity_at_a_direction(self):
        p = isotropic(TH_S, PHI_S)
        d = calculate_directivity(p, frequency=FREQS[0], theta=30.0, phi=45.0)
        assert d == pytest.approx(0.0, abs=0.01)


class TestCopy:
    def test_preserves_swe(self):
        """copy() dropped the dynamically attached swe dict, so a pattern read
        from .sph lost its coefficients after any internal copy."""
        p = make(TH_S, PHI_S)
        p.swe = {FREQS[0]: 'sentinel'}
        assert p.copy().swe == {FREQS[0]: 'sentinel'}

    def test_metadata_is_deep_copied(self):
        """The 'operations' list used to be shared with the original."""
        p = make(TH_S, PHI_S)
        q = p.copy()
        n_before = len(p.metadata['operations'])
        q.metadata['operations'].append({'type': 'marker'})
        assert len(p.metadata['operations']) == n_before

    def test_copy_without_swe_has_no_attribute(self):
        p = make(TH_S, PHI_S)
        assert not getattr(p.copy(), 'swe', None)


class TestSwapPolarizationAxes:
    def test_records_one_operation(self):
        """The method contained a verbatim duplicate of its own tail, so each
        call recorded two history entries and redid the decomposition twice."""
        p = make(TH_S, PHI_S)
        before = len(p.metadata['operations'])
        p.swap_polarization_axes()
        assert len(p.metadata['operations']) == before + 1


class TestValidation:
    def test_interpolate_frequency_needs_two_frequencies(self):
        """A single frequency produced an all-NaN pattern and a scipy warning."""
        p = make(TH_S, PHI_S)
        single = FarFieldSpherical(TH_S, PHI_S, FREQS[:1],
                                   p.data.e_theta.values[:1], p.data.e_phi.values[:1],
                                   polarization='x')
        with pytest.raises(ValueError, match='at least two frequencies'):
            single.interpolate_frequency(np.array([8.5e9]))

    def test_interpolate_frequency_warns_on_extrapolation(self, caplog):
        p = make(TH_S, PHI_S)
        with caplog.at_level(logging.WARNING, logger='farfield_spherical.farfield_operations'):
            p.interpolate_frequency(np.array([FREQS[0] * 0.5]))
        assert 'extrapolated' in caplog.text

    def test_subsample_phi_endpoint_does_not_wrap_onto_zero(self):
        """phi_range=(0, 360) ended on 360, which wrapped onto 0 and left the
        phi axis non-monotonic with a duplicate."""
        p = make(TH_S, PHI_S)
        q = p.subsample(phi_range=(0.0, 360.0), phi_step=30.0)
        phi = q.phi_angles
        assert len(np.unique(phi)) == len(phi)
        assert np.all(np.diff(phi) > 0)
        assert phi.max() < 360.0

    def test_normalize_phase_rejects_non_uniform_theta(self):
        """The theta coordinate holds indices for a non-uniform pattern, so
        using it as degrees gave silently wrong results."""
        theta_grid = np.stack([TH_S, TH_S + 1.0], axis=1)
        e = np.ones((1, len(TH_S), 2), dtype=complex)
        p = FarFieldSpherical(theta_grid, np.array([0.0, 90.0]), FREQS[:1], e, e,
                              polarization='x')
        assert not p.has_uniform_theta
        with pytest.raises(NotImplementedError, match='normalize_phase'):
            p.normalize_phase()
        with pytest.raises(NotImplementedError, match='normalize_at_boresight'):
            p.normalize_at_boresight()

    def test_directivity_rejects_non_uniform_theta(self):
        theta_grid = np.stack([TH_S, TH_S + 1.0], axis=1)
        e = np.ones((1, len(TH_S), 2), dtype=complex)
        p = FarFieldSpherical(theta_grid, np.array([0.0, 90.0]), FREQS[:1], e, e,
                              polarization='x')
        with pytest.raises(NotImplementedError, match='non-uniform'):
            calculate_directivity(p, frequency=FREQS[0])

    def test_copy_history_is_not_shared_by_to_uniform_theta(self):
        theta_grid = np.stack([TH_S, TH_S], axis=1)
        e = np.ones((1, len(TH_S), 2), dtype=complex)
        p = FarFieldSpherical(theta_grid, np.array([0.0, 90.0]), FREQS[:1], e, e,
                              polarization='x')
        before = len(p.metadata.get('operations', []))
        p.to_uniform_theta()
        assert len(p.metadata.get('operations', [])) == before


class TestDualSphere:
    def test_complete_sided_sphere_is_not_dual(self):
        """Any full sided sphere used to be reported as a dual measurement."""
        from farfield_spherical import detect_dual_sphere

        p = make(TH_S, np.arange(0, 360, 10.0))
        result = detect_dual_sphere(p)
        assert result['is_dual_sphere'] is False
        assert 'sided' in result['message']

    def test_central_full_phi_is_dual(self):
        from farfield_spherical import detect_dual_sphere

        p = make(TH_C, np.arange(0, 360, 10.0))
        assert detect_dual_sphere(p)['is_dual_sphere'] is True


class TestPackageFunctions:
    def test_difference_does_not_mutate_input(self):
        """difference_patterns used to convert the caller's pattern2 in place."""
        a = make(TH_S, PHI_S)
        b = make(TH_S, PHI_S)
        b.change_polarization('rhcp')
        assert b.polarization == 'rhcp'
        difference_patterns(a, b)
        assert b.polarization == 'rhcp'

    def test_average_keeps_polarization(self):
        """Without an explicit polarization the result is auto-detected, and
        auto-detection can only return x/y/rhcp/lhcp."""
        a = make(TH_S, PHI_S)
        b = make(TH_S, PHI_S)
        a.change_polarization('theta')
        b.change_polarization('theta')
        assert average_patterns([a, b]).polarization == 'theta'

    def test_average_warns_on_mixed_polarization(self, caplog):
        a = make(TH_S, PHI_S)
        b = make(TH_S, PHI_S)
        b.change_polarization('rhcp')
        with caplog.at_level(logging.WARNING, logger='farfield_spherical.package_functions'):
            average_patterns([a, b])
        assert 'mixed polarizations' in caplog.text
