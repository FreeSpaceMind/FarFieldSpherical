"""
Pattern metrics read from a cut: peak, beamwidth at any level, first
nulls, first sidelobe, values at chosen angles, and boresight cross-polar
discrimination.

Everything here works on one trace at a time, theta in degrees against a
value in dB, so the same numbers come out whether the caller is a plot,
a marker, a frequency sweep or a script. ``pattern_metrics`` runs it over
the cuts of a FarFieldSpherical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Beamwidth:
    level_db: float                      # how far below the peak the width is taken
    left: Optional[float] = None         # interpolated theta of the left crossing
    right: Optional[float] = None
    width: Optional[float] = None
    symmetric_assumed: bool = False      # sided cut with its peak at theta = 0: one side mirrored


@dataclass
class CutMetrics:
    peak_theta: float
    peak_value: float
    beamwidths: Dict[float, Beamwidth] = field(default_factory=dict)   # keyed by level_db
    null_left: Optional[float] = None    # first local minimum either side of the main lobe
    null_right: Optional[float] = None
    null_depth: Optional[float] = None   # deeper first null relative to the peak (dB, negative)
    sidelobe_theta: Optional[float] = None
    sidelobe_level: Optional[float] = None   # highest lobe beyond the first nulls, relative to the peak (dB)

    # -- convenience for the common 3 dB case ----------------------------
    @property
    def hpbw(self) -> Optional[float]:
        bw = self.beamwidths.get(3.0)
        return bw.width if bw else None

    @property
    def hp_left(self) -> Optional[float]:
        bw = self.beamwidths.get(3.0)
        return bw.left if bw else None

    @property
    def hp_right(self) -> Optional[float]:
        bw = self.beamwidths.get(3.0)
        return bw.right if bw else None

    @property
    def symmetric_assumed(self) -> bool:
        return any(bw.symmetric_assumed for bw in self.beamwidths.values())

    def summary(self, unit: str = 'dBi') -> str:
        parts = [f"peak {self.peak_value:.2f} {unit} @ {self.peak_theta:.1f}°"]
        for level in sorted(self.beamwidths):
            bw = self.beamwidths[level]
            if bw.width is None:
                continue
            name = "HPBW" if level == 3.0 else f"BW{level:g}dB"
            parts.append(f"{name} {bw.width:.1f}°" + (" (sym.)" if bw.symmetric_assumed else ""))
        if self.sidelobe_level is not None:
            parts.append(f"SLL {self.sidelobe_level:+.1f} dB @ {self.sidelobe_theta:.1f}°")
        if self.null_depth is not None:
            parts.append(f"null {self.null_depth:+.1f} dB")
        return ", ".join(parts)


# ----------------------------------------------------------------- helpers

def _clean(theta, values) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(theta) & np.isfinite(values)
    theta, values = theta[keep], values[keep]
    order = np.argsort(theta)
    return theta[order], values[order]


def _crossing(theta, values, level, start, step) -> Optional[float]:
    """Interpolated theta where ``values`` first drops below ``level`` walking
    from index ``start`` in direction ``step``. None if it never does."""
    i = start
    n = len(values)
    while 0 <= i + step < n:
        j = i + step
        if values[j] < level:
            v0, v1 = values[i], values[j]
            if v1 == v0:
                return float(theta[j])
            frac = (v0 - level) / (v0 - v1)
            return float(theta[i] + frac * (theta[j] - theta[i]))
        i = j
    return None


def _first_minimum(values, start, step) -> Optional[int]:
    """Index of the first local minimum walking from ``start`` in direction ``step``."""
    i = start
    n = len(values)
    while 0 <= i + step < n:
        j = i + step
        if values[j] > values[i]:
            return i if i != start else None
        i = j
    return None


# ------------------------------------------------------------------ public

def beamwidth_at(theta, values, level_db: float = 3.0, peak_index: Optional[int] = None) -> Beamwidth:
    """
    Width of the main lobe ``level_db`` below its peak, with interpolated
    crossings. A sided cut (theta from 0) whose peak sits at theta = 0 only
    holds half the lobe; the other half is taken as its mirror image and
    ``symmetric_assumed`` says so.
    """
    theta, values = _clean(theta, values)
    result = Beamwidth(level_db=float(level_db))
    if len(values) < 3:
        return result
    peak = int(np.argmax(values)) if peak_index is None else int(peak_index)
    level = values[peak] - level_db
    result.left = _crossing(theta, values, level, peak, -1)
    result.right = _crossing(theta, values, level, peak, +1)
    if peak == 0 and abs(theta[0]) < 1e-9 and result.left is None and result.right is not None:
        result.left = -result.right
        result.symmetric_assumed = True
    if result.left is not None and result.right is not None:
        result.width = result.right - result.left
    return result


def value_at(theta, values, at: float) -> Optional[float]:
    """The trace's value at angle ``at``, linearly interpolated; None outside the cut."""
    theta, values = _clean(theta, values)
    if theta.size == 0 or at < theta[0] or at > theta[-1]:
        return None
    return float(np.interp(at, theta, values))


def analyze_cut(theta, values, levels_db: Sequence[float] = (3.0,)) -> Optional[CutMetrics]:
    """
    Peak, beamwidths at each of ``levels_db``, first nulls and first sidelobe
    of one trace in dB.

    Args:
        theta: angles in degrees, monotonic after sorting
        values: the trace in dB (any offset; relative results are unaffected)
        levels_db: beamwidth levels below the peak, 3 dB by default

    Returns None when the trace has fewer than three finite samples.
    """
    theta, values = _clean(theta, values)
    if len(values) < 3:
        return None
    peak = int(np.argmax(values))
    result = CutMetrics(peak_theta=float(theta[peak]), peak_value=float(values[peak]))
    for level in levels_db:
        result.beamwidths[float(level)] = beamwidth_at(theta, values, float(level), peak_index=peak)

    left = _first_minimum(values, peak, -1)
    right = _first_minimum(values, peak, +1)
    if left is not None:
        result.null_left = float(theta[left])
    if right is not None:
        result.null_right = float(theta[right])
    depths = [values[i] - values[peak] for i in (left, right) if i is not None]
    if depths:
        result.null_depth = float(min(depths))

    candidates = []
    if left is not None and left > 0:
        i = int(np.argmax(values[:left]))
        candidates.append((values[i], theta[i]))
    if right is not None and right < len(values) - 1:
        i = right + 1 + int(np.argmax(values[right + 1:]))
        candidates.append((values[i], theta[i]))
    if candidates:
        value, angle = max(candidates)
        result.sidelobe_level = float(value - values[peak])
        result.sidelobe_theta = float(angle)
    return result


def boresight_xpd(theta, co_db, cx_db) -> Optional[float]:
    """Co minus cross-polar level at the sample nearest theta = 0, or None if not finite."""
    theta = np.asarray(theta, dtype=float)
    if theta.size == 0:
        return None
    i0 = int(np.argmin(np.abs(theta)))
    value = float(np.asarray(co_db, dtype=float)[i0]) - float(np.asarray(cx_db, dtype=float)[i0])
    return value if np.isfinite(value) else None


METRIC_NAMES: Dict[str, str] = {
    'peak_gain': 'Peak gain (dBi)',
    'boresight_gain': 'Boresight gain (dBi)',
    'hpbw': 'Half-power beamwidth (deg)',
    'sidelobe_level': 'First sidelobe level (dB)',
    'squint': 'Peak angle (deg)',
    'xpd_boresight': 'Boresight XPD (dB)',
    'null_depth': 'First null depth (dB)',
}


def _gain_db(pattern, component: str) -> np.ndarray:
    with np.errstate(divide='ignore'):
        return 20 * np.log10(np.abs(pattern.data[component].values))


def pattern_metrics(pattern, phi=None, component: str = 'e_co',
                    levels_db: Sequence[float] = (3.0,)) -> Dict[str, np.ndarray]:
    """
    Every metric in ``METRIC_NAMES`` for the selected phi cuts, as arrays
    shaped (n_frequency, n_phi). A value that does not exist (no sidelobe,
    no finite cross-pol) is NaN.

    Args:
        pattern: FarFieldSpherical
        phi: phi angles to evaluate (nearest grid cut each), or None for all
        component: the co-polar component name; its cross partner is used for XPD
        levels_db: beamwidth levels; 'hpbw' reports the 3 dB one
    """
    phi_angles = np.asarray(pattern.phi_angles, dtype=float)
    if phi is None:
        phi_idx = list(range(len(phi_angles)))
    else:
        wanted = np.atleast_1d(np.asarray(phi, dtype=float))
        phi_idx = []
        for value in wanted:
            i = int(np.argmin(np.abs(phi_angles - value)))
            if i not in phi_idx:
                phi_idx.append(i)
    cross = 'e_cx' if component == 'e_co' else 'e_co'
    main = _gain_db(pattern, component)
    cx = _gain_db(pattern, cross)
    n_freq = len(pattern.frequencies)
    out = {key: np.full((n_freq, len(phi_idx)), np.nan) for key in METRIC_NAMES}
    levels = tuple(levels_db) if 3.0 in levels_db else (3.0, *levels_db)
    for fi in range(n_freq):
        for j, pj in enumerate(phi_idx):
            theta = np.asarray(pattern.get_theta_for_phi(pj) if hasattr(pattern, 'get_theta_for_phi')
                               else pattern.theta_angles, dtype=float)
            values = main[fi, :, pj]
            metrics = analyze_cut(theta, values, levels)
            if metrics is None:
                continue
            out['peak_gain'][fi, j] = metrics.peak_value
            out['squint'][fi, j] = metrics.peak_theta
            if metrics.hpbw is not None:
                out['hpbw'][fi, j] = metrics.hpbw
            if metrics.sidelobe_level is not None:
                out['sidelobe_level'][fi, j] = metrics.sidelobe_level
            if metrics.null_depth is not None:
                out['null_depth'][fi, j] = metrics.null_depth
            i0 = int(np.argmin(np.abs(theta)))
            out['boresight_gain'][fi, j] = values[i0]
            xpd = boresight_xpd(theta, values, cx[fi, :, pj])
            if xpd is not None:
                out['xpd_boresight'][fi, j] = xpd
    return out


def cut_metrics(pattern, frequency: Optional[float] = None, phi: float = 0.0,
                component: str = 'e_co', levels_db: Sequence[float] = (3.0,)) -> Optional[CutMetrics]:
    """CutMetrics for one cut of a pattern at the nearest frequency and phi."""
    frequencies = np.asarray(pattern.frequencies, dtype=float)
    fi = 0 if frequency is None else int(np.argmin(np.abs(frequencies - float(frequency))))
    pj = int(np.argmin(np.abs(np.asarray(pattern.phi_angles, dtype=float) - float(phi))))
    theta = np.asarray(pattern.get_theta_for_phi(pj) if hasattr(pattern, 'get_theta_for_phi')
                       else pattern.theta_angles, dtype=float)
    return analyze_cut(theta, _gain_db(pattern, component)[fi, :, pj], levels_db)
