"""
Feed cross-polarization metrics over an illumination cone.

Implements the quantities defined in "Cross-Polarization Metrics for a
Reflector Feed": integrated XPD, azimuthal mode content of the cross-polarized
field, and supporting point-wise and edge-taper values.

All functions operate on ``pattern.data.e_co`` and ``pattern.data.e_cx`` as
currently assigned. Ludwig-3 definitions require ``pattern.polarization`` to be
``'x'`` or ``'y'``; other polarizations are accepted with a warning because the
integrals are well defined for any co/cross pair.

The integration domain is ``0 <= theta <= theta_e``, ``0 <= phi < 360``, on a
copy of the pattern transformed to sided format. Quadrature is rectangular
with weights ``sin(theta) dtheta dphi``, which is what the requirement text
specifies. Uniform theta spacing inside the cone and a uniform phi grid
covering a full 360 degrees exactly once are required.
"""
import logging
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Floor applied inside logarithms to avoid divide-by-zero on identically zero fields.
_FLOOR = 1e-300


class _Cone(NamedTuple):
    """Field samples restricted to the illumination cone, shaped (frequency, theta, phi)."""
    e_co: np.ndarray          # co-pol inside the cone
    e_cx: np.ndarray          # cross-pol inside the cone
    theta: np.ndarray         # cone theta samples, radians
    phi: np.ndarray           # phi samples, radians (full 360, no duplicates)
    dtheta: float             # theta step, radians
    dphi: float               # phi step, radians
    e_co_full: np.ndarray     # co-pol over the full (sided) pattern, for the peak
    theta_full: np.ndarray    # full sided theta grid, degrees


def _dedupe_phi(phi: np.ndarray, *fields: np.ndarray):
    """
    Reduce phi to unique angles modulo 360, keeping the first occurrence of
    each and sorting ascending. Handles a duplicated endpoint (0 and 360, or
    -180 and +180) as well as an interior duplicate such as the 180/180 pair
    produced when a central pattern with phi 0..180 inclusive is transformed
    to sided format.
    """
    phi_mod = np.mod(phi, 360.0)
    # Snap values within tolerance of 360 back to 0 so 359.9999 and 0 collapse.
    phi_mod = np.where(np.isclose(phi_mod, 360.0, atol=1e-6), 0.0, phi_mod)
    order = np.argsort(phi_mod, kind="stable")
    phi_sorted = phi_mod[order]
    keep = np.ones(len(phi_sorted), dtype=bool)
    keep[1:] = ~np.isclose(np.diff(phi_sorted), 0.0, atol=1e-6)
    idx = order[keep]
    return phi_sorted[keep], tuple(f[:, :, idx] for f in fields)


def _prepare_cone(pattern, theta_e: float) -> _Cone:
    """
    Restrict a pattern to the illumination cone in sided format.

    Returns a ``_Cone`` with arrays shaped (frequency, theta, phi). The
    caller's pattern is never modified.

    Raises:
        ValueError: If the phi grid is non-uniform or does not cover a full
            360 degrees, if theta spacing inside the cone is non-uniform, or
            if ``theta_e`` lies outside the pattern's theta range.
    """
    if pattern.polarization not in ("x", "y"):
        logger.warning(
            "Cross-pol metrics computed with polarization '%s'; Ludwig-3 "
            "requires 'x' or 'y'.", pattern.polarization)

    p = pattern.copy()
    p.transform_coordinates("sided")

    theta = np.asarray(p.theta_angles, dtype=float)
    phi = np.asarray(p.phi_angles, dtype=float)
    e_co = np.asarray(p.data.e_co.values, dtype=np.complex128)
    e_cx = np.asarray(p.data.e_cx.values, dtype=np.complex128)

    phi, (e_co, e_cx) = _dedupe_phi(phi, e_co, e_cx)

    if len(phi) < 2:
        raise ValueError("Cross-pol metrics require at least two distinct phi samples.")
    dphi_all = np.diff(phi)
    if not np.allclose(dphi_all, dphi_all[0], atol=1e-6):
        raise ValueError("Cross-pol metrics require a uniform phi grid.")
    dphi = np.deg2rad(dphi_all[0])
    if not np.isclose(len(phi) * dphi, 2 * np.pi, atol=1e-6):
        raise ValueError(
            "Cross-pol metrics require phi coverage of a full 360 degrees "
            f"(found {len(phi)} samples at {np.rad2deg(dphi):.4g} deg spacing).")

    if theta_e > theta.max() + 1e-9:
        raise ValueError(
            f"theta_e = {theta_e} deg exceeds pattern theta range "
            f"(max {theta.max():.4g} deg).")
    mask = theta <= theta_e + 1e-9
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Fewer than two theta samples inside theta_e = {theta_e} deg.")

    theta_c = theta[mask]
    dth_all = np.diff(theta_c)
    if not np.allclose(dth_all, dth_all[0], atol=1e-6):
        raise ValueError("Cross-pol metrics require uniform theta spacing inside the cone.")
    dtheta = np.deg2rad(dth_all[0])

    return _Cone(
        e_co=e_co[:, mask, :],
        e_cx=e_cx[:, mask, :],
        theta=np.deg2rad(theta_c),
        phi=np.deg2rad(phi),
        dtheta=dtheta,
        dphi=dphi,
        e_co_full=e_co,
        theta_full=theta,
    )


def _cone_power(field: np.ndarray, cone: _Cone) -> np.ndarray:
    """Rectangular-rule power integral of |field|^2 over the cone, per frequency."""
    w = (np.sin(cone.theta) * cone.dtheta * cone.dphi)[None, :, None]
    return np.sum(np.abs(field) ** 2 * w, axis=(1, 2))


def _integrated_xpd(cone: _Cone) -> np.ndarray:
    p_co = _cone_power(cone.e_co, cone)
    p_cx = _cone_power(cone.e_cx, cone)
    return 10 * np.log10(np.maximum(p_co, _FLOOR) / np.maximum(p_cx, _FLOOR))


def integrated_xpd(pattern, theta_e: float) -> xr.DataArray:
    """
    Integrated cross-polarization discrimination over 0 <= theta <= theta_e.

        XPD_int = 10 log10 [ sum |E_co|^2 sin(theta) dtheta dphi
                           / sum |E_cx|^2 sin(theta) dtheta dphi ]

    Args:
        pattern: FarFieldSpherical object
        theta_e: Illumination half-angle in degrees

    Returns:
        DataArray ``xpd_int_db`` in dB indexed by frequency.
    """
    cone = _prepare_cone(pattern, theta_e)
    return xr.DataArray(_integrated_xpd(cone),
                        coords={"frequency": pattern.frequencies},
                        dims=["frequency"], name="xpd_int_db",
                        attrs={"theta_e_deg": theta_e, "units": "dB"})


def _azimuthal_modes(cone: _Cone, n_max: int, component: str) -> xr.Dataset:
    if component not in ("e_cx", "e_co"):
        raise ValueError(f"component must be 'e_cx' or 'e_co', got '{component}'.")
    field = cone.e_cx if component == "e_cx" else cone.e_co
    nfreq, _, nphi = field.shape
    if n_max < 0:
        raise ValueError("n_max must be non-negative.")
    if n_max > nphi // 2:
        raise ValueError(f"n_max = {n_max} exceeds Nyquist for {nphi} phi samples.")

    # c_n = (1/2pi) int E e^{-jn phi} dphi  ==  FFT(E)/N on a uniform grid.
    spec = np.fft.fft(field, axis=2) / nphi            # (freq, theta, k)
    n_signed = np.arange(-n_max, n_max + 1)
    idx = np.mod(n_signed, nphi)
    c_n = spec[:, :, idx].transpose(0, 2, 1)           # (freq, n_signed, theta)

    # Power in each signed bin over the cone: int |c_n|^2 sin(theta) dtheta * 2pi
    wt = (np.sin(cone.theta) * cone.dtheta * 2 * np.pi)[None, None, :]
    p_signed = np.sum(np.abs(c_n) ** 2 * wt, axis=2)   # (freq, n_signed)

    n = np.arange(0, n_max + 1)
    mode_power = np.zeros((nfreq, n_max + 1))
    zero = n_max                                       # position of n = 0 in n_signed
    mode_power[:, 0] = p_signed[:, zero]
    for k in range(1, n_max + 1):
        if 2 * k == nphi:
            # +k and -k alias to the same Nyquist bin; count it once.
            mode_power[:, k] = p_signed[:, zero + k]
        else:
            mode_power[:, k] = p_signed[:, zero + k] + p_signed[:, zero - k]

    # Normalise to the total power of the component over the cone (all bins,
    # by Parseval), so the relative values do not depend on n_max.
    total = _cone_power(field, cone)[:, None]
    rel_db = 10 * np.log10(np.maximum(mode_power, _FLOOR) / np.maximum(total, _FLOOR))

    return xr.Dataset(
        {
            "c_n": (("frequency", "n_signed", "theta"), c_n),
            "mode_power": (("frequency", "n"), mode_power),
            "mode_power_rel_db": (("frequency", "n"), rel_db),
        },
        coords={"n_signed": n_signed, "n": n, "theta": np.rad2deg(cone.theta)},
        attrs={"component": component},
    )


def azimuthal_modes(pattern, theta_e: float, n_max: int = 6,
                    component: str = "e_cx") -> xr.Dataset:
    """
    Azimuthal Fourier decomposition of a field component over the cone.

        c_n(theta) = (1/2pi) int E(theta, phi) e^{-j n phi} dphi

    evaluated with an FFT along phi.

    Args:
        pattern: FarFieldSpherical object
        theta_e: Illumination half-angle in degrees
        n_max: Highest azimuthal order retained (must not exceed Nyquist)
        component: ``'e_cx'`` (default) or ``'e_co'``

    Returns:
        Dataset with
            c_n                complex, dims (frequency, n_signed, theta): raw
                               bins for n_signed in [-n_max, n_max]
            mode_power         dims (frequency, n): power in |n| over the cone;
                               n = 0 is the DC bin, n >= 1 is the sum of the
                               +n and -n bins
            mode_power_rel_db  dims (frequency, n): mode_power relative to the
                               total power of the component over the cone, dB
    """
    cone = _prepare_cone(pattern, theta_e)
    ds = _azimuthal_modes(cone, n_max, component)
    ds = ds.assign_coords(frequency=pattern.frequencies)
    ds.attrs["theta_e_deg"] = theta_e
    return ds


def _co_peak(cone: _Cone) -> np.ndarray:
    """Peak co-pol amplitude over the whole pattern, per frequency."""
    return np.max(np.abs(cone.e_co_full), axis=(1, 2))


def _n0_level(cone: _Cone) -> np.ndarray:
    modes = _azimuthal_modes(cone, n_max=0, component="e_cx")
    c0 = modes["c_n"].values[:, 0, :]                   # (freq, theta)
    c0_max = np.max(np.abs(c0), axis=1)
    return 20 * np.log10(_co_peak(cone) / np.maximum(c0_max, _FLOOR))


def n0_crosspol_level(pattern, theta_e: float) -> xr.DataArray:
    """
    Requirement 1 quantity: peak co-pol amplitude over the n = 0 azimuthal
    component of the cross-pol field, worst case over theta <= theta_e, in dB.

        L0 = 20 log10 ( max|E_co| / max_theta |c_0(theta)| )

    Larger is better. The peak co-pol amplitude is taken over the whole
    pattern, not just the cone.
    """
    cone = _prepare_cone(pattern, theta_e)
    return xr.DataArray(_n0_level(cone),
                        coords={"frequency": pattern.frequencies},
                        dims=["frequency"], name="n0_level_db",
                        attrs={"theta_e_deg": theta_e, "units": "dB"})


def _point_xpd(cone: _Cone) -> Tuple[np.ndarray, np.ndarray]:
    ratio = 20 * np.log10(np.maximum(np.abs(cone.e_co), _FLOOR)
                          / np.maximum(np.abs(cone.e_cx), _FLOOR))
    worst = np.min(ratio, axis=(1, 2))
    cx_pk = np.max(np.abs(cone.e_cx), axis=(1, 2))
    peak = 20 * np.log10(_co_peak(cone) / np.maximum(cx_pk, _FLOOR))
    return worst, peak


def point_xpd(pattern, theta_e: float) -> xr.Dataset:
    """
    Point-wise cross-pol figures over the cone, for comparison only.

        xpd_worst_db   min over the cone of 20 log10(|E_co| / |E_cx|) at the same angle
        xpol_peak_db   20 log10( max|E_co| / max over the cone of |E_cx| )
    """
    cone = _prepare_cone(pattern, theta_e)
    worst, peak = _point_xpd(cone)
    return xr.Dataset(
        {"xpd_worst_db": ("frequency", worst), "xpol_peak_db": ("frequency", peak)},
        coords={"frequency": pattern.frequencies}, attrs={"theta_e_deg": theta_e})


def _edge_taper(pattern, theta_e: float) -> Tuple[np.ndarray, float]:
    p = pattern.copy()
    p.transform_coordinates("sided")
    theta = np.asarray(p.theta_angles, dtype=float)
    i = int(np.argmin(np.abs(theta - theta_e)))
    e_co = np.asarray(p.data.e_co.values, dtype=np.complex128)
    edge = np.mean(np.abs(e_co[:, i, :]), axis=1)
    pk = np.max(np.abs(e_co), axis=(1, 2))
    val = 20 * np.log10(np.maximum(edge, _FLOOR) / np.maximum(pk, _FLOOR))
    return val, float(theta[i])


def edge_taper(pattern, theta_e: float) -> xr.DataArray:
    """
    phi-averaged co-pol amplitude at theta_e relative to peak, in dB.

    Sanity check on the pairing of feed and illumination angle. The theta
    sample nearest ``theta_e`` is used; its value is reported in the
    ``theta_actual_deg`` attribute.
    """
    val, theta_actual = _edge_taper(pattern, theta_e)
    return xr.DataArray(val, coords={"frequency": pattern.frequencies},
                        dims=["frequency"], name="edge_taper_db",
                        attrs={"theta_e_deg": theta_e, "theta_actual_deg": theta_actual,
                               "units": "dB"})


def crosspol_report(pattern, theta_e: float, n_max: int = 6) -> xr.Dataset:
    """
    All feed cross-pol metrics in one Dataset indexed by frequency.

    Variables:
        xpd_int_db, n0_level_db, xpd_worst_db, xpol_peak_db, edge_taper_db
            dims (frequency)
        mode_power_rel_db
            dims (frequency, n): cross-pol azimuthal mode power relative to
            total cross-pol power in the cone

    Attrs: theta_e_deg, theta_actual_deg, n_max, polarization
    """
    cone = _prepare_cone(pattern, theta_e)
    worst, peak = _point_xpd(cone)
    taper, theta_actual = _edge_taper(pattern, theta_e)
    modes = _azimuthal_modes(cone, n_max, "e_cx")

    ds = xr.Dataset(
        {
            "xpd_int_db": ("frequency", _integrated_xpd(cone)),
            "n0_level_db": ("frequency", _n0_level(cone)),
            "xpd_worst_db": ("frequency", worst),
            "xpol_peak_db": ("frequency", peak),
            "edge_taper_db": ("frequency", taper),
            "mode_power_rel_db": (("frequency", "n"), modes["mode_power_rel_db"].values),
        },
        coords={"frequency": pattern.frequencies, "n": modes["n"].values},
        attrs={"theta_e_deg": theta_e, "theta_actual_deg": theta_actual,
               "n_max": n_max, "polarization": pattern.polarization},
    )
    for name in ("xpd_int_db", "n0_level_db", "xpd_worst_db", "xpol_peak_db",
                 "edge_taper_db", "mode_power_rel_db"):
        ds[name].attrs["units"] = "dB"
    return ds


def check_requirements(report: xr.Dataset,
                       xpd_int_min_db: Optional[float] = None,
                       n0_min_db: Optional[float] = None,
                       bands_hz: Optional[Sequence[Tuple[float, float]]] = None
                       ) -> xr.Dataset:
    """
    Evaluate pass/fail and margin per frequency against the two requirements.

    Frequencies outside ``bands_hz`` (if given) are flagged out of band and
    excluded from the pass/fail summary.

    Adds to a copy of ``report``:
        in_band (bool), xpd_int_margin_db, xpd_int_pass (bool),
        n0_margin_db, n0_pass (bool)
    and attrs: worst_xpd_int_in_band_db, worst_n0_in_band_db, all_pass.

    ``all_pass`` is True when every in-band frequency meets every requirement
    that was given. If no frequency is in band, ``all_pass`` is False.
    """
    r = report.copy()
    f = r["frequency"].values
    if bands_hz:
        in_band = np.zeros(len(f), dtype=bool)
        for lo, hi in bands_hz:
            in_band |= (f >= lo) & (f <= hi)
    else:
        in_band = np.ones(len(f), dtype=bool)
    r["in_band"] = ("frequency", in_band)

    any_in_band = bool(np.any(in_band))
    all_pass = any_in_band
    if xpd_int_min_db is not None:
        m = r["xpd_int_db"].values - xpd_int_min_db
        r["xpd_int_margin_db"] = ("frequency", m)
        r["xpd_int_pass"] = ("frequency", m >= 0)
        if any_in_band:
            r.attrs["worst_xpd_int_in_band_db"] = float(np.min(r["xpd_int_db"].values[in_band]))
            all_pass &= bool(np.all(m[in_band] >= 0))
        r.attrs["xpd_int_min_db"] = float(xpd_int_min_db)
    if n0_min_db is not None:
        m = r["n0_level_db"].values - n0_min_db
        r["n0_margin_db"] = ("frequency", m)
        r["n0_pass"] = ("frequency", m >= 0)
        if any_in_band:
            r.attrs["worst_n0_in_band_db"] = float(np.min(r["n0_level_db"].values[in_band]))
            all_pass &= bool(np.all(m[in_band] >= 0))
        r.attrs["n0_min_db"] = float(n0_min_db)
    r.attrs["all_pass"] = all_pass
    return r
