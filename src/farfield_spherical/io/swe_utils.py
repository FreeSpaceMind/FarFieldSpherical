from typing import Optional, Sequence

import numpy as np
from ..farfield import FarFieldSpherical

try:
    from swe import SphericalWaveExpansion  # pyright: ignore[reportMissingImports]
    _SWE_AVAILABLE = True
except ImportError:
    _SWE_AVAILABLE = False
    SphericalWaveExpansion = None  # type: ignore[assignment,misc]


def _match_frequency(freq: float, available: Sequence[float], rtol: float = 1e-6) -> float:
    """Return the available frequency matching freq within tolerance."""
    available_array = np.asarray(available, dtype=float).reshape(-1)
    if available_array.size == 0:
        raise ValueError("No SWE frequencies are available")

    target = float(freq)
    matches = available_array[np.isclose(available_array, target, rtol=rtol, atol=0.0)]
    if matches.size:
        return float(matches[np.argmin(np.abs(matches - target))])

    available_text = ", ".join(f"{value / 1e9:.9g} GHz" for value in available_array)
    raise ValueError(
        f"Frequency {target / 1e9:.9g} GHz is not available. "
        f"Available frequencies: {available_text}"
    )


def _available_swe_frequencies(swe: SphericalWaveExpansion) -> np.ndarray:
    frequencies = getattr(swe, "frequencies", None)
    if frequencies is not None and len(frequencies) > 0:
        return np.asarray(frequencies, dtype=float)

    frequency = getattr(swe, "frequency", None)
    if frequency is not None:
        return np.asarray([float(frequency)], dtype=float)

    raise ValueError("SWE object does not define any frequencies")


def create_pattern_from_swe(
    swe: SphericalWaveExpansion,
    theta_angles: Optional[np.ndarray] = None,
    phi_angles: Optional[np.ndarray] = None,
    frequencies: Optional[Sequence[float]] = None,
    frequency: Optional[float] = None,
) -> FarFieldSpherical:
    """
    Create FarFieldSpherical from a SphericalWaveExpansion object.

    Args:
        swe: SphericalWaveExpansion object.
        theta_angles: Theta angles in degrees. Defaults to 0..180 in 1 degree steps.
        phi_angles: Phi angles in degrees. Defaults to 0..360 in 5 degree steps.
        frequencies: Frequencies in Hz to reconstruct. None reconstructs all.
        frequency: Backward-compatible alias for a single requested frequency.

    Returns:
        FarFieldSpherical object.
    """
    if frequencies is not None and frequency is not None:
        raise ValueError("Specify either frequencies or frequency, not both")
    if frequency is not None:
        frequencies = [frequency]

    if theta_angles is None:
        theta_angles = np.linspace(0, 180, 181)
    if phi_angles is None:
        phi_angles = np.arange(0, 361, 5.0)

    theta_rad = np.radians(theta_angles)
    phi_rad = np.radians(phi_angles)
    THETA, PHI = np.meshgrid(theta_rad, phi_rad, indexing="ij")

    available = _available_swe_frequencies(swe)
    if frequencies is None:
        requested = [float(freq) for freq in available]
    else:
        requested = [
            _match_frequency(float(freq), available)
            for freq in np.asarray(frequencies, dtype=float).reshape(-1)
        ]
    if not requested:
        raise ValueError("At least one frequency is required")

    e_theta_fields = []
    e_phi_fields = []
    for freq in requested:
        try:
            E_theta, E_phi = swe.far_field(
                THETA.ravel(),
                PHI.ravel(),
                frequency=freq,
            )
        except TypeError:
            swe.frequency = freq
            E_theta, E_phi = swe.far_field(THETA.ravel(), PHI.ravel())
        e_theta_fields.append(E_theta.reshape(THETA.shape))
        e_phi_fields.append(E_phi.reshape(PHI.shape))

    frequency_array = np.asarray(requested, dtype=float)
    pattern = FarFieldSpherical(
        theta=theta_angles,
        phi=phi_angles,
        frequency=frequency_array,
        e_theta=np.stack(e_theta_fields, axis=0),
        e_phi=np.stack(e_phi_fields, axis=0),
        polarization="theta",
    )

    pattern.swe = {}
    for freq in frequency_array:
        pattern.swe[float(freq)] = swe

    return pattern
