"""
Analysis functions for antenna radiation patterns.
"""
import logging

import numpy as np
from scipy import optimize
from typing import Dict, Tuple, Optional, List, Union
import xarray as xr

from .utilities import find_nearest, frequency_to_wavelength, lightspeed
from .pattern_operations import unwrap_phase, phase_pattern_translate
from .polarization import polarization_tp2rl

logger = logging.getLogger(__name__)


def _integrate_solid_angle(values: np.ndarray, theta_rad: np.ndarray,
                           phi_rad: np.ndarray) -> float:
    """
    Integrate ``values`` over solid angle on a (theta, phi) grid.

        I = int int values * sin|theta| dtheta dphi

    ``sin|theta|`` keeps the weight positive for central-format grids, where
    theta runs negative. Theta uses the trapezoid rule. Phi uses a periodic
    (rectangular) rule when the grid is uniform and spans a full period, which
    is exact for a periodic integrand and avoids losing the last wedge of a
    grid such as 0, 15, ..., 345; otherwise it falls back to the trapezoid rule.

    Args:
        values: Array shaped (n_theta, n_phi)
        theta_rad: Theta samples in radians, ascending
        phi_rad: Phi samples in radians, ascending

    Returns:
        float: The integral
    """
    weighted = values * np.sin(np.abs(theta_rad))[:, None]

    dphi_all = np.diff(phi_rad)
    period = np.pi if _is_central_phi_span(phi_rad) else 2 * np.pi
    periodic = (len(phi_rad) > 1
                and np.allclose(dphi_all, dphi_all[0], atol=1e-9)
                and np.isclose(len(phi_rad) * dphi_all[0], period, atol=1e-6))
    if periodic:
        over_phi = np.sum(weighted, axis=1) * dphi_all[0]
    else:
        over_phi = np.trapezoid(weighted, phi_rad, axis=1)

    return float(np.trapezoid(over_phi, theta_rad))


def _is_central_phi_span(phi_rad: np.ndarray) -> bool:
    """True when the phi grid spans about 180 degrees rather than 360."""
    return bool(phi_rad[-1] - phi_rad[0] < 1.05 * np.pi)


def calculate_phase_center(pattern, theta_angle: float, frequency: Optional[float] = None, 
                       n_iter: int = 10) -> np.ndarray:
    """
    Finds the optimum phase center given a theta angle and frequency.
    
    The optimum phase center is the point that, when used as the origin,
    minimizes the phase variation across the beam within the +/- theta_angle range.
    Uses basinhopping optimization to find the global minimum.
    
    Args:
        pattern: FarFieldSpherical object
        theta_angle: Angle in degrees to optimize phase center for
        frequency: Optional specific frequency to use, or None to use first frequency
        n_iter: Number of iterations for basinhopping
        
    Returns:
        np.ndarray: [x, y, z] coordinates of the optimum phase center
    """
    # Always work in central format for symmetric +/- theta_angle handling
    original_format = detect_coordinate_format(pattern)
    if original_format == 'sided':
        pattern = pattern.copy()
        pattern.transform_coordinates('central')
    
    # Get data arrays
    freq_array = pattern.data.frequency.values
    theta_array = pattern.data.theta.values
    phi_array = pattern.data.phi.values
    
    # Handle frequency selection
    if frequency is None:
        freq_idx = 0
        freq = freq_array[freq_idx]
    else:
        freq_val, freq_idx = find_nearest(freq_array, frequency)
        if isinstance(freq_idx, np.ndarray):
            freq_idx = freq_idx.item()
        freq = freq_array[freq_idx]
    
    # Find the indices corresponding to +/- theta_angle
    theta_n, idx_n = find_nearest(theta_array, -theta_angle)
    theta_p, idx_p = find_nearest(theta_array, theta_angle)
    
    if isinstance(idx_n, np.ndarray):
        idx_n = idx_n.item()
    if isinstance(idx_p, np.ndarray):
        idx_p = idx_p.item()
    
    # Ensure idx_n < idx_p
    if idx_n > idx_p:
        idx_n, idx_p = idx_p, idx_n
    
    # Get original co-pol data
    e_co = pattern.data.e_co.values[freq_idx, :, :]
    
    # Define cost function for optimization
    def phase_center_cost(translation):
        """Calculate phase spread/flatness after applying translation."""
        # Convert theta and phi to radians for phase_pattern_translate
        theta_rad = np.radians(theta_array)
        phi_rad = np.radians(phi_array)
        
        # Get original phase
        co_phase = np.angle(e_co)
        
        # Apply the translation to the phase pattern
        translated_phase = phase_pattern_translate(
            freq, theta_rad, phi_rad, translation, co_phase)
        
        # Extract the region of interest (+/- theta_angle)
        roi_phase = translated_phase[idx_n:idx_p+1, :]
        
        # Unwrap phase
        unwrapped_phases = unwrap_phase(roi_phase, axis=0)
        
        # Calculate flatness metric (overall deviation from zero across whole region)
        theta_indices = np.arange(unwrapped_phases.shape[0])
        boresight_idx = np.argmin(np.abs(theta_indices - (len(theta_indices) // 2)))
        boresight_phase = np.mean(unwrapped_phases[boresight_idx, :])
        normalized_phases = unwrapped_phases - boresight_phase
        flatness_metric = np.std(normalized_phases)
        
        # Return appropriate metric based on method
        return flatness_metric
    
    # Calculate wavelength for scaled step sizes
    wavelength = lightspeed / freq
    step_size = wavelength / 20  # Reasonable step size based on wavelength
    
    # Define a step-taking function for basinhopping
    class PhaseStepTaker(object):
        def __init__(self, stepsize=step_size):
            self.stepsize = stepsize
            
        def __call__(self, x):
            # Take random steps proportional to wavelength
            x_new = np.copy(x)
            x_new += np.random.uniform(-self.stepsize, self.stepsize, x.shape)
            return x_new
    
    # Define a simple bounds function
    def bounds_check(x_new, **kwargs):
        """Check if the new position is within bounds."""
        max_value = 2.0  # Max distance in meters
        return bool(np.all(np.abs(x_new) < max_value))
        
    # Run basinhopping
    initial_guess = np.zeros(3)
    minimizer_kwargs = {"method": "Nelder-Mead"}
    
    result = optimize.basinhopping(
        phase_center_cost,
        initial_guess,
        niter=n_iter,
        T=1.0,  # Temperature parameter (higher allows escaping deeper local minima)
        stepsize=step_size,
        take_step=PhaseStepTaker(),
        accept_test=bounds_check,
        minimizer_kwargs=minimizer_kwargs
    )
    
    translation = result.x
    
    # Check for reasonable results
    if np.any(np.isnan(translation)) or np.any(np.isinf(translation)):
        translation = np.zeros(3)
        
    # Limit to reasonable range
    max_value = 2  # 2 meters
    if np.any(np.abs(translation) > max_value):
        translation = np.clip(translation, -max_value, max_value)
        
    return translation

def principal_plane_phase_center(frequency, theta1, theta2, theta3, phase1, phase2, phase3):
    """
    Calculate phase center using three points on a principal plane.
    
    Args:
        frequency: Frequency in Hz
        theta1: Theta angle (radians) of first point
        theta2: Theta angle (radians) of second point
        theta3: Theta angle (radians) of third point
        phase1: Phase (radians) of first point
        phase2: Phase (radians) of second point
        phase3: Phase (radians) of third point
        
    Returns:
        Tuple[ndarray, ndarray]: Planar and z-axis displacement
    """
    
    wavelength = frequency_to_wavelength(frequency)
    wavenumber = 2 * np.pi / wavelength
    
    # Ensure all inputs are arrays
    if np.isscalar(theta1): theta1 = np.array([theta1])
    if np.isscalar(theta2): theta2 = np.array([theta2])
    if np.isscalar(theta3): theta3 = np.array([theta3])
    if np.isscalar(phase1): phase1 = np.array([phase1])
    if np.isscalar(phase2): phase2 = np.array([phase2])
    if np.isscalar(phase3): phase3 = np.array([phase3])
    
    # Compute denominators first to check for division by zero
    denom1 = ((np.cos(theta2) - np.cos(theta3)) * (np.sin(theta2) - np.sin(theta1))) - \
            ((np.cos(theta2) - np.cos(theta1)) * (np.sin(theta2) - np.sin(theta3)))
    
    # Avoid division by zero
    if np.any(np.abs(denom1) < 1e-10):
        denom1 = np.where(np.abs(denom1) < 1e-10, 1e-10 * np.sign(denom1), denom1)
    
    planar_displacement = (1 / wavenumber) * (
        (
            ((phase2 - phase1) * (np.cos(theta2) - np.cos(theta3)))
            - ((phase2 - phase3) * (np.cos(theta2) - np.cos(theta1)))
        ) / denom1
    )
    
    zaxis_displacement = (1 / wavenumber) * (
        (
            ((phase2 - phase3) * (np.sin(theta2) - np.sin(theta1)))
            - ((phase2 - phase1) * (np.sin(theta2) - np.sin(theta3)))
        ) / denom1
    )
        
    # Check for invalid results
    if np.any(np.isnan(planar_displacement)) or np.any(np.isinf(planar_displacement)) or \
        np.any(np.isnan(zaxis_displacement)) or np.any(np.isinf(zaxis_displacement)):
        return np.zeros_like(planar_displacement), np.zeros_like(zaxis_displacement)
    
    return planar_displacement.flatten(), zaxis_displacement.flatten()


def get_axial_ratio(pattern):
    """
    Calculate the axial ratio (ratio of major to minor axis of polarization ellipse).
    
    Args:
        pattern: FarFieldSpherical object

    Returns:
        xr.DataArray: Axial ratio (linear scale)
    """

    # Convert to circular polarization components if not already
    if pattern.polarization in ['rhcp', 'lhcp']:
        e_r = pattern.data.e_co if pattern.polarization == 'rhcp' else pattern.data.e_cx
        e_l = pattern.data.e_cx if pattern.polarization == 'rhcp' else pattern.data.e_co
    else:
        # Need to calculate circular components
        e_r, e_l = polarization_tp2rl(
            pattern.data.phi.values,
            pattern.data.e_theta.values, 
            pattern.data.e_phi.values
        )
        e_r = xr.DataArray(
            e_r, 
            dims=pattern.data.e_theta.dims,
            coords=pattern.data.e_theta.coords
        )
        e_l = xr.DataArray(
            e_l, 
            dims=pattern.data.e_theta.dims,
            coords=pattern.data.e_theta.coords
        )
    
    # Calculate axial ratio
    er_mag = np.abs(e_r)
    el_mag = np.abs(e_l)
    
    # Handle pure circular polarization case
    min_val = 1e-15
    er_mag = xr.where(er_mag < min_val, min_val, er_mag)
    el_mag = xr.where(el_mag < min_val, min_val, el_mag)
    
    # Calculate axial ratio
    return 20 * np.log10((er_mag + el_mag) / np.maximum(np.abs(er_mag - el_mag), min_val))

def calculate_directivity(
    pattern, 
    frequency: Optional[float] = None,
    theta: Optional[float] = None,
    phi: Optional[float] = None,
    component: str = 'total',
    partial_sphere_threshold: float = 0.8,
    edge_extrapolation_db: float = -10.0,
    far_sidelobe_level_db: Optional[float] = None
) -> Union[float, Tuple[float, float, float]]:
    """
    Calculate antenna directivity from radiation pattern.
    
    Automatically handles both full-sphere and partial-sphere patterns.
    For partial coverage (e.g., near-field transformed data), uses power 
    extrapolation to estimate total radiated power.
    
    Directivity is calculated as D(θ,φ) = 4π * U(θ,φ) / P_total where:
    - U(θ,φ) is the radiation intensity at angle (θ,φ)
    - P_total is the total radiated power integrated over the sphere
    
    Args:
        pattern: FarFieldSpherical object
        frequency: Frequency in Hz. If None, uses first frequency
        theta: Theta angle in degrees for specific direction calculation.
               If None, calculates peak directivity
        phi: Phi angle in degrees for specific direction calculation.
             If None, calculates peak directivity  
        component: Field component to use ('e_co', 'e_cx', 'e_theta', 'e_phi', or 'total')
                  - 'total': |E_θ|² + |E_φ|²
                  - 'e_co'/'e_cx': co-pol or cross-pol components
                  - 'e_theta'/'e_phi': theta or phi components
        partial_sphere_threshold: Coverage fraction below which to use partial sphere method
        edge_extrapolation_db: Additional dB drop for unmeasured regions (used if far_sidelobe_level_db not specified)
        far_sidelobe_level_db: If specified, assume unmeasured regions are at this level below peak (dB).
                              This is more accurate than edge extrapolation for antenna patterns.
                              Typical values: -35 to -50 dB for high-gain antennas.
    
    Returns:
        If theta and phi are specified:
            float: Directivity in dB at the specified direction
        If peak directivity is requested (theta=None, phi=None):
            Tuple[float, float, float]: (peak_directivity_dB, peak_theta_deg, peak_phi_deg)
    
    Raises:
        ValueError: If component is not recognized or pattern data is invalid
    """
    
    # Get data arrays
    freq_array = pattern.data.frequency.values
    theta_array = pattern.data.theta.values
    phi_array = pattern.data.phi.values
    
    # Handle frequency selection
    if frequency is None:
        freq_idx = 0
        freq = freq_array[freq_idx]
    else:
        freq_val, freq_idx = find_nearest(freq_array, frequency)
        if isinstance(freq_idx, np.ndarray):
            freq_idx = freq_idx.item()
        freq = freq_array[freq_idx]
    
    if not pattern.has_uniform_theta:
        raise NotImplementedError(
            "calculate_directivity does not support non-uniform theta grids. "
            "Use .to_uniform_theta() first to interpolate to a common grid.")

    if (theta is None) != (phi is None):
        raise ValueError(
            "calculate_directivity requires both theta and phi, or neither "
            "(for peak directivity).")

    # Convert angles to radians for integration
    theta_rad = np.deg2rad(theta_array)
    phi_rad = np.deg2rad(phi_array)

    if len(theta_rad) < 2 or len(phi_rad) < 2:
        raise ValueError(
            "calculate_directivity needs at least two theta and two phi samples "
            f"to integrate power over solid angle (got {len(theta_rad)} x {len(phi_rad)}). "
            "A single cut does not determine total radiated power.")

    # Measured solid angle, using exactly the quadrature used for the power
    # integral, so the coverage fraction and the power are consistent. This is
    # correct for both coordinate formats: a full sided sphere (theta 0..180,
    # phi 0..360) and a full central sphere (theta -180..180, phi 0..180) both
    # integrate to 4*pi.
    solid_angle_measured = _integrate_solid_angle(
        np.ones((len(theta_rad), len(phi_rad))), theta_rad, phi_rad)
    coverage_fraction = solid_angle_measured / (4 * np.pi)

    if coverage_fraction > 1.0 + 1e-3:
        # e.g. central theta with phi 0..360: every direction is sampled twice
        logger.warning(
            "calculate_directivity: the angular grid covers the sphere %.2f times "
            "(theta %.1f..%.1f, phi %.1f..%.1f); power is being multiplied by that "
            "factor. Use transform_coordinates() or split_dual_sphere() first.",
            coverage_fraction, np.min(theta_array), np.max(theta_array),
            np.min(phi_array), np.max(phi_array))

    # Get field components based on requested component
    if component == 'total':
        e_theta_data = pattern.data.e_theta.values[freq_idx, :, :]
        e_phi_data = pattern.data.e_phi.values[freq_idx, :, :]
        radiation_intensity = np.abs(e_theta_data)**2 + np.abs(e_phi_data)**2
    elif component == 'e_co':
        e_co_data = pattern.data.e_co.values[freq_idx, :, :]
        radiation_intensity = np.abs(e_co_data)**2
    elif component == 'e_cx':
        e_cx_data = pattern.data.e_cx.values[freq_idx, :, :]
        radiation_intensity = np.abs(e_cx_data)**2
    elif component == 'e_theta':
        e_theta_data = pattern.data.e_theta.values[freq_idx, :, :]
        radiation_intensity = np.abs(e_theta_data)**2
    elif component == 'e_phi':
        e_phi_data = pattern.data.e_phi.values[freq_idx, :, :]
        radiation_intensity = np.abs(e_phi_data)**2
    else:
        raise ValueError(f"Unknown component '{component}'. "
                        "Must be 'total', 'e_co', 'e_cx', 'e_theta', or 'e_phi'")

    # Total radiated power always comes from the TOTAL field: partial
    # directivity is the power in one component relative to the power radiated
    # in all of them, so the denominator must not depend on `component`.
    total_intensity = (np.abs(pattern.data.e_theta.values[freq_idx, :, :])**2
                       + np.abs(pattern.data.e_phi.values[freq_idx, :, :])**2)
    measured_power = _integrate_solid_angle(total_intensity, theta_rad, phi_rad)

    # Determine if we need partial sphere method
    use_partial_sphere = (coverage_fraction < partial_sphere_threshold) or (measured_power <= 0)
    
    if use_partial_sphere:
        
        peak_intensity = np.max(total_intensity)

        # Estimate power in unmeasured regions
        if far_sidelobe_level_db is not None:
            # Use far sidelobe assumption - more accurate for antenna patterns
            peak_intensity = np.max(total_intensity)
            far_sidelobe_intensity = peak_intensity * (10 ** (far_sidelobe_level_db / 10))
            
            # Calculate unmeasured solid angle
            total_solid_angle = 4 * np.pi
            unmeasured_solid_angle = max(0, total_solid_angle - solid_angle_measured)
            
            unmeasured_power = far_sidelobe_intensity * unmeasured_solid_angle
            total_power = measured_power + unmeasured_power
        else:
            # Use edge extrapolation method (original approach)
            edge_values = []
            # Theta edges
            if len(theta_rad) > 1:
                edge_values.extend([total_intensity[0, :], total_intensity[-1, :]])
            # Phi edges  
            if len(phi_rad) > 1:
                edge_values.extend([total_intensity[:, 0], total_intensity[:, -1]])
            
            if edge_values:
                edge_power_linear = np.mean([np.mean(edge) for edge in edge_values])
                # Apply additional drop-off for unmeasured regions
                unmeasured_power_linear = edge_power_linear * (10**(edge_extrapolation_db/10))
                
                # Calculate unmeasured solid angle
                total_solid_angle = 4 * np.pi
                unmeasured_solid_angle = max(0, total_solid_angle - solid_angle_measured)
                
                unmeasured_power = unmeasured_power_linear * unmeasured_solid_angle
                total_power = measured_power + unmeasured_power
                
            else:
                total_power = measured_power
        
    else:
        total_power = measured_power
    
    # Avoid division by zero
    if total_power <= 0:
        total_power = 1e-15
    
    # Calculate directivity pattern: D(θ,φ) = 4π * U(θ,φ) / P_total
    directivity_linear = 4 * np.pi * radiation_intensity / total_power
    
    # Convert to dB
    directivity_db = 10 * np.log10(np.maximum(directivity_linear, 1e-15))
    
    if theta is not None:
        # Calculate directivity at specific direction
        theta_val, theta_idx = find_nearest(theta_array, theta)
        phi_val, phi_idx = find_nearest(phi_array, phi)
        
        if isinstance(theta_idx, np.ndarray):
            theta_idx = theta_idx.item()
        if isinstance(phi_idx, np.ndarray):
            phi_idx = phi_idx.item()
        
        return directivity_db[theta_idx, phi_idx]
    
    else:
        # Find peak directivity and its location
        max_idx = np.unravel_index(np.argmax(directivity_db), directivity_db.shape)
        peak_directivity = directivity_db[max_idx]
        peak_theta = theta_array[max_idx[0]]
        peak_phi = phi_array[max_idx[1]]
        
        return float(peak_directivity), float(peak_theta), float(peak_phi)
    
def detect_coordinate_format(pattern) -> str:
    """
    Detect whether pattern data is in 'central' or 'sided' coordinate format.

    Detection is based on whether negative theta values are present:
    - Central format has negative theta (theta spans some range including negatives)
    - Sided format has only non-negative theta (theta >= 0)

    This works correctly for both full-sphere and partial-sphere patterns.

    Args:
        pattern: FarFieldSpherical object (or any object with theta_angles and phi_angles attributes)

    Returns:
        str: 'central' or 'sided'

    Notes:
        - Sided format: theta >= 0, phi 0:360 (spherical convention)
        - Central format: theta includes negatives, phi 0:180 (common for antenna patterns)
    """
    theta_angles = pattern.theta_angles
    theta_min = np.min(theta_angles)

    if theta_min < -0.5:
        return 'central'
    return 'sided'