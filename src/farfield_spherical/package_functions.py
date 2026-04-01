"""
Functions for working across multiple far-field spherical patterns.
"""
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple

from .farfield import FarFieldSpherical
from .polarization import polarization_tp2xy, polarization_xy2tp, polarization_rl2tp

def average_patterns(patterns: List[FarFieldSpherical], weights: Optional[List[float]] = None) -> FarFieldSpherical:
    """
    Create a new far-field pattern by averaging multiple patterns.
    
    This function computes a weighted average of the provided patterns. All patterns
    must have compatible dimensions (same theta, phi, and frequency values).
    
    Args:
        patterns: List of FarFieldSpherical objects to average
        weights: Optional list of weights for each pattern. If None, equal weights are used.
            Weights will be normalized to sum to 1.
            
    Returns:
        FarFieldSpherical: A new far-field pattern containing the weighted average
        
    Raises:
        ValueError: If patterns have incompatible dimensions
        ValueError: If weights are provided but don't match the number of patterns
    """
    if len(patterns) < 1:
        raise ValueError("At least one pattern is required for averaging")
    
    # Get reference dimensions from first pattern
    theta = patterns[0].theta_angles
    phi = patterns[0].phi_angles
    freq = patterns[0].frequencies
    
    # Check that all patterns have the same dimensions
    for i, pattern in enumerate(patterns[1:], 1):
        if not np.array_equal(pattern.theta_angles, theta):
            raise ValueError(f"Pattern {i} has different theta angles than pattern 0")
        if not np.array_equal(pattern.phi_angles, phi):
            raise ValueError(f"Pattern {i} has different phi angles than pattern 0")
        if not np.array_equal(pattern.frequencies, freq):
            raise ValueError(f"Pattern {i} has different frequencies than pattern 0")
    
    # Handle weights
    if weights is None:
        # Equal weights
        weights = np.ones(len(patterns)) / len(patterns)
    else:
        if len(weights) != len(patterns):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of patterns ({len(patterns)})")
        
        # Normalize weights to sum to 1
        weights = np.array(weights) / np.sum(weights)
    
    # Initialize output arrays
    e_theta_avg = np.zeros((len(freq), len(theta), len(phi)), dtype=complex)
    e_phi_avg = np.zeros((len(freq), len(theta), len(phi)), dtype=complex)
    
    # Compute weighted average
    for i, pattern in enumerate(patterns):
        e_theta_avg += weights[i] * pattern.data.e_theta.values
        e_phi_avg += weights[i] * pattern.data.e_phi.values
    
    # Create combined polarization string and metadata
    polarizations = [pattern.polarization for pattern in patterns]
    
    # Create metadata for the averaged pattern
    metadata = {
        'source': 'averaged_pattern',
        'weights': weights.tolist(),
        'source_polarizations': polarizations,
        'operations': []
    }
    
    # Include original pattern metadata if available
    for i, pattern in enumerate(patterns):
        if hasattr(pattern, 'metadata') and pattern.metadata:
            metadata[f'source_pattern_{i}_metadata'] = pattern.metadata
    
    # Create a new pattern with the averaged data
    return FarFieldSpherical(
        theta=theta,
        phi=phi,
        frequency=freq,
        e_theta=e_theta_avg,
        e_phi=e_phi_avg,
        metadata=metadata
    )

def difference_patterns(
    pattern1: FarFieldSpherical, 
    pattern2: FarFieldSpherical
) -> FarFieldSpherical:
    """
    Create a new far-field pattern representing the difference between two patterns.
    
    This function computes the complex ratio pattern1/pattern2 for the co-polarized 
    component and cross-polarized component separately, then converts back to 
    theta/phi components. This is useful for analyzing pattern differences, comparing 
    measured vs simulated patterns, or assessing the impact of pattern processing.
    
    Works directly with co-polarized component of the first pattern.
    
    Args:
        pattern1: First FarFieldSpherical object (typically the original pattern)
        pattern2: Second FarFieldSpherical object (typically the processed pattern)
            
    Returns:
        FarFieldSpherical: A new far-field pattern containing the difference (pattern1/pattern2)
                        with polarization matching pattern1
    """
    # Check that patterns have the same dimensions
    theta1 = pattern1.theta_angles
    phi1 = pattern1.phi_angles
    freq1 = pattern1.frequencies
    
    theta2 = pattern2.theta_angles
    phi2 = pattern2.phi_angles
    freq2 = pattern2.frequencies
    
    # Verify the patterns have compatible dimensions
    if not np.array_equal(theta1, theta2):
        raise ValueError("Patterns have different theta angles")
    if not np.array_equal(phi1, phi2):
        raise ValueError("Patterns have different phi angles")
    if not np.array_equal(freq1, freq2):
        raise ValueError("Patterns have different frequencies")
    
    # Find boresight index (closest to theta=0)
    boresight_idx = np.argmin(np.abs(theta1))
    
    # Ensure both patterns have the same polarization
    pol = pattern1.polarization
    if pattern2.polarization != pol:
        pattern2.change_polarization(pol)
    
    # Get the field components - work directly with co-pol and cross-pol
    e_co1 = pattern1.data.e_co.values
    e_cx1 = pattern1.data.e_cx.values
    
    e_co2 = pattern2.data.e_co.values
    e_cx2 = pattern2.data.e_cx.values
    
    # Initialize arrays for difference pattern
    e_co_diff = np.zeros_like(e_co1, dtype=complex)
    e_cx_diff = np.zeros_like(e_cx1, dtype=complex)
    
    # Compute ratio for each frequency
    for f_idx in range(len(freq1)):
        # Get boresight phase for normalization
        boresight_phase1 = np.angle(e_co1[f_idx, boresight_idx, 0])
        boresight_phase2 = np.angle(e_co2[f_idx, boresight_idx, 0])
        phase_offset = boresight_phase1 - boresight_phase2
        
        # Compute complex ratio with phase normalization
        # This removes global phase offset while preserving relative phase structure
        e_co2_normalized = e_co2[f_idx] * np.exp(1j * phase_offset)
        e_cx2_normalized = e_cx2[f_idx] * np.exp(1j * phase_offset)
        
        # Compute ratio (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            e_co_diff[f_idx] = np.where(
                np.abs(e_co2_normalized) > 1e-30,
                e_co1[f_idx] / e_co2_normalized,
                e_co1[f_idx]
            )
            e_cx_diff[f_idx] = np.where(
                np.abs(e_cx2_normalized) > 1e-30,
                e_cx1[f_idx] / e_cx2_normalized,
                e_cx1[f_idx]
            )
    
    # Convert back to theta/phi components based on polarization
    e_theta_diff = np.zeros_like(e_co_diff, dtype=complex)
    e_phi_diff = np.zeros_like(e_co_diff, dtype=complex)
    
    if pol in ('rhcp', 'rh', 'r', 'lhcp', 'lh', 'l'):
        # For circular: convert from RL back to theta/phi
        for f_idx in range(len(freq1)):
            e_theta_temp, e_phi_temp = polarization_rl2tp(e_co_diff[f_idx], e_cx_diff[f_idx])
            e_theta_diff[f_idx] = e_theta_temp
            e_phi_diff[f_idx] = e_phi_temp
    elif pol in ('x', 'l3x'):
        # For X: e_co = e_x, e_cx = e_y
        for f_idx in range(len(freq1)):
            e_theta_temp, e_phi_temp = polarization_xy2tp(phi1, e_co_diff[f_idx], e_cx_diff[f_idx])
            e_theta_diff[f_idx] = e_theta_temp
            e_phi_diff[f_idx] = e_phi_temp
    elif pol in ('y', 'l3y'):
        # For Y: e_co = e_y, e_cx = e_x
        for f_idx in range(len(freq1)):
            e_theta_temp, e_phi_temp = polarization_xy2tp(phi1, e_cx_diff[f_idx], e_co_diff[f_idx])
            e_theta_diff[f_idx] = e_theta_temp
            e_phi_diff[f_idx] = e_phi_temp
    else:
        # For theta/phi, directly use values
        if pol == 'theta':
            e_theta_diff = e_co_diff
            e_phi_diff = e_cx_diff
        else:  # phi polarization
            e_theta_diff = e_cx_diff
            e_phi_diff = e_co_diff
    
    # Create metadata for the difference pattern
    metadata = {
        'source': 'difference_pattern',
        'pattern1_polarization': pattern1.polarization,
        'pattern2_polarization': pattern2.polarization,
        'difference_method': 'direct_co_cx_ratio',
        'operations': []
    }
    
    # Create a new pattern with the difference data
    result_pattern = FarFieldSpherical(
        theta=theta1,
        phi=phi1,
        frequency=freq1,
        e_theta=e_theta_diff,
        e_phi=e_phi_diff,
        polarization=pattern1.polarization,
        metadata=metadata
    )
    
    return result_pattern


def detect_dual_sphere(pattern: FarFieldSpherical) -> Dict[str, Any]:
    """
    Detect whether a pattern contains dual sphere measurement data.

    A dual sphere pattern has phi spanning 0-360, with the first half
    (phi 0 to <180) representing one measurement sphere and the second half
    (phi 180 to <360) representing a second sphere. Works regardless of
    whether theta is in sided or central format.

    Args:
        pattern: FarFieldSpherical object to check

    Returns:
        dict with keys:
            'is_dual_sphere': bool - True if pattern contains two spheres
            'phi_split_index': int or None - index where phi >= 180 starts
            'sphere1_phi_count': int - number of phi cuts in sphere 1
            'sphere2_phi_count': int - number of phi cuts in sphere 2
            'message': str - human-readable status message
    """
    result = {
        'is_dual_sphere': False,
        'phi_split_index': None,
        'sphere1_phi_count': 0,
        'sphere2_phi_count': 0,
        'message': ''
    }

    if not pattern.has_uniform_theta:
        result['message'] = 'Non-uniform theta grid'
        return result

    phi = pattern.phi_angles

    # Check phi range: must span approximately 0-360
    phi_min, phi_max = phi[0], phi[-1]
    if phi_min > 5 or phi_max < 350:
        result['message'] = f'Phi range {phi_min:.1f}-{phi_max:.1f} does not span 0-360'
        return result

    # Find split index where phi >= 180
    split_idx = int(np.searchsorted(phi, 180.0, side='left'))

    # Exclude phi=360 if present (duplicate of phi=0)
    phi2_end = len(phi)
    if np.isclose(phi[-1], 360.0, atol=0.1):
        phi2_end -= 1

    sphere1_count = split_idx
    sphere2_count = phi2_end - split_idx

    result['phi_split_index'] = split_idx
    result['sphere1_phi_count'] = sphere1_count
    result['sphere2_phi_count'] = sphere2_count

    if sphere1_count == 0 or sphere2_count == 0:
        result['message'] = 'No data on one side of phi=180'
        return result

    # Verify counts are close (allow +/- 1 for edge cases)
    if abs(sphere1_count - sphere2_count) > 1:
        result['message'] = (
            f'Unequal phi halves: {sphere1_count} vs {sphere2_count}'
        )
        return result

    # Verify phi grids align (sphere2 phi - 180 should match sphere1 phi)
    phi1 = phi[:split_idx]
    phi2 = phi[split_idx:phi2_end] - 180.0
    n_compare = min(len(phi1), len(phi2))
    if not np.allclose(phi1[:n_compare], phi2[:n_compare], atol=0.5):
        result['message'] = 'Phi grids do not align between spheres'
        return result

    result['is_dual_sphere'] = True
    result['message'] = f'Dual sphere detected ({sphere1_count} + {sphere2_count} phi cuts)'
    return result


def split_dual_sphere(
    pattern: FarFieldSpherical
) -> Tuple[FarFieldSpherical, FarFieldSpherical]:
    """
    Split a dual-sphere pattern into two separate FarFieldSpherical objects.

    Sphere 1 is extracted from phi 0 to <180.
    Sphere 2 is extracted from phi 180 to <360, with phi remapped to 0-180
    and both e_theta and e_phi negated to account for the unit vector reversal
    at phi+180. Works regardless of theta format (sided or central).

    Args:
        pattern: FarFieldSpherical with phi spanning 0-360

    Returns:
        tuple: (sphere1, sphere2) as FarFieldSpherical objects with phi 0-180

    Raises:
        ValueError: If pattern is not a valid dual sphere
    """
    detection = detect_dual_sphere(pattern)
    if not detection['is_dual_sphere']:
        raise ValueError(f"Pattern is not a dual sphere: {detection['message']}")

    split_idx = detection['phi_split_index']
    phi = pattern.phi_angles
    theta = pattern.theta_angles
    freq = pattern.frequencies

    # Determine sphere 2 end index (exclude phi=360 if present)
    phi2_end = len(phi)
    if np.isclose(phi[-1], 360.0, atol=0.1):
        phi2_end -= 1

    # Sphere 1: phi < 180
    phi1 = phi[:split_idx].copy()
    e_theta1 = pattern.data.e_theta.values[:, :, :split_idx].copy()
    e_phi1 = pattern.data.e_phi.values[:, :, :split_idx].copy()

    sphere1 = FarFieldSpherical(
        theta=theta.copy(),
        phi=phi1,
        frequency=freq.copy(),
        e_theta=e_theta1,
        e_phi=e_phi1,
        polarization=pattern.polarization,
        metadata={
            'source': 'dual_sphere_split',
            'sphere': 1,
            'operations': []
        }
    )

    # Sphere 2: phi >= 180, remap to 0-180, negate fields
    # In spherical coordinates, the physical direction at (θ, φ+180) is the same
    # as (-θ, φ). So remapping phi by -180 also requires flipping the theta axis.
    phi2_raw = phi[split_idx:phi2_end].copy() - 180.0
    e_theta2 = -pattern.data.e_theta.values[:, :, split_idx:phi2_end].copy()
    e_phi2 = -pattern.data.e_phi.values[:, :, split_idx:phi2_end].copy()

    # Flip theta axis: reverse data along theta dimension so that the
    # measurement at (θ, φ+180) maps to (-θ, φ) correctly
    e_theta2 = np.flip(e_theta2, axis=1)
    e_phi2 = np.flip(e_phi2, axis=1)
    # Compute sphere 2's theta: negate and reverse to maintain ascending order
    sphere2_theta = -np.flip(theta)

    # Use sphere1's phi grid if they're close but not exactly equal (float precision)
    # This ensures average_patterns() will work without dimension mismatch
    n_common = min(len(phi1), len(phi2_raw))
    if np.allclose(phi1[:n_common], phi2_raw[:n_common], atol=0.5):
        phi2 = phi1[:n_common].copy()
        e_theta2 = e_theta2[:, :, :n_common]
        e_phi2 = e_phi2[:, :, :n_common]
    else:
        phi2 = phi2_raw

    # Use sphere1's theta grid if they're close (symmetric central grids)
    if np.allclose(sphere2_theta, theta, atol=0.5):
        sphere2_theta = theta.copy()

    sphere2 = FarFieldSpherical(
        theta=sphere2_theta,
        phi=phi2,
        frequency=freq.copy(),
        e_theta=e_theta2,
        e_phi=e_phi2,
        polarization=pattern.polarization,
        metadata={
            'source': 'dual_sphere_split',
            'sphere': 2,
            'operations': []
        }
    )

    return sphere1, sphere2