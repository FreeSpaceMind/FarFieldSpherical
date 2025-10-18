"""
Functions for working across multiple far-field spherical patterns.
"""
import numpy as np
from typing import List, Optional, Union, Dict, Any

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
        pattern2 = pattern2.change_polarization(pol)
    
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