from typing import Union, Optional
from pathlib import Path
import numpy as np
from swe import SphericalWaveExpansion
from ..farfield import FarFieldSpherical
import logging

# Configure logging
logger = logging.getLogger(__name__)

def create_pattern_from_swe(swe: 'SphericalWaveExpansion',
                           theta_angles: Optional[np.ndarray] = None,
                           phi_angles: Optional[np.ndarray] = None) -> FarFieldSpherical:
    """
    Create AntennaPattern from SphericalWaveExpansion object.
    
    Args:
        swe: SphericalWaveExpansion object
        theta_angles: Theta angles in degrees (default: 0 to 180, 1°)
        phi_angles: Phi angles in degrees (default: 0 to 360, 5°)
        
    Returns:
        AntennaPattern object
    """
    # Default angles - SIDED convention [0°, 180°]
    if theta_angles is None:
        theta_angles = np.linspace(0, 180, 181)
    if phi_angles is None:
        phi_angles = np.arange(0, 361, 5.0)
    
    # Convert to radians
    theta_rad = np.radians(theta_angles)
    phi_rad = np.radians(phi_angles)
    
    # Create meshgrid
    THETA, PHI = np.meshgrid(theta_rad, phi_rad, indexing='ij')
    
    # Calculate far field
    E_theta, E_phi = swe.far_field(THETA.ravel(), PHI.ravel())
    E_theta = E_theta.reshape(THETA.shape)
    E_phi = E_phi.reshape(PHI.shape)
    
    # Create pattern (single frequency)
    frequencies = np.array([swe.frequency])
    e_theta = E_theta[np.newaxis, :, :]  # Add frequency dimension
    e_phi = E_phi[np.newaxis, :, :]
    
    pattern = FarFieldSpherical(
        theta=theta_angles,
        phi=phi_angles,
        frequency=frequencies,
        e_theta=e_theta,
        e_phi=e_phi,
        polarization='theta'
    )
    
    # Attach SWE object
    pattern.swe = {swe.frequency: swe}
    
    logger.info(f"Pattern created from SWE at f={swe.frequency/1e9:.3f} GHz")
    return pattern