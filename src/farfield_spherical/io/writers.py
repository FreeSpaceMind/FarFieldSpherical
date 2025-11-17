from typing import Union
from pathlib import Path

from ..farfield import FarFieldSpherical
from pathlib import Path
from swe import SphericalWaveExpansion
import numpy as np

def write_cut(pattern, file_path: Union[str, Path], polarization_format: int = 1) -> None:
    """
    Write an antenna pattern to GRASP CUT format.
    
    Args:
        pattern: AntennaPattern object to save
        file_path: Path to save the file to
        polarization_format: Output polarization format:
            1 = theta/phi (spherical)
            2 = RHCP/LHCP (circular)
            3 = X/Y (Ludwig-3 linear)
            
    Raises:
        OSError: If file cannot be written
        ValueError: If polarization_format is invalid
    """
    file_path = Path(file_path)
    
    # Ensure .cut extension
    if file_path.suffix.lower() != '.cut':
        file_path = file_path.with_suffix('.cut')
    
    if polarization_format not in [1, 2, 3]:
        raise ValueError("polarization_format must be 1 (theta/phi), 2 (RHCP/LHCP), or 3 (X/Y)")
    
    # Get pattern data
    theta = pattern.theta_angles
    phi = pattern.phi_angles
    frequencies = pattern.frequencies
    e_theta = pattern.data.e_theta.values
    e_phi = pattern.data.e_phi.values
    
    with open(file_path, 'w') as f:
        # Write data for each frequency
        for freq_idx, freq in enumerate(frequencies):
            # Write data for each phi cut
            for phi_idx, phi_val in enumerate(phi):
                # Write text description line for this cut
                f.write(f"{freq/1e6:.3f} MHz, Phi = {phi_val:.1f} deg\n")
                
                # Write cut header: theta_start, theta_step, num_theta, phi, icomp, icut, ncomp
                theta_start = theta[0]
                theta_step = theta[1] - theta[0] if len(theta) > 1 else 1.0
                num_theta = len(theta)
                icut = 1  # Standard polar cut (phi fixed, theta varying)
                ncomp = 2  # Two field components
                
                f.write(f"{theta_start:.2f} {theta_step:.6f} {num_theta} {phi_val:.2f} {polarization_format} {icut} {ncomp}\n")
                
                # Convert field components based on polarization format
                for theta_idx in range(len(theta)):
                    if polarization_format == 1:
                        # Theta/phi format
                        comp1 = e_theta[freq_idx, theta_idx, phi_idx]
                        comp2 = e_phi[freq_idx, theta_idx, phi_idx]
                        
                    elif polarization_format == 2:
                        # RHCP/LHCP format
                        from .polarization import polarization_tp2rl
                        e_r, e_l = polarization_tp2rl(
                            phi_val,
                            e_theta[freq_idx, theta_idx, phi_idx:phi_idx+1],
                            e_phi[freq_idx, theta_idx, phi_idx:phi_idx+1]
                        )
                        comp1 = e_r[0]  # RHCP
                        comp2 = e_l[0]  # LHCP
                        
                    elif polarization_format == 3:
                        # X/Y format
                        from .polarization import polarization_tp2xy
                        e_x, e_y = polarization_tp2xy(
                            phi_val,
                            e_theta[freq_idx, theta_idx, phi_idx:phi_idx+1],
                            e_phi[freq_idx, theta_idx, phi_idx:phi_idx+1]
                        )
                        comp1 = e_x[0]  # X component
                        comp2 = e_y[0]  # Y component
                    
                    # Write complex components
                    f.write(f"{comp1.real:.6e} {comp1.imag:.6e} {comp2.real:.6e} {comp2.imag:.6e}\n")

def write_ffd(pattern, file_path: Union[str, Path]) -> None:
    """
    Write an antenna pattern to HFSS far field data format (.ffd).
    
    Args:
        pattern: AntennaPattern object to save
        file_path: Path to save the file to
        
    Raises:
        OSError: If file cannot be written
    """
    file_path = Path(file_path)
    
    # Ensure .ffd extension
    if file_path.suffix.lower() != '.ffd':
        file_path = file_path.with_suffix('.ffd')
    
    # Get pattern data
    theta = pattern.theta_angles
    phi = pattern.phi_angles
    frequencies = pattern.frequencies
    e_theta = pattern.data.e_theta.values
    e_phi = pattern.data.e_phi.values
    
    with open(file_path, 'w') as f:
        # Write header lines
        f.write(f"{theta[0]} {theta[-1]} {len(theta)}\n")
        f.write(f"{phi[0]} {phi[-1]} {len(phi)}\n")
        f.write(f"Freq {len(frequencies)}\n")
        
        # Write data for each frequency
        for freq_idx, freq in enumerate(frequencies):
            f.write(f"Frequency {freq}\n")
            
            # Write field data for all theta/phi combinations
            # FFD format: theta is outer loop, phi is inner loop (opposite of what I had)
            for theta_idx in range(len(theta)):
                for phi_idx in range(len(phi)):
                    # Convert to HFSS units (multiply by sqrt(60))
                    eth = e_theta[freq_idx, theta_idx, phi_idx] * np.sqrt(60)
                    eph = e_phi[freq_idx, theta_idx, phi_idx] * np.sqrt(60)
                    
                    f.write(f"{eth.real:.6e} {eth.imag:.6e} {eph.real:.6e} {eph.imag:.6e}\n")

def write_ticra_sph(swe: 'SphericalWaveExpansion', file_path: Union[str, Path],
                    program_tag: str = "AntPy", id_string: str = "SWE Export") -> None:
    """
    Write spherical mode coefficients to TICRA .sph format.
    
    Args:
        swe: SphericalWaveExpansion object
        file_path: Output file path
        program_tag: Program tag (ignored, for compatibility)
        id_string: Description string
    """
    # Use the new module's writer
    swe.to_sph_file(str(file_path), description=id_string)