from typing import Union
from pathlib import Path

from ..farfield import FarFieldSpherical
from pathlib import Path
from swe import SphericalWaveExpansion # pyright: ignore[reportMissingImports]
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
                        from ..polarization import polarization_tp2rl
                        e_r, e_l = polarization_tp2rl(
                            phi_val,
                            e_theta[freq_idx, theta_idx, phi_idx:phi_idx+1],
                            e_phi[freq_idx, theta_idx, phi_idx:phi_idx+1]
                        )
                        comp1 = e_r[0]  # RHCP
                        comp2 = e_l[0]  # LHCP
                        
                    elif polarization_format == 3:
                        # X/Y format
                        from ..polarization import polarization_tp2xy
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


def write_csv(pattern, file_path: Union[str, Path],
              components: str = 'copol',
              include_complex: bool = False) -> None:
    """
    Write an antenna pattern to CSV format.

    Args:
        pattern: FarFieldSpherical object to save
        file_path: Path to save the file to
        components: Which components to include:
            'copol' = co-pol and cross-pol gain/phase (default)
            'spherical' = e_theta and e_phi gain/phase
            'all' = all components (co-pol, cross-pol, e_theta, e_phi)
        include_complex: If True, also include real/imaginary parts

    Raises:
        OSError: If file cannot be written
        ValueError: If components is invalid

    Example:
        >>> pattern.write_csv('output.csv')
        >>> pattern.write_csv('output.csv', components='all', include_complex=True)
    """
    file_path = Path(file_path)

    # Ensure .csv extension
    if file_path.suffix.lower() != '.csv':
        file_path = file_path.with_suffix('.csv')

    if components not in ['copol', 'spherical', 'all']:
        raise ValueError("components must be 'copol', 'spherical', or 'all'")

    # Get pattern data
    theta = pattern.theta_angles
    phi = pattern.phi_angles
    frequencies = pattern.frequencies

    # Build header
    header_parts = ['frequency_hz', 'theta_deg', 'phi_deg']

    if components in ['copol', 'all']:
        header_parts.extend(['e_co_mag_db', 'e_co_phase_deg', 'e_cx_mag_db', 'e_cx_phase_deg'])
        if include_complex:
            header_parts.extend(['e_co_real', 'e_co_imag', 'e_cx_real', 'e_cx_imag'])

    if components in ['spherical', 'all']:
        header_parts.extend(['e_theta_mag_db', 'e_theta_phase_deg', 'e_phi_mag_db', 'e_phi_phase_deg'])
        if include_complex:
            header_parts.extend(['e_theta_real', 'e_theta_imag', 'e_phi_real', 'e_phi_imag'])

    # Get field data
    e_theta = pattern.data.e_theta.values
    e_phi = pattern.data.e_phi.values
    e_co = pattern.data.e_co.values
    e_cx = pattern.data.e_cx.values

    with open(file_path, 'w') as f:
        # Write header
        f.write(','.join(header_parts) + '\n')

        # Write data for each frequency, theta, phi combination
        for freq_idx, freq in enumerate(frequencies):
            for theta_idx, theta_val in enumerate(theta):
                for phi_idx, phi_val in enumerate(phi):
                    # Start with coordinates
                    row_parts = [f'{freq:.6e}', f'{theta_val:.6f}', f'{phi_val:.6f}']

                    if components in ['copol', 'all']:
                        co_val = e_co[freq_idx, theta_idx, phi_idx]
                        cx_val = e_cx[freq_idx, theta_idx, phi_idx]

                        # Magnitude in dB (handle zeros)
                        co_mag_db = 20 * np.log10(max(np.abs(co_val), 1e-30))
                        cx_mag_db = 20 * np.log10(max(np.abs(cx_val), 1e-30))

                        # Phase in degrees
                        co_phase = np.degrees(np.angle(co_val))
                        cx_phase = np.degrees(np.angle(cx_val))

                        row_parts.extend([
                            f'{co_mag_db:.6f}', f'{co_phase:.6f}',
                            f'{cx_mag_db:.6f}', f'{cx_phase:.6f}'
                        ])

                        if include_complex:
                            row_parts.extend([
                                f'{co_val.real:.6e}', f'{co_val.imag:.6e}',
                                f'{cx_val.real:.6e}', f'{cx_val.imag:.6e}'
                            ])

                    if components in ['spherical', 'all']:
                        eth_val = e_theta[freq_idx, theta_idx, phi_idx]
                        eph_val = e_phi[freq_idx, theta_idx, phi_idx]

                        # Magnitude in dB (handle zeros)
                        eth_mag_db = 20 * np.log10(max(np.abs(eth_val), 1e-30))
                        eph_mag_db = 20 * np.log10(max(np.abs(eph_val), 1e-30))

                        # Phase in degrees
                        eth_phase = np.degrees(np.angle(eth_val))
                        eph_phase = np.degrees(np.angle(eph_val))

                        row_parts.extend([
                            f'{eth_mag_db:.6f}', f'{eth_phase:.6f}',
                            f'{eph_mag_db:.6f}', f'{eph_phase:.6f}'
                        ])

                        if include_complex:
                            row_parts.extend([
                                f'{eth_val.real:.6e}', f'{eth_val.imag:.6e}',
                                f'{eph_val.real:.6e}', f'{eph_val.imag:.6e}'
                            ])

                    f.write(','.join(row_parts) + '\n')