from typing import Union, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import json
from ..farfield import FarFieldSpherical
from swe import SphericalWaveExpansion

def save_pattern_npz(pattern, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save an antenna pattern to NPZ format for efficient loading.

    Args:
        pattern: FarFieldSpherical object to save
        file_path: Path to save the file to
        metadata: Optional metadata to include

    Raises:
        OSError: If file cannot be written
    """
    file_path = Path(file_path)

    # Ensure .npz extension
    if file_path.suffix.lower() != '.npz':
        file_path = file_path.with_suffix('.npz')

    # Extract data from pattern
    phi = pattern.phi_angles
    frequency = pattern.frequencies
    e_theta = pattern.data.e_theta.values
    e_phi = pattern.data.e_phi.values

    # Create metadata dictionary
    meta_dict = {
        'polarization': pattern.polarization,
        'version': '1.1',  # Updated version for non-uniform theta support
        'format': 'AntPy Pattern NPZ'
    }

    # Handle non-uniform theta grids
    if pattern.has_uniform_theta:
        theta = pattern.data.theta.values
        meta_dict['non_uniform_theta'] = False
    else:
        # For non-uniform, store the 2D theta grid
        theta = pattern.theta_grid  # 2D array (n_theta, n_phi)
        meta_dict['non_uniform_theta'] = True

    # Add additional metadata if provided
    if metadata:
        meta_dict.update(metadata)

    # Convert metadata to JSON string
    meta_json = json.dumps(meta_dict)

    # Prepare save dictionary
    save_dict = {
        'theta': theta,  # 1D for uniform, 2D for non-uniform
        'phi': phi,
        'frequency': frequency,
        'e_theta': e_theta,
        'e_phi': e_phi,
        'metadata': meta_json
    }
    
    # Check if pattern has spherical wave expansion data
    if hasattr(pattern, 'swe') and pattern.swe:
        # Save each frequency's SWE data
        swe_frequencies = list(pattern.swe.keys())
        save_dict['swe_frequencies'] = np.array(swe_frequencies)
        
        for i, freq in enumerate(swe_frequencies):
            swe_obj = pattern.swe[freq]
            prefix = f'swe_{i}_'
            
            # Convert coefficient dicts to arrays for storage
            modes = sorted(set(swe_obj.Q1_coeffs.keys()) | set(swe_obj.Q2_coeffs.keys()))
            
            Q1_array = np.array([swe_obj.Q1_coeffs.get(mode, 0.0) for mode in modes])
            Q2_array = np.array([swe_obj.Q2_coeffs.get(mode, 0.0) for mode in modes])
            modes_array = np.array(modes)
            
            save_dict[f'{prefix}Q1_coeffs'] = Q1_array
            save_dict[f'{prefix}Q2_coeffs'] = Q2_array
            save_dict[f'{prefix}modes'] = modes_array
            
            # Save metadata
            swe_meta = {
                'NMAX': int(swe_obj.NMAX),
                'MMAX': int(swe_obj.MMAX),
                'frequency': float(swe_obj.frequency) if swe_obj.frequency else None,
            }
            
            save_dict[f'{prefix}metadata'] = json.dumps(swe_meta)
    
    # Save data to NPZ file
    np.savez_compressed(file_path, **save_dict)

def load_pattern_npz(file_path: Union[str, Path]) -> Tuple:
    """
    Load an antenna pattern from NPZ format.

    Args:
        file_path: Path to the NPZ file

    Returns:
        Tuple containing (pattern, metadata)

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is invalid
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {file_path}")

    # Load data from NPZ file
    with np.load(file_path, allow_pickle=False) as data:
        theta = data['theta']
        phi = data['phi']
        frequency = data['frequency']
        e_theta = data['e_theta']
        e_phi = data['e_phi']

        # Load metadata
        metadata_json = str(data['metadata'])
        metadata = json.loads(metadata_json)

        polarization = metadata.get('polarization')

        # Check for non-uniform theta (2D theta array or metadata flag)
        # The theta array dimensionality determines uniform vs non-uniform
        # 2D theta -> non-uniform, 1D theta -> uniform
        # (metadata flag 'non_uniform_theta' is for documentation only)

        # Create FarFieldSpherical object
        # The constructor handles both 1D and 2D theta arrays automatically
        pattern = FarFieldSpherical(
            theta=theta,
            phi=phi,
            frequency=frequency,
            e_theta=e_theta,
            e_phi=e_phi,
            polarization=polarization
        )
        
        # Load SWE data if present
        if 'swe_frequencies' in data:
            pattern.swe = {}
            swe_frequencies = data['swe_frequencies']
            
            for i, freq in enumerate(swe_frequencies):
                prefix = f'swe_{i}_'
                
                # Load arrays
                Q1_array = data[f'{prefix}Q1_coeffs']
                Q2_array = data[f'{prefix}Q2_coeffs']
                modes_array = data[f'{prefix}modes']
                
                # Reconstruct coefficient dicts
                Q1_coeffs = {tuple(mode): Q1_array[j] for j, mode in enumerate(modes_array)}
                Q2_coeffs = {tuple(mode): Q2_array[j] for j, mode in enumerate(modes_array)}
                
                # Load metadata
                swe_meta = json.loads(str(data[f'{prefix}metadata']))
                
                # Create SWE object
                swe_obj = SphericalWaveExpansion(
                    Q1_coeffs=Q1_coeffs,
                    Q2_coeffs=Q2_coeffs,
                    frequency=swe_meta['frequency'],
                    NMAX=swe_meta['NMAX'],
                    MMAX=swe_meta['MMAX']
                )
                
                pattern.swe[float(freq)] = swe_obj
                
        return pattern, metadata