from pathlib import Path
from typing import Union, Optional
import numpy as np
from ..farfield import FarFieldSpherical
from swe import SphericalWaveExpansion

def read_cut(file_path: Union[str, Path], frequency_start: float, frequency_end: float):
    """
    Read an antenna CUT file and store it in an FarFieldSpherical.
    
    Optimized version with faster file reading and data processing.
    
    Args:
        file_path: Path to the CUT file
        frequency_start: Frequency of first pattern in Hz
        frequency_end: Frequency of last pattern in Hz
        
    Returns:
        FarFieldSpherical: The imported antenna pattern
    """
    from ..polarization import polarization_rl2tp, polarization_xy2tp
    
    # Validate inputs
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CUT file not found: {file_path}")
    
    if frequency_start <= 0 or frequency_end <= 0:
        raise ValueError("Frequencies must be positive")
    if frequency_start > frequency_end:
        raise ValueError("frequency_start must be less than or equal to frequency_end")
    
    # Read entire file at once for faster processing
    with open(file_path, "r") as reader:
        lines = reader.readlines()
    
    total_lines = len(lines)
    line_index = 0
    
    # Use lists with pre-estimated size for better performance
    estimated_phi_count = 36  # Typical number of phi cuts
    estimated_freq_count = 5  # Typical number of frequencies
    estimated_theta_count = 181  # Typical number of theta points
    
    phi = []
    ya_data = []
    yb_data = []
    
    # Use NumPy arrays to store theta array once
    theta = None
    icomp = None
    
    # Preallocate header info
    theta_start = 0
    theta_increment = 0
    theta_length = 0
    
    # Scan file structure to determine pattern dimensions
    # First pass to determine dimensions
    first_pass = True
    phi_values = set()
    header_count = 0
    
    if first_pass:
        while line_index < min(1000, total_lines):  # Just scan the first part of the file
            if "MHz" in lines[line_index]:
                line_index += 1
                if line_index >= total_lines:
                    break
                    
                # This should be a header line (numeric data)
                header_parts = lines[line_index].strip().split()
                if len(header_parts) >= 7:  # Updated from 5 to 7
                    header_count += 1
                    phi_values.add(float(header_parts[3]))
                
            line_index += 1
        
        # Reset line index for main parsing
        line_index = 0
        first_pass = False
        
        # Estimate dimensions more accurately
        estimated_phi_count = len(phi_values)
        
        if header_count > 0:
            estimated_freq_count = header_count // estimated_phi_count
    
    # Main parsing - optimized for speed
    header_flag = True
    first_flag = True
    data_counter = 0
    line_data_a = []
    line_data_b = []
    
    while line_index < total_lines:
        data_str = lines[line_index]
        line_index += 1
        
        if "MHz" in data_str:
            # This is a description line for a new cut
            header_flag = True
            continue
            
        if header_flag:
            # Parse header efficiently
            header_parts = data_str.strip().split()
            if len(header_parts) < 7:  # Updated from 5 to 7
                continue
                
            theta_length = int(header_parts[2])
            phi.append(float(header_parts[3]))
            
            if first_flag:
                theta_start = float(header_parts[0])
                theta_increment = float(header_parts[1])
                theta = np.linspace(
                    theta_start,
                    theta_start + (theta_length - 1) * theta_increment,
                    theta_length
                )
                icomp = int(header_parts[4])
                icut = int(header_parts[5])   # ICUT parameter (should be 1)
                ncomp = int(header_parts[6])  # NCOMP parameter (should be 2)
                
                if icomp not in [1, 2, 3]:
                    raise ValueError(f"Invalid polarization format (ICOMP): {icomp}")
                if icut != 1:
                    print(f"Unexpected ICUT value: {icut}. Expected 1 (standard polar cut)")
                if ncomp != 2:
                    print(f"Unexpected NCOMP value: {ncomp}. Expected 2 (two field components)")
                    
                first_flag = False
            
            # Preallocate data arrays for this section
            line_data_a = []
            line_data_b = []
            
            data_counter = theta_length
            header_flag = False
        else:
            parts = np.fromstring(data_str, dtype=float, sep=" ")
            if len(parts) >= 4:
                line_data_a.append(complex(parts[0], parts[1]))
                line_data_b.append(complex(parts[2], parts[3]))
                data_counter -= 1
                
                if data_counter == 0:
                    header_flag = True
                    ya_data.append(line_data_a)
                    yb_data.append(line_data_b)

    # Consistency checks
    if len(ya_data) == 0 or len(yb_data) == 0:
        raise ValueError("No valid data found in CUT file")
    
    # Make frequency vector more accurately
    phi_array = np.array(phi)
    unique_phi = np.sort(np.unique(phi_array))
    freq_num = len(ya_data) // len(unique_phi)
    
    if freq_num <= 0:
        raise ValueError(f"Invalid frequency count: {freq_num}")
    
    frequency = np.linspace(frequency_start, frequency_end, freq_num)
    
    # Convert to numpy arrays efficiently - specify dtype for better performance
    ya_np = np.array(ya_data, dtype=complex)
    yb_np = np.array(yb_data, dtype=complex)
    
    # Determine the correct shape
    num_theta = len(theta)
    num_phi = len(unique_phi)
    
    # Use optimized reshape approach with direct indexing
    e_theta = np.zeros((freq_num, num_theta, num_phi), dtype=complex)
    e_phi = np.zeros((freq_num, num_theta, num_phi), dtype=complex)
    
    # Custom reshape logic for the specific data structure
    for i in range(len(ya_data)):
        freq_idx = i // num_phi
        phi_idx = i % num_phi
        
        if freq_idx < freq_num and phi_idx < num_phi:
            e_theta[freq_idx, :, phi_idx] = ya_data[i]
            e_phi[freq_idx, :, phi_idx] = yb_data[i]
    
    # Convert polarizations based on icomp
    if icomp == 1:
        # Polarization is theta, phi - already in right form
        pass
    elif icomp == 2:
        # Polarization is right and left - vectorized conversion
        for phi_idx, phi_val in enumerate(unique_phi):
            theta_slice, phi_slice = polarization_rl2tp(
                phi_val, 
                e_theta[:, :, phi_idx], 
                e_phi[:, :, phi_idx]
            )
            e_theta[:, :, phi_idx] = theta_slice
            e_phi[:, :, phi_idx] = phi_slice
    elif icomp == 3:
        # Polarization is linear co and cross (x and y) - vectorized conversion
        for phi_idx, phi_val in enumerate(unique_phi):
            theta_slice, phi_slice = polarization_xy2tp(
                phi_val, 
                e_theta[:, :, phi_idx], 
                e_phi[:, :, phi_idx]
            )
            e_theta[:, :, phi_idx] = theta_slice
            e_phi[:, :, phi_idx] = phi_slice
            
    # Create FarFieldSpherical with results
    return FarFieldSpherical(
        theta=theta,
        phi=unique_phi,
        frequency=frequency,
        e_theta=e_theta,
        e_phi=e_phi
    )

def read_ffd(file_path: Union[str, Path]):
    """
    Read a far field data file from HFSS.
    
    Args:
        file_path: Path to the FFD file
        
    Returns:
        FarFieldSpherical: The imported antenna pattern
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a valid FFD file
    """
    
    # Validate input
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FFD file not found: {file_path}")
    
    # Read a far field data file, format ffd
    with open(file_path, "r") as file_handle:
        lines = file_handle.readlines()

    # Read theta, phi, and frequency information
    if len(lines) < 3:
        raise ValueError("FFD file is too short")
        
    theta_info = lines[0].strip().split()
    phi_info = lines[1].strip().split()
    freq_info = lines[2].strip().split()

    if len(theta_info) < 3 or len(phi_info) < 3 or len(freq_info) < 2:
        raise ValueError("Invalid FFD file header format")

    theta_start, theta_stop, theta_points = map(float, theta_info[:3])
    theta_points = np.round(theta_points).astype(int)
    phi_start, phi_stop, phi_points = map(float, phi_info[:3])
    phi_points = np.round(phi_points).astype(int)
    num_frequencies = int(freq_info[1])

    theta = np.linspace(theta_start, theta_stop, theta_points)
    phi = np.linspace(phi_start, phi_stop, phi_points)

    # Initialize storage lists
    frequency_list = []
    e_theta_list = []
    e_phi_list = []

    # Read file
    index = 3
    for freq_idx in range(num_frequencies):
        if index >= len(lines):
            raise ValueError(f"Unexpected end of file at frequency {freq_idx+1}")
            
        freq_line = lines[index].strip().split()
        if len(freq_line) < 2:
            raise ValueError(f"Invalid frequency line: {lines[index].strip()}")
            
        frequency = float(freq_line[1])
        e_theta = []
        e_phi = []

        index += 1
        for _ in range(int(theta_points) * int(phi_points)):
            if index >= len(lines):
                raise ValueError(f"Unexpected end of file at frequency {freq_idx+1}")
                
            radiation_line = list(map(float, lines[index].strip().split()))
            if len(radiation_line) < 4:
                raise ValueError(f"Invalid radiation line: {lines[index].strip()}")
                
            e_th = radiation_line[0] + 1j * radiation_line[1]
            e_ph = radiation_line[2] + 1j * radiation_line[3]
            
            # Convert from HFSS units to standard field units
            e_theta.append(e_th / np.sqrt(60))
            e_phi.append(e_ph / np.sqrt(60))
            index += 1

        # Append into storage lists
        frequency_list.append(frequency)
        e_theta_list.append(e_theta)
        e_phi_list.append(e_phi)

    # Consistency checks
    if len(frequency_list) == 0:
        raise ValueError("No frequency data found in FFD file")
        
    # Convert to numpy
    frequency_np = np.array(frequency_list)
    e_theta_np = np.array(e_theta_list)
    e_phi_np = np.array(e_phi_list)

    # Reshape into 3D (freq, theta, phi) format
    e_theta_final = np.zeros((len(frequency_np), len(theta), len(phi)), dtype=complex)
    e_phi_final = np.zeros((len(frequency_np), len(theta), len(phi)), dtype=complex)
    
    
    for freq_idx in range(len(frequency_np)):
        # Process each theta/phi combination for this frequency
        for theta_idx in range(len(theta)):
            for phi_idx in range(len(phi)):
                # Reversed indexing formula
                data_idx = theta_idx * len(phi) + phi_idx
                
                if data_idx < len(e_theta_np[freq_idx]):
                    e_theta_final[freq_idx, theta_idx, phi_idx] = e_theta_np[freq_idx][data_idx]
                    e_phi_final[freq_idx, theta_idx, phi_idx] = e_phi_np[freq_idx][data_idx]

    # Create FarFieldSpherical - polarization will be auto-detected
    return FarFieldSpherical(
        theta=theta,
        phi=phi,
        frequency=frequency_np,
        e_theta=e_theta_final,
        e_phi=e_phi_final
    )

def read_ticra_sph(file_path: Union[str, Path]) -> 'SphericalWaveExpansion':
    """
    Read spherical mode coefficients from TICRA .sph format.

    Args:
        file_path: Path to .sph file
        frequency: Frequency in Hz (for metadata, not read from file)

    Returns:
        SphericalWaveExpansion object
    """

    # Use the new module's reader
    swe = SphericalWaveExpansion.from_sph_file(str(file_path))

    return swe


def read_atams(file_path: Union[str, Path], interpolate: bool = False,
               theta: Optional[np.ndarray] = None) -> FarFieldSpherical:
    """
    Read an ATAMS antenna measurement file.

    The ATAMS format stores antenna test and measurement data with:
    - Multiple frequencies per spatial position
    - Per-phi theta grids (actual measured positions)
    - Theta-pol and Phi-pol amplitude (dB) and phase (degrees)

    File structure:
    - Row 1: "Frequency" followed by frequency values in GHz
    - Row 2: Fixed axis name and value (e.g., "Elevation  0.00")
    - Row 3: "Head" followed by head position values (phi angles in degrees)
    - Row 4: "Azimuth" followed by nominal azimuth values (stored in metadata)
    - Row 5+: Data blocks (5 rows each)

    Each data block contains:
    - Location: az_actual, el_actual, head_actual (actual measurement position)
    - Theta-pol (mag): Amplitude values in dB for each frequency
    - Theta-pol (phase): Phase values in degrees for each frequency
    - Phi-pol (mag): Amplitude values in dB for each frequency
    - Phi-pol (phase): Phase values in degrees for each frequency

    Args:
        file_path: Path to the .atams file
        interpolate: If True, interpolate onto a uniform theta grid before returning.
                     Uses the nominal header azimuth values as the target grid unless
                     theta is specified.
        theta: Optional explicit uniform theta array for interpolation.
               Implies interpolate=True.

    Returns:
        FarFieldSpherical: Per-phi theta grid if interpolate=False,
                           uniform theta grid if interpolate=True.

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is invalid
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ATAMS file not found: {file_path}")

    # Read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 5:
        raise ValueError("ATAMS file too short - expected at least 5 header lines")

    # Parse header
    # Row 1: Frequency values (GHz)
    freq_parts = lines[0].strip().split('\t')
    if freq_parts[0] != 'Frequency':
        raise ValueError(f"Expected 'Frequency' in first row, got '{freq_parts[0]}'")
    frequencies_ghz = [float(x) for x in freq_parts[1:] if x.strip()]
    frequencies_hz = np.array(frequencies_ghz) * 1e9
    n_freq = len(frequencies_hz)

    # Row 2: Fixed axis (e.g., "Elevation  0.00")
    fixed_parts = lines[1].strip().split('\t')
    fixed_axis_name = fixed_parts[0]
    fixed_axis_value = float(fixed_parts[1]) if len(fixed_parts) > 1 and fixed_parts[1].strip() else 0.0

    # Row 3: Head positions (phi angles)
    head_parts = lines[2].strip().split('\t')
    if head_parts[0] != 'Head':
        raise ValueError(f"Expected 'Head' in third row, got '{head_parts[0]}'")
    phi_array = np.array([float(x) for x in head_parts[1:] if x.strip()])
    n_phi = len(phi_array)

    # Row 4: Nominal azimuth values (theta reference)
    az_parts = lines[3].strip().split('\t')
    if az_parts[0] != 'Azimuth':
        raise ValueError(f"Expected 'Azimuth' in fourth row, got '{az_parts[0]}'")
    nominal_theta = np.array([float(x) for x in az_parts[1:] if x.strip()])

    # Parse data blocks starting from row 5
    # Each block is 5 lines:
    # 1. Location: az_actual, el_actual, head_actual
    # 2. Theta-pol (mag): values for each frequency
    # 3. Theta-pol (phase): values for each frequency
    # 4. Phi-pol (mag): values for each frequency
    # 5. Phi-pol (phase): values for each frequency

    # First pass: count blocks and organize by head value
    data_start = 4
    blocks_by_head = {phi: [] for phi in phi_array}

    line_idx = data_start
    while line_idx + 4 < len(lines):
        # Parse Location line
        loc_parts = lines[line_idx].strip().split('\t')
        if loc_parts[0] != 'Location':
            line_idx += 1
            continue

        az_actual = float(loc_parts[1])
        el_actual = float(loc_parts[2]) if len(loc_parts) > 2 else 0.0
        head_actual = float(loc_parts[3]) if len(loc_parts) > 3 else 0.0

        # Parse field data
        theta_mag_parts = lines[line_idx + 1].strip().split('\t')
        theta_phase_parts = lines[line_idx + 2].strip().split('\t')
        phi_mag_parts = lines[line_idx + 3].strip().split('\t')
        phi_phase_parts = lines[line_idx + 4].strip().split('\t')

        # Extract values (skip first element which is the label)
        theta_mag_db = np.array([float(x) for x in theta_mag_parts[1:n_freq+1]])
        theta_phase_deg = np.array([float(x) for x in theta_phase_parts[1:n_freq+1]])
        phi_mag_db = np.array([float(x) for x in phi_mag_parts[1:n_freq+1]])
        phi_phase_deg = np.array([float(x) for x in phi_phase_parts[1:n_freq+1]])

        # Find matching head in phi_array (with tolerance)
        head_idx = np.argmin(np.abs(phi_array - head_actual))

        blocks_by_head[phi_array[head_idx]].append({
            'az_actual': az_actual,
            'theta_mag_db': theta_mag_db,
            'theta_phase_deg': theta_phase_deg,
            'phi_mag_db': phi_mag_db,
            'phi_phase_deg': phi_phase_deg
        })

        line_idx += 5

    # Determine n_theta (should be same for all heads)
    n_theta_per_head = [len(blocks_by_head[phi]) for phi in phi_array]
    if len(set(n_theta_per_head)) != 1:
        raise ValueError(f"Different number of theta points per head: {n_theta_per_head}")
    n_theta = n_theta_per_head[0]

    # Build 2D theta grid and field arrays
    theta_grid = np.zeros((n_theta, n_phi))
    e_theta = np.zeros((n_freq, n_theta, n_phi), dtype=np.complex64)
    e_phi = np.zeros((n_freq, n_theta, n_phi), dtype=np.complex64)

    for phi_idx, phi_val in enumerate(phi_array):
        blocks = blocks_by_head[phi_val]
        # Sort by azimuth to ensure consistent ordering
        blocks.sort(key=lambda b: b['az_actual'])

        for theta_idx, block in enumerate(blocks):
            theta_grid[theta_idx, phi_idx] = block['az_actual']

            # Convert dB + phase to complex linear
            # E = 10^(mag_db/20) * exp(-j * phase_deg * pi/180)
            # Conjugate phase to match expected convention
            e_theta_linear = 10 ** (block['theta_mag_db'] / 20) * np.exp(
                -1j * np.deg2rad(block['theta_phase_deg'])
            )
            e_phi_linear = 10 ** (block['phi_mag_db'] / 20) * np.exp(
                -1j * np.deg2rad(block['phi_phase_deg'])
            )

            e_theta[:, theta_idx, phi_idx] = e_theta_linear
            e_phi[:, theta_idx, phi_idx] = e_phi_linear

    # Create metadata
    metadata = {
        'source_format': 'atams',
        'source_file': str(file_path),
        'fixed_axis': {'name': fixed_axis_name, 'value': fixed_axis_value},
        'nominal_theta': nominal_theta.tolist(),
        'operations': []
    }

    # Create FarFieldSpherical with 2D theta grid (non-uniform mode)
    pattern = FarFieldSpherical(
        theta=theta_grid,
        phi=phi_array,
        frequency=frequencies_hz,
        e_theta=e_theta,
        e_phi=e_phi,
        metadata=metadata
    )

    # Apply optional interpolation
    if theta is not None:
        interpolate = True

    if interpolate:
        target = theta if theta is not None else nominal_theta
        pattern = pattern.to_uniform_theta(target)

    return pattern