from pathlib import Path
from typing import Union, Optional, Iterable
import re
import logging
import numpy as np
from ..farfield import FarFieldSpherical

logger = logging.getLogger(__name__)

try:
    from swe import SphericalWaveExpansion  # pyright: ignore[reportMissingImports]
    _SWE_AVAILABLE = True
except ImportError:
    _SWE_AVAILABLE = False
    SphericalWaveExpansion = None  # type: ignore[assignment,misc]


def scan_sph_frequencies(file_path: Union[str, Path]) -> list[float]:
    """Return frequencies (Hz) of all blocks in a TICRA .sph file.

    Reads only header lines and does not parse coefficient records.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"SPH file not found: {file_path}")

    frequencies = []
    pattern = re.compile(r"Freq\s*\[GHz\]\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)")
    with open(file_path, "r") as reader:
        for line in reader:
            match = pattern.search(line)
            if match:
                frequencies.append(float(match.group(1)) * 1e9)

    if not frequencies:
        raise ValueError(f"No 'Freq [GHz]:' lines found in {file_path}")
    return frequencies

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
    
    phi = []
    ya_data = []
    yb_data = []

    theta = None
    icomp = None

    theta_start = 0
    theta_increment = 0
    theta_length = 0

    # Main parsing
    header_flag = True
    first_flag = True
    data_counter = 0
    line_data_a = []
    line_data_b = []
    cut_frequencies = []
    
    while line_index < total_lines:
        data_str = lines[line_index]
        line_index += 1
        
        if "MHz" in data_str:
            # Description line for a new cut. GRASP writes the frequency here
            # (and so does write_cut), which is more reliable than assuming the
            # cuts are evenly spaced between frequency_start and frequency_end.
            header_flag = True
            match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*MHz", data_str)
            cut_frequencies.append(float(match.group(1)) * 1e6 if match else None)
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
                    logger.warning("Unexpected ICUT value: %s. Expected 1 (standard polar cut)", icut)
                if ncomp != 2:
                    logger.warning("Unexpected NCOMP value: %s. Expected 2 (two field components)", ncomp)
                    
                first_flag = False
            
            # Preallocate data arrays for this section
            line_data_a = []
            line_data_b = []
            
            data_counter = theta_length
            header_flag = False
        else:
            parts = np.array(data_str.split(), dtype=float)
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
    
    phi_array = np.array(phi, dtype=float)
    unique_phi = np.sort(np.unique(phi_array))
    num_phi = len(unique_phi)
    num_theta = len(theta)

    if len(ya_data) % num_phi != 0:
        raise ValueError(
            f"CUT file holds {len(ya_data)} cuts, which is not a whole number of "
            f"frequency blocks of {num_phi} phi cuts. The file may be truncated.")
    freq_num = len(ya_data) // num_phi

    if freq_num <= 0:
        raise ValueError(f"Invalid frequency count: {freq_num}")

    # Prefer the frequency recorded on each cut's description line; fall back to
    # spreading the caller's range evenly when the file does not carry them.
    frequency = np.linspace(frequency_start, frequency_end, freq_num)
    if len(cut_frequencies) == len(ya_data) and all(f is not None for f in cut_frequencies):
        block_frequencies = np.array(cut_frequencies[::num_phi], dtype=float)
        if len(block_frequencies) == freq_num:
            frequency = block_frequencies

    e_theta = np.zeros((freq_num, num_theta, num_phi), dtype=complex)
    e_phi = np.zeros((freq_num, num_theta, num_phi), dtype=complex)

    # Place each cut at its recorded phi value rather than at its position in
    # the file: GRASP does not guarantee that cuts are written in ascending phi.
    for i in range(len(ya_data)):
        freq_idx = i // num_phi
        phi_idx = int(np.argmin(np.abs(unique_phi - phi_array[i])))

        if freq_idx < freq_num:
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

def read_ffd(file_path: Union[str, Path], frequency_hz: Optional[float] = None):
    """
    Read a far field data file from HFSS.
    
    Args:
        file_path: Path to the FFD file
        frequency_hz: Frequency in Hz, used only when a single-frequency file
            omits its 'Frequency' line
        
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
        # Some single-frequency HFSS exports omit the "Frequency <Hz>" line and
        # start the data immediately. Only consume the line when it really is
        # one, so the first data row is not parsed as a frequency.
        has_frequency_line = (len(freq_line) >= 2
                              and freq_line[0].lower().startswith('freq'))
        if has_frequency_line:
            frequency = float(freq_line[1])
        elif num_frequencies == 1 and frequency_hz is not None:
            # Some single-frequency HFSS exports omit the "Frequency <Hz>" line
            # and start the data immediately; the caller has supplied it.
            frequency = float(frequency_hz)
            index -= 1   # the index += 1 below re-reads this line as data
        elif not has_frequency_line:
            raise ValueError(
                f"No 'Frequency <Hz>' line before the data block at line "
                f"{index + 1} of {file_path.name}. Some exports omit it; pass "
                f"frequency_hz=<value> to read_ffd to supply it.")
        else:
            raise ValueError(f"Invalid frequency line: {lines[index].strip()}")
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

    # Reshape into 3D (freq, theta, phi) format. The file stores theta
    # varying slowest and phi fastest, so a plain reshape is enough; the old
    # triple loop also silently left zeros when a block was short.
    expected = len(theta) * len(phi)
    for freq_idx, block in enumerate(e_theta_list):
        if len(block) != expected:
            raise ValueError(
                f"Frequency block {freq_idx + 1} holds {len(block)} samples but the "
                f"header declares {len(theta)} x {len(phi)} = {expected}.")

    e_theta_final = np.asarray(e_theta_list, dtype=complex).reshape(
        len(frequency_np), len(theta), len(phi))
    e_phi_final = np.asarray(e_phi_list, dtype=complex).reshape(
        len(frequency_np), len(theta), len(phi))

    # Create FarFieldSpherical - polarization will be auto-detected
    return FarFieldSpherical(
        theta=theta,
        phi=phi,
        frequency=frequency_np,
        e_theta=e_theta_final,
        e_phi=e_phi_final
    )

def read_ticra_sph(file_path: Union[str, Path],
                   frequencies: Optional[Iterable[float]] = None) -> 'SphericalWaveExpansion':
    """
    Read spherical mode coefficients from TICRA .sph format.

    Args:
        file_path: Path to .sph file
        frequency: Frequency in Hz (for metadata, not read from file)

    Returns:
        SphericalWaveExpansion object
    """

    # Use the new module's reader
    if not _SWE_AVAILABLE:
        raise ImportError(
            "The 'swe' package is required to read TICRA .sph files. "
            "Install it with: pip install farfield-spherical[swe]")
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"SPH file not found: {file_path}")

    swe = SphericalWaveExpansion.from_sph_file(str(file_path), frequencies=frequencies)

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

            # Convert dB + phase to complex linear:
            #     E = 10^(mag_db/20) * exp(-j * phase_deg * pi/180)
            #
            # The phase is negated deliberately. ATAMS uses the exp(+j w t)
            # time convention while this package (like the CUT and FFD readers)
            # uses exp(-j w t), and the two differ by a complex conjugate. Do
            # not "simplify" this sign away: it would mirror every
            # phase-sensitive result (translate, find_phase_center, apply_mars)
            # in z for ATAMS data relative to data from the other readers.
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
