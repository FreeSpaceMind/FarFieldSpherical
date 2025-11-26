"""
Core class for far-field spherical antenna pattern representation and manipulation.
"""
import numpy as np
import xarray as xr
from typing import Optional, Union, Tuple, Dict, Any, Set, Generator
from pathlib import Path
from contextlib import contextmanager

from .utilities import find_nearest
from .polarization import (
    polarization_tp2xy, polarization_tp2rl
)
from .analysis import (
    calculate_phase_center, get_axial_ratio
    )
from .farfield_operations import FarFieldOperationsMixin
from swe import SphericalWaveExpansion # pyright: ignore[reportMissingImports]

class FarFieldSpherical(FarFieldOperationsMixin):
    """
    A class to represent antenna far field patterns in spherical coordinates.
    
    This class was formerly known as AntennaPattern. It encapsulates far-field 
    pattern data and provides methods for manipulation, analysis and conversion 
    between different formats and coordinate systems.
    
    Attributes:
        data (xarray.Dataset): The core dataset containing all pattern information
            with dimensions (frequency, theta, phi) and data variables:
            - e_theta: Complex theta polarization component
            - e_phi: Complex phi polarization component
            - e_co: Co-polarized component (determined by polarization attribute)
            - e_cx: Cross-polarized component (determined by polarization attribute)
        polarization (str): The polarization type ('rhcp', 'lhcp', 'x', 'y', 'theta', 'phi')
        metadata (Dict[str, Any]): Optional metadata for the pattern including operations history
    """
    
    VALID_POLARIZATIONS: Set[str] = {
        'rhcp', 'rh', 'r',            # Right-hand circular
        'lhcp', 'lh', 'l',            # Left-hand circular
        'x', 'l3x',                   # Linear X (Ludwig's 3rd)
        'y', 'l3y',                   # Linear Y (Ludwig's 3rd)
        'theta',                      # Spherical theta
        'phi'                         # Spherical phi
    }
    
    def __init__(self, 
                 theta: np.ndarray, 
                 phi: np.ndarray, 
                 frequency: np.ndarray,
                 e_theta: np.ndarray, 
                 e_phi: np.ndarray,
                 polarization: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a FarFieldSpherical with the given parameters.
        
        Args:
            theta: Array of theta angles in degrees
            phi: Array of phi angles in degrees
            frequency: Array of frequencies in Hz
            e_theta: Complex array of e_theta values [freq, theta, phi]
            e_phi: Complex array of e_phi values [freq, theta, phi]
            polarization: Optional polarization type. If None, determined automatically.
            metadata: Optional metadata dictionary
            
        Raises:
            ValueError: If arrays have incompatible dimensions
            ValueError: If polarization is invalid
        """
        # Convert arrays to efficient dtypes for faster processing
        e_theta = np.asarray(e_theta, dtype=np.complex64)
        e_phi = np.asarray(e_phi, dtype=np.complex64)

        # Validate array dimensions
        expected_shape = (len(frequency), len(theta), len(phi))
        if e_theta.shape != expected_shape:
            raise ValueError(f"e_theta shape mismatch: expected {expected_shape}, got {e_theta.shape}")
        if e_phi.shape != expected_shape:
            raise ValueError(f"e_phi shape mismatch: expected {expected_shape}, got {e_phi.shape}")
        
        # Create core dataset
        self.data = xr.Dataset(
            data_vars={
                'e_theta': (('frequency', 'theta', 'phi'), e_theta),
                'e_phi': (('frequency', 'theta', 'phi'), e_phi),
            },
            coords={
                'theta': theta,
                'phi': phi,
                'frequency': frequency,
            }
        )
        
        # Initialize metadata if provided
        self.metadata = metadata.copy() if metadata is not None else {'operations': []}

        # Assign polarization and compute derived components
        self.assign_polarization(polarization)
        
        # Initialize cache
        self._cache: Dict[str, Any] = {}
    
    @property
    def frequencies(self) -> np.ndarray:
        """Get frequencies in Hz."""
        return self.data.frequency.values
    
    @property
    def theta_angles(self) -> np.ndarray:
        """Get theta angles in degrees."""
        return self.data.theta.values
    
    @property
    def phi_angles(self) -> np.ndarray:
        """Get phi angles in degrees."""
        return self.data.phi.values
    
    @property
    def e_co_db(self) -> xr.DataArray:
        return 10 * np.log10(np.maximum(np.abs(self.data.e_co)**2, 1e-30))

    @property
    def e_cx_db(self) -> xr.DataArray:
        return 10 * np.log10(np.maximum(np.abs(self.data.e_cx)**2, 1e-30))
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache = {}
    
    @contextmanager
    def at_frequency(self, frequency: float) -> 'Generator[FarFieldSpherical, None, None]':
        """
        Context manager to temporarily extract a single-frequency pattern.
        
        Args:
            frequency: Frequency in Hz
            
        Yields:
            FarFieldSpherical: Single-frequency pattern
            
        Example:
            ```python
            with pattern.at_frequency(2.4e9) as single_freq_pattern:
                # Work with single_freq_pattern
            ```
        """
        freq_value, freq_idx = find_nearest(self.frequencies, frequency)
        
        # Extract data for the nearest frequency
        single_freq_data = self.data.isel(frequency=freq_idx)
        
        # Create a new pattern with the extracted data
        single_freq_pattern = FarFieldSpherical(
            theta=self.theta_angles,
            phi=self.phi_angles,
            frequency=np.array([freq_value]),
            e_theta=np.expand_dims(single_freq_data.e_theta.values, axis=0),
            e_phi=np.expand_dims(single_freq_data.e_phi.values, axis=0),
            polarization=self.polarization,
            metadata={'parent_pattern': 'Single frequency view'}
        )
        
        yield single_freq_pattern
    
    @classmethod
    def from_ticra_sph(cls, file_path: Union[str, Path], frequency: float,
                    theta_angles: Optional[np.ndarray] = None,
                    phi_angles: Optional[np.ndarray] = None) -> 'FarFieldSpherical':
        """
        Create FarFieldSpherical from TICRA .sph file.
        
        Args:
            file_path: Path to .sph file
            frequency: Frequency in Hz
            theta_angles: Theta angles for reconstruction (default: -180 to 180, 1°)
            phi_angles: Phi angles for reconstruction (default: 0 to 360, 5°)
            
        Returns:
            FarFieldSpherical object with SWE coefficients attached
            
        Example:
        ```python
                pattern = FarFieldSpherical.from_ticra_sph('antenna.sph', frequency=9.2e9)
        ```
        """
        from .io.readers import read_ticra_sph
        from .io.swe_utils import create_pattern_from_swe

        # Read SWE coefficients
        swe_data = read_ticra_sph(file_path)

        # Create pattern from coefficients
        pattern = create_pattern_from_swe(swe_data, theta_angles, phi_angles)

        return pattern

    def copy(self) -> 'FarFieldSpherical':
        """
        Create a deep copy of the far-field pattern.
        
        Returns:
            FarFieldSpherical: A new FarFieldSpherical instance with copied data
        """
        return FarFieldSpherical(
            theta=self.theta_angles.copy(),
            phi=self.phi_angles.copy(),
            frequency=self.frequencies.copy(),
            e_theta=self.data.e_theta.values.copy(),
            e_phi=self.data.e_phi.values.copy(),
            polarization=self.polarization,
            metadata=self.metadata.copy() if self.metadata else None
        )
    
    def assign_polarization(self, polarization: Optional[str] = None) -> None:
        """
        Assign a polarization to the antenna pattern and compute e_co and e_cx.
        
        If polarization is None, it is automatically determined based on which 
        polarization component has the highest peak gain.
        
        Args:
            polarization: Polarization type or None to auto-detect
            
        Raises:
            ValueError: If the specified polarization is invalid
        """
        # Get underlying numpy arrays for calculations
        phi = self.data.phi.values
        e_theta = self.data.e_theta.values
        e_phi = self.data.e_phi.values
        
        # Calculate different polarization components
        e_x, e_y = polarization_tp2xy(phi, e_theta, e_phi)
        e_r, e_l = polarization_tp2rl(phi, e_theta, e_phi)
        
        # Auto-detect polarization if not specified
        if polarization is None:
            e_x_max = np.max(np.abs(e_x))
            e_y_max = np.max(np.abs(e_y))
            e_r_max = np.max(np.abs(e_r))
            e_l_max = np.max(np.abs(e_l))
            
            max_val = max(e_x_max, e_y_max, e_r_max, e_l_max)
            
            if e_x_max == max_val:
                polarization = "x"
            elif e_y_max == max_val:
                polarization = "y"
            elif e_r_max == max_val:
                polarization = "rhcp"
            elif e_l_max == max_val:
                polarization = "lhcp"
        
        # Map variations to standard polarization names
        pol_lower = polarization.lower() if polarization else ""
        
        if pol_lower in {"rhcp", "rh", "r"}:
            e_co, e_cx = e_r, e_l
            standard_pol = "rhcp"
        elif pol_lower in {"lhcp", "lh", "l"}:
            e_co, e_cx = e_l, e_r
            standard_pol = "lhcp"
        elif pol_lower in {"x", "l3x"}:
            e_co, e_cx = e_x, e_y
            standard_pol = "x"
        elif pol_lower in {"y", "l3y"}:
            e_co, e_cx = e_y, e_x
            standard_pol = "y"
        elif pol_lower == "theta":
            e_co, e_cx = e_theta, e_phi
            standard_pol = "theta"
        elif pol_lower == "phi":
            e_co, e_cx = e_phi, e_theta
            standard_pol = "phi"
        else:
            raise ValueError(f"Invalid polarization: {polarization}")
        
        # Store polarization components and type
        self.data['e_co'] = (('frequency', 'theta', 'phi'), e_co)
        self.data['e_cx'] = (('frequency', 'theta', 'phi'), e_cx)
        self.polarization = standard_pol

        # Update metadata
        if self.metadata is not None:
            self.metadata['polarization'] = standard_pol

        self.clear_cache()

    def find_phase_center(self, theta_angle: float, frequency: Optional[float] = None, 
                        n_iter: int = 10) -> np.ndarray:
        """
        Finds the optimum phase center given a theta angle and frequency.
        
        Args:
            theta_angle: Angle in degrees to optimize phase center for
            frequency: Optional specific frequency to use, or None to use first frequency
            n_iter: Number of iterations for basinhopping
            
        Returns:
            np.ndarray: [x, y, z] coordinates of the optimum phase center
        """
        return calculate_phase_center(self, theta_angle, frequency, n_iter)

    def shift_to_phase_center(self, theta_angle: float, frequency: Optional[float] = None,
                            n_iter: int = 10) -> np.ndarray:
        """
        Find the phase center and shift the pattern to it.
        
        Args:
            theta_angle: Angle in degrees to optimize phase center for
            frequency: Optional specific frequency to use
            n_iter: Number of iterations for basinhopping
            
        Returns:
            np.ndarray: The translation vector used
        """  
        translation = calculate_phase_center(self, theta_angle, frequency, n_iter)
        self.translate(translation)
        
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'shift_to_phase_center',
                'theta_angle': theta_angle,
                'frequency': frequency,
                'translation': translation.tolist() if hasattr(translation, 'tolist') else translation,
                'n_iter': n_iter
            })
        
        return translation

    def get_gain_db(self, component: str = 'e_co') -> xr.DataArray:
        """
        Get gain in dB for a specific field component.
        
        Args:
            component: Field component ('e_co', 'e_cx', 'e_theta', 'e_phi')
            
        Returns:
            xr.DataArray: Gain in dB
        """
        cache_key = f"gain_db_{component}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        result = 20 * np.log10(np.abs(self.data[component]))
        self._cache[cache_key] = result
        return result

    def get_phase(self, component: str = 'e_co', unwrapped: bool = False) -> xr.DataArray:
        """
        Get phase for a specific field component.
        
        Args:
            component: Field component ('e_co', 'e_cx', 'e_theta', 'e_phi')
            unwrapped: If True, return unwrapped phase (no 2π discontinuities)
            
        Returns:
            xr.DataArray: Phase in radians
        """
        from .pattern_operations import unwrap_phase

        if component not in self.data:
            raise KeyError(f"Component {component} not found in pattern data.")
        
        cache_key = f"phase_{component}_{'unwrapped' if unwrapped else 'wrapped'}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        phase = np.angle(self.data[component].values)
        
        if unwrapped:
            phase = unwrap_phase(phase, axis=1)
        
        # Convert to degrees
        phase = np.degrees(phase)

        result = xr.DataArray(
            phase,
            coords=self.data[component].coords,
            dims=self.data[component].dims
        )
        
        self._cache[cache_key] = result
        return result
        
    def get_axial_ratio(self) -> np.ndarray:
        """
        Calculate axial ratio in dB.
        
        Returns:
            Array of axial ratio values in dB with shape (frequency, theta, phi)
        """
        ar = get_axial_ratio(self)
        
        return ar
    
    def mirror_pattern(self) -> None:
        """
        Mirror the pattern across the theta=0 plane.
        
        This function reflects the pattern data across the theta=0 plane,
        effectively mirroring the pattern. It's useful for creating symmetric patterns
        or for fixing incomplete measurement data.
        
        Notes:
            If the pattern does not include theta=0, the function will raise a ValueError.
            The pattern should have theta values in [-180, 180] range.
        """
        # Call the mixin method
        super().mirror_pattern()

    def write_ffd(self, file_path: Union[str, Path]) -> None:
        """
        Write the far-field pattern to HFSS far field data format (.ffd).
        
        Args:
            file_path: Path to save the file to
            
        Raises:
            OSError: If file cannot be written
        """
        from .io.writers import write_ffd
        write_ffd(self, file_path)

    def write_cut(self, file_path: Union[str, Path], polarization_format: int = 1) -> None:
        """
        Write the far-field pattern to GRASP CUT format.
        
        Args:
            file_path: Path to save the file to
            polarization_format: Output polarization format:
                1 = theta/phi (spherical)
                2 = RHCP/LHCP (circular)
                3 = X/Y (Ludwig-3 linear)
                
        Raises:
            OSError: If file cannot be written
            ValueError: If polarization_format is invalid
        """
        from .io.writers import write_cut
        write_cut(self, file_path)

    def save_pattern_npz(self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a far-field pattern to NPZ format for efficient loading.

        Args:
            file_path: Path to save the file to
            metadata: Optional metadata to include

        Raises:
            OSError: If file cannot be written
        """
        from .io.npz_format import save_pattern_npz
        save_pattern_npz(self, file_path, metadata)

    def write_csv(self, file_path: Union[str, Path],
                  components: str = 'copol',
                  include_complex: bool = False) -> None:
        """
        Write the far-field pattern to CSV format.

        Args:
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
        from .io.writers import write_csv
        write_csv(self, file_path, components, include_complex)

    def calculate_spherical_modes(self, radius: Optional[float] = None, 
                                frequency: Optional[float] = None) -> 'SphericalWaveExpansion':
        """Calculate spherical wave expansion from the far-field pattern."""
        
        if frequency is None:
            frequency = self.frequencies[0]
        
        # Use context manager to get single frequency
        with self.at_frequency(frequency) as single_freq_pattern:
            single_freq_pattern.transform_coordinates('sided')
            
            # Extract angle arrays and field data
            theta_deg = single_freq_pattern.theta_angles
            phi_deg = single_freq_pattern.phi_angles
            theta_1d = np.radians(theta_deg)
            phi_1d = np.radians(phi_deg)
            E_theta_2d = single_freq_pattern.data.e_theta.values[0, :, :]
            E_phi_2d = single_freq_pattern.data.e_phi.values[0, :, :]

            # Create meshgrid and flatten for SWE (it expects flattened meshgrid arrays)
            THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')
            theta = THETA.ravel()
            phi = PHI.ravel()
            E_theta = E_theta_2d.ravel()
            E_phi = E_phi_2d.ravel()

            
            swe_obj = SphericalWaveExpansion.from_far_field(
                theta=theta,
                phi=phi,
                E_theta=E_theta,
                E_phi=E_phi,
                frequency=frequency,
            )
        
        return swe_obj
    
    def find_beamwidth_at_db_level(self, db_level: float, 
                                frequency: Optional[float] = None,
                                phi_cut: float = 0.0) -> float:
        """
        Find the beamwidth at a specified dB level below peak.
        
        This method finds the half-angle beamwidth where the pattern drops
        to the specified dB level below the peak. Useful for determining
        the angular extent for phase center calculations.
        
        Args:
            db_level: dB level below peak (should be negative, e.g., -10)
            frequency: Frequency to use (defaults to first frequency)
            phi_cut: Phi angle for the cut (default: 0°)
            
        Returns:
            Beamwidth in degrees (half-angle from boresight)
        """
        from .utilities import find_nearest
        
        # Get frequency index
        if frequency is None:
            freq_idx = 0
        else:
            _, freq_idx = find_nearest(self.frequencies, frequency)  # Returns (value, index)
            if isinstance(freq_idx, np.ndarray):
                freq_idx = freq_idx.item()
        
        # Get co-pol data and convert to dB (suppress divide by zero warnings)
        e_co = self.data.e_co.values[freq_idx, :, :]
        power_linear = np.abs(e_co) ** 2
        
        # Avoid log10(0) warnings by setting minimum value
        with np.errstate(divide='ignore', invalid='ignore'):
            power_db = 10 * np.log10(np.maximum(power_linear, 1e-30))
        
        # Replace any remaining inf/nan with very low dB value
        power_db = np.nan_to_num(power_db, nan=-300.0, neginf=-300.0, posinf=300.0)
        
        # Find peak (assume near boresight)
        peak_db = np.max(power_db)
        target_db = peak_db + db_level  # db_level is negative
        
        # Find phi cut index
        _, phi_idx = find_nearest(self.phi_angles, phi_cut)  # Returns (value, index)
        if isinstance(phi_idx, np.ndarray):
            phi_idx = phi_idx.item()
        
        # Get theta cut at specified phi
        theta_cut_db = power_db[:, phi_idx]
        theta_angles = self.theta_angles
        
        # Find peak location in theta cut
        peak_theta_idx = np.argmax(theta_cut_db)
        peak_theta = theta_angles[peak_theta_idx]
        
        # Find where the pattern crosses the target level
        # Search forward from peak
        forward_idx = peak_theta_idx
        while forward_idx < len(theta_cut_db) - 1 and theta_cut_db[forward_idx] >= target_db:
            forward_idx += 1
        
        # Interpolate for exact crossing
        if forward_idx < len(theta_cut_db) - 1 and forward_idx > peak_theta_idx:
            # Linear interpolation between points
            db1 = theta_cut_db[forward_idx - 1]
            db2 = theta_cut_db[forward_idx]
            theta1 = theta_angles[forward_idx - 1]
            theta2 = theta_angles[forward_idx]
            
            # Avoid division by zero
            if abs(db2 - db1) > 1e-10:
                frac = (target_db - db1) / (db2 - db1)
                beamwidth_theta = theta1 + frac * (theta2 - theta1)
            else:
                beamwidth_theta = theta1
        else:
            # Use the index directly if no interpolation possible
            if forward_idx < len(theta_angles):
                beamwidth_theta = theta_angles[forward_idx]
            else:
                # Pattern didn't drop to target level - use last angle
                beamwidth_theta = theta_angles[-1]
        
        # Return half-angle beamwidth (relative to peak)
        half_angle = abs(beamwidth_theta - peak_theta)
        
        return half_angle