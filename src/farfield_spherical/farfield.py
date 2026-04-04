"""
Core class for far-field spherical antenna pattern representation and manipulation.
"""
import warnings
import numpy as np
import xarray as xr
from typing import Optional, Union, Tuple, Dict, Any, Set, Generator
from pathlib import Path
from contextlib import contextmanager
from scipy.interpolate import interp1d

from .utilities import find_nearest
from .polarization import (
    polarization_tp2xy, polarization_tp2rl
)
from .analysis import (
    calculate_phase_center, get_axial_ratio
    )
from .farfield_operations import FarFieldOperationsMixin
from .pattern_operations import unwrap_phase
try:
    from swe import SphericalWaveExpansion  # pyright: ignore[reportMissingImports]
    _SWE_AVAILABLE = True
except ImportError:
    _SWE_AVAILABLE = False
    SphericalWaveExpansion = None  # type: ignore[assignment,misc]

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
            theta: Array of theta angles in degrees. Can be:
                - 1D array (n_theta,): Uniform grid shared by all phi cuts
                - 2D array (n_theta, n_phi): Per-phi theta grid (non-uniform)
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
        theta = np.asarray(theta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        frequency = np.asarray(frequency, dtype=np.float64)

        # Handle 2D theta (per-phi theta grids)
        if theta.ndim == 2:
            # Non-uniform theta: shape is (n_theta, n_phi)
            n_theta, n_phi_from_theta = theta.shape
            if n_phi_from_theta != len(phi):
                raise ValueError(
                    f"2D theta array n_phi dimension ({n_phi_from_theta}) must match "
                    f"phi array length ({len(phi)})"
                )
            # Store the full 2D theta grid
            self._theta_grid = theta.copy()
            # Use integer indices as the theta coordinate dimension
            theta_coord = np.arange(n_theta)
        elif theta.ndim == 1:
            # Uniform theta: standard behavior
            self._theta_grid = None
            theta_coord = theta
            n_theta = len(theta)
        else:
            raise ValueError(f"theta must be 1D or 2D array, got {theta.ndim}D")

        # Validate array dimensions
        expected_shape = (len(frequency), n_theta, len(phi))
        if e_theta.shape != expected_shape:
            raise ValueError(f"e_theta shape mismatch: expected {expected_shape}, got {e_theta.shape}")
        if e_phi.shape != expected_shape:
            raise ValueError(f"e_phi shape mismatch: expected {expected_shape}, got {e_phi.shape}")

        # Create core dataset
        data_vars = {
            'e_theta': (('frequency', 'theta', 'phi'), e_theta),
            'e_phi': (('frequency', 'theta', 'phi'), e_phi),
        }

        # For non-uniform theta, also store the 2D theta grid as a data variable
        if self._theta_grid is not None:
            data_vars['theta_grid'] = (('theta', 'phi'), self._theta_grid)

        self.data = xr.Dataset(
            data_vars=data_vars,
            coords={
                'theta': theta_coord,
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
    def has_uniform_theta(self) -> bool:
        """True if all phi cuts share the same theta grid."""
        return self._theta_grid is None

    @property
    def theta_grid(self) -> np.ndarray:
        """
        Get theta values as 2D array (n_theta, n_phi).

        Works for both uniform and non-uniform patterns.
        For uniform patterns, broadcasts the 1D array across phi.

        Returns:
            np.ndarray: 2D array of theta values with shape (n_theta, n_phi)
        """
        if self._theta_grid is not None:
            return self._theta_grid
        # Broadcast 1D to 2D for uniform case
        return np.tile(self.data.theta.values[:, np.newaxis], (1, len(self.phi_angles)))

    def get_theta_for_phi(self, phi_idx: int) -> np.ndarray:
        """
        Get the 1D theta array for a specific phi cut index.

        Args:
            phi_idx: Index of the phi cut

        Returns:
            np.ndarray: 1D array of theta values for the specified phi cut
        """
        if self._theta_grid is not None:
            return self._theta_grid[:, phi_idx]
        return self.data.theta.values

    @property
    def theta_angles(self) -> np.ndarray:
        """
        Get theta angles in degrees.

        For uniform grids, returns the shared 1D theta array.
        For non-uniform grids, raises ValueError directing to theta_grid or get_theta_for_phi().

        Returns:
            np.ndarray: 1D array of theta values (uniform grids only)

        Raises:
            ValueError: If pattern has non-uniform theta grids
        """
        if self._theta_grid is not None:
            raise ValueError(
                "Pattern has non-uniform theta grids (per-phi). "
                "Use .theta_grid for 2D array or .get_theta_for_phi(phi_idx) for a specific cut. "
                "Use .to_uniform_theta() to interpolate to a common grid."
            )
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

        # Use theta_grid for non-uniform, theta_angles for uniform
        if self._theta_grid is not None:
            theta_param = self._theta_grid
        else:
            theta_param = self.data.theta.values

        # Create a new pattern with the extracted data
        single_freq_pattern = FarFieldSpherical(
            theta=theta_param,
            phi=self.phi_angles,
            frequency=np.array([freq_value]),
            e_theta=np.expand_dims(single_freq_data.e_theta.values, axis=0),
            e_phi=np.expand_dims(single_freq_data.e_phi.values, axis=0),
            polarization=self.polarization,
            metadata={'parent_pattern': 'Single frequency view'}
        )

        yield single_freq_pattern

    def to_uniform_theta(self, theta: Optional[np.ndarray] = None) -> 'FarFieldSpherical':
        """
        Return a new FarFieldSpherical interpolated onto a uniform theta grid.

        Args:
            theta: Target uniform theta array. If None, uses a linear grid spanning
                   the common range across all phi cuts with the median spacing.

        Returns:
            New FarFieldSpherical with uniform theta grid.

        Notes:
            If the pattern already has a uniform theta grid, returns a copy.
            Uses scipy.interpolate.interp1d (linear) per-phi, per-frequency.
        """
        # If already uniform, return a copy
        if self.has_uniform_theta:
            return self.copy()

        # Compute default target theta if not specified
        if theta is None:
            # Find common overlap region across all phi cuts
            theta_mins = []
            theta_maxs = []
            theta_steps = []

            for phi_idx in range(len(self.phi_angles)):
                phi_theta = self.get_theta_for_phi(phi_idx)
                theta_mins.append(np.min(phi_theta))
                theta_maxs.append(np.max(phi_theta))
                if len(phi_theta) > 1:
                    # Compute median step for this cut
                    steps = np.diff(phi_theta)
                    theta_steps.append(np.median(steps))

            # Common range is max of mins to min of maxs
            theta_min = np.max(theta_mins)
            theta_max = np.min(theta_maxs)

            # Use median step across all cuts
            step = np.median(theta_steps) if theta_steps else 1.0

            # Generate uniform theta array
            theta = np.arange(theta_min, theta_max + step / 2, step)

        # Warn if the target theta range exceeds any phi cut's measured range
        any_extrap = False
        for phi_idx in range(len(self.phi_angles)):
            phi_theta = self.get_theta_for_phi(phi_idx)
            if theta[0] < np.min(phi_theta) or theta[-1] > np.max(phi_theta):
                any_extrap = True
                break
        if any_extrap:
            warnings.warn(
                "to_uniform_theta: the target theta range extends beyond the measured "
                "range for one or more phi cuts. Field values will be linearly "
                "extrapolated, which may be inaccurate. Consider limiting the target "
                "theta range to the common measured region.",
                UserWarning,
                stacklevel=2
            )

        # Interpolate fields to the new uniform theta grid
        n_freq = len(self.frequencies)
        n_theta_new = len(theta)
        n_phi = len(self.phi_angles)

        e_theta_new = np.zeros((n_freq, n_theta_new, n_phi), dtype=np.complex64)
        e_phi_new = np.zeros((n_freq, n_theta_new, n_phi), dtype=np.complex64)

        for freq_idx in range(n_freq):
            for phi_idx in range(n_phi):
                # Get per-phi theta array
                phi_theta = self.get_theta_for_phi(phi_idx)

                # Get field values for this freq/phi
                e_theta_vals = self.data.e_theta.values[freq_idx, :, phi_idx]
                e_phi_vals = self.data.e_phi.values[freq_idx, :, phi_idx]

                # Interpolate real and imaginary parts separately for stability
                # E_theta interpolation
                interp_real = interp1d(phi_theta, e_theta_vals.real, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
                interp_imag = interp1d(phi_theta, e_theta_vals.imag, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
                e_theta_new[freq_idx, :, phi_idx] = interp_real(theta) + 1j * interp_imag(theta)

                # E_phi interpolation
                interp_real = interp1d(phi_theta, e_phi_vals.real, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
                interp_imag = interp1d(phi_theta, e_phi_vals.imag, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
                e_phi_new[freq_idx, :, phi_idx] = interp_real(theta) + 1j * interp_imag(theta)

        # Build metadata
        new_metadata = self.metadata.copy() if self.metadata else {'operations': []}
        if 'operations' not in new_metadata:
            new_metadata['operations'] = []
        new_metadata['operations'].append({
            'type': 'to_uniform_theta',
            'original_theta_grid_shape': list(self._theta_grid.shape) if self._theta_grid is not None else None,
            'new_theta_range': [float(theta[0]), float(theta[-1])],
            'new_theta_points': len(theta)
        })

        # Create new pattern with uniform theta
        return FarFieldSpherical(
            theta=theta,  # 1D array -> uniform mode
            phi=self.phi_angles.copy(),
            frequency=self.frequencies.copy(),
            e_theta=e_theta_new,
            e_phi=e_phi_new,
            polarization=self.polarization,
            metadata=new_metadata
        )

    @classmethod
    def from_ticra_sph(cls, file_path: Union[str, Path], frequency: float,
                    theta_angles: Optional[np.ndarray] = None,
                    phi_angles: Optional[np.ndarray] = None) -> 'FarFieldSpherical':
        """
        Create FarFieldSpherical from TICRA .sph file.

        Note:
            When exporting .sph files from TICRA/GRASP, enable power normalization
            in the export settings so that the coefficients are normalized to unit
            radiated power. Unnormalized exports will produce a correct far-field
            pattern shape but incorrect absolute near-field levels in PO analysis.

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

        # Create pattern from coefficients at the requested frequency
        pattern = create_pattern_from_swe(swe_data, theta_angles, phi_angles, frequency)

        return pattern

    def copy(self) -> 'FarFieldSpherical':
        """
        Create a deep copy of the far-field pattern.

        Returns:
            FarFieldSpherical: A new FarFieldSpherical instance with copied data
        """
        # Use theta_grid for non-uniform, theta coordinates for uniform
        if self._theta_grid is not None:
            theta_param = self._theta_grid.copy()
        else:
            theta_param = self.data.theta.values.copy()

        return FarFieldSpherical(
            theta=theta_param,
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
        Get field magnitude in dB (20 * log10|E|) for a specific field component.

        This returns the field amplitude level in dB relative to the stored field
        units, not antenna gain in dBi. To compute true antenna gain or directivity
        use ``calculate_directivity()``.

        Args:
            component: Field component ('e_co', 'e_cx', 'e_theta', 'e_phi')

        Returns:
            xr.DataArray: Field magnitude in dB
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
        write_cut(self, file_path, polarization_format)

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

    def calculate_spherical_modes(self, frequency: Optional[float] = None,
                                nmax: Optional[int] = None,
                                mmax: Optional[int] = None) -> 'SphericalWaveExpansion':
        """Calculate spherical wave expansion from the far-field pattern.

        Parameters
        ----------
        frequency : float, optional
            Frequency in Hz. Defaults to first frequency.
        nmax : int, optional
            Maximum polar mode index. If None, determined automatically.
        mmax : int, optional
            Maximum azimuthal mode index. If None, determined automatically.
        """

        if not _SWE_AVAILABLE:
            raise ImportError(
                "The 'swe' package is required for spherical wave expansion. "
                "Install it with: pip install farfield-spherical[swe]"
            )

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

        # Post-process: truncate to user-specified NMAX/MMAX
        if nmax is not None or mmax is not None:
            n_limit = nmax if nmax is not None else swe_obj.NMAX
            m_limit = mmax if mmax is not None else swe_obj.MMAX
            swe_obj.Q1_coeffs = {
                (n, m): v for (n, m), v in swe_obj.Q1_coeffs.items()
                if n <= n_limit and abs(m) <= m_limit
            }
            swe_obj.Q2_coeffs = {
                (n, m): v for (n, m), v in swe_obj.Q2_coeffs.items()
                if n <= n_limit and abs(m) <= m_limit
            }
            swe_obj.NMAX = n_limit
            swe_obj.MMAX = m_limit

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
        self._require_uniform_theta('find_beamwidth_at_db_level')

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