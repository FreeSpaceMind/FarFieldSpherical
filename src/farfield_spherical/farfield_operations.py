"""
Mixin class that contains operations for FarFieldSpherical objects.
This class is designed to be mixed into the FarFieldSpherical class.
"""

import numpy as np
import xarray as xr
from typing import Tuple, Union, Optional, List, Any, Callable
from scipy.interpolate import interp1d
import logging

from .utilities import lightspeed, frequency_to_wavelength, find_nearest
from .polarization import polarization_tp2xy, polarization_tp2rl, polarization_xy2tp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .farfield import FarFieldSpherical

logger = logging.getLogger(__name__)

class FarFieldOperationsMixin:
    """Mixin class providing operations for far-field patterns."""

    def _require_uniform_theta(self, operation_name: str) -> None:
        """
        Raise NotImplementedError if the pattern has non-uniform theta grids.

        Args:
            operation_name: Name of the operation for the error message
        """
        if not self.has_uniform_theta:
            raise NotImplementedError(
                f"{operation_name} does not yet support non-uniform theta grids. "
                "Use .to_uniform_theta() first to interpolate to a common grid."
            )

    def change_polarization(self, new_polarization: str) -> None:
        """
        Change the polarization of the far-field pattern.
        
        Args:
            new_polarization: New polarization type to use
            
        Raises:
            ValueError: If the new polarization is invalid
        """
        # Simply call assign_polarization with the new polarization type
        self.assign_polarization(new_polarization)
        
        # Clear cache due to change in polarization
        self.clear_cache()
        
        # Update metadata if needed
        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata['polarization'] = self.polarization
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'change_polarization',
                'new_polarization': new_polarization
            })

    def translate(self, translation: np.ndarray) -> None:
        """
        Shifts the antenna phase pattern to place the origin at the location defined by the shift.

        This applies a linear phase shift to the pattern corresponding to a translation
        of the phase center. The phase shift is frequency-dependent.

        Args:
            translation: 3D translation vector [x, y, z] in meters

        Note:
            This modifies the pattern in-place.
            Use normalize_phase() separately if phase normalization is needed.
        """
        self._require_uniform_theta('translate')

        from .pattern_operations import phase_pattern_translate

        # Convert angles to radians for phase_pattern_translate
        theta_rad = np.radians(self.theta_angles)
        phi_rad = np.radians(self.phi_angles)
        
        # Apply translation to each frequency
        for freq_idx, freq in enumerate(self.frequencies):
            # Apply phase shift to theta component
            phase_e_theta = np.angle(self.data.e_theta.values[freq_idx])
            shifted_phase_theta = phase_pattern_translate(
                freq, theta_rad, phi_rad, translation, phase_e_theta
            )
            mag_e_theta = np.abs(self.data.e_theta.values[freq_idx])
            self.data.e_theta.values[freq_idx] = mag_e_theta * np.exp(1j * shifted_phase_theta)
            
            # Apply phase shift to phi component  
            phase_e_phi = np.angle(self.data.e_phi.values[freq_idx])
            shifted_phase_phi = phase_pattern_translate(
                freq, theta_rad, phi_rad, translation, phase_e_phi
            )
            mag_e_phi = np.abs(self.data.e_phi.values[freq_idx])
            self.data.e_phi.values[freq_idx] = mag_e_phi * np.exp(1j * shifted_phase_phi)
        
        # Recompute co/cx polarization
        self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'translate',
                'translation': translation.tolist()
            })

    def normalize_amplitude(self, reference_value: str = 'peak') -> None:
        """
        Normalize the amplitude of the pattern to a reference value.
        
        Args:
            reference_value: Normalization reference - 'peak' (default), 'boresight', or 'mean'
            
        Note:
            This modifies the pattern in-place.
            - 'peak': Normalizes to peak gain (0 dB at maximum)
            - 'boresight': Normalizes to boresight gain
            - 'mean': Normalizes to mean gain
        """
        # Get power patterns for all frequencies
        power_pattern = np.abs(self.data.e_co.values)**2 + np.abs(self.data.e_cx.values)**2
        
        # Find normalization factors for each frequency
        norm_factors = np.zeros(len(self.frequencies))
        
        for f_idx in range(len(self.frequencies)):
            if reference_value == 'peak':
                norm_factors[f_idx] = np.max(power_pattern[f_idx])
            elif reference_value == 'boresight':
                theta0_idx = np.argmin(np.abs(self.theta_angles))
                phi0_idx = np.argmin(np.abs(self.phi_angles))
                norm_factors[f_idx] = power_pattern[f_idx, theta0_idx, phi0_idx]
            elif reference_value == 'mean':
                norm_factors[f_idx] = np.mean(power_pattern[f_idx])
            else:
                raise ValueError(f"Unknown reference_value: {reference_value}")
        
        # Apply normalization (sqrt because we normalize field, not power)
        for f_idx in range(len(self.frequencies)):
            if norm_factors[f_idx] > 0:
                scale = 1.0 / np.sqrt(norm_factors[f_idx])
                self.data.e_theta.values[f_idx] *= scale
                self.data.e_phi.values[f_idx] *= scale
        
        # Recompute co/cx polarization
        self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'normalize_amplitude',
                'reference': reference_value
            })

    def normalize_phase(self, reference_theta=0, reference_phi=0) -> None:
        """
        Normalize the phase of an antenna pattern based on its polarization type.
        
        This function sets the phase of the co-polarized component at the reference
        point (closest to reference_theta, reference_phi) to zero, while preserving
        the relative phase between components.
        
        Args:
            reference_theta: Reference theta angle in degrees (default: 0)
            reference_phi: Reference phi angle in degrees (default: 0)
        """
        
        # Get underlying numpy arrays
        frequency = self.data.frequency.values
        theta = self.data.theta.values
        phi = self.data.phi.values
        e_theta = self.data.e_theta.values
        e_phi = self.data.e_phi.values
        
        # Find the indices for reference angles (or closest values)
        theta_ref_idx = np.argmin(np.abs(theta - reference_theta))
        phi_ref_idx = np.argmin(np.abs(phi - reference_phi))
        
        # Actual reference angle values (recorded in metadata)
        theta_ref_actual = theta[theta_ref_idx]
        phi_ref_actual = phi[phi_ref_idx]
        
        # Determine which component to use as reference based on polarization
        pol = self.polarization.lower()
        
        # Process each frequency separately
        for f_idx in range(len(frequency)):
            # Select reference component based on polarization type
            if pol in ('theta', 'phi'):
                # For spherical polarization, use the corresponding component
                if pol == 'theta':
                    ref_phase = np.angle(e_theta[f_idx, theta_ref_idx, phi_ref_idx])
                else:  # phi polarization
                    ref_phase = np.angle(e_phi[f_idx, theta_ref_idx, phi_ref_idx])
            
            elif pol in ('x', 'l3x', 'y', 'l3y'):
                # For Ludwig-3 polarization, calculate e_x and e_y
                e_x, e_y = polarization_tp2xy(
                    phi, 
                    e_theta[f_idx], 
                    e_phi[f_idx]
                )
                if pol in ('x', 'l3x'):
                    ref_phase = np.angle(e_x[theta_ref_idx, phi_ref_idx])
                else:  # y polarization
                    ref_phase = np.angle(e_y[theta_ref_idx, phi_ref_idx])
            
            elif pol in ('rhcp', 'rh', 'r', 'lhcp', 'lh', 'l'):
                # For circular polarization, calculate RHCP and LHCP components
                e_r, e_l = polarization_tp2rl(
                    phi,
                    e_theta[f_idx],
                    e_phi[f_idx]
                )
                if pol in ('rhcp', 'rh', 'r'):
                    ref_phase = np.angle(e_r[theta_ref_idx, phi_ref_idx])
                else:  # LHCP polarization
                    ref_phase = np.angle(e_l[theta_ref_idx, phi_ref_idx])
            
            else:
                # Fallback to e_theta for unknown polarization
                ref_phase = np.angle(e_theta[f_idx, theta_ref_idx, phi_ref_idx])
            
            # Apply phase normalization by subtracting reference phase
            # This preserves relative phase relationships
            phase_correction = np.exp(-1j * ref_phase)
            e_theta[f_idx] = e_theta[f_idx] * phase_correction
            e_phi[f_idx] = e_phi[f_idx] * phase_correction
        
        # Update the pattern data directly
        self.data['e_theta'].values = e_theta
        self.data['e_phi'].values = e_phi
        
        # Recalculate derived components e_co and e_cx
        self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata if needed
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'normalize_phase',
                'reference_theta': reference_theta,
                'reference_phi': reference_phi,
                'actual_theta': float(theta_ref_actual),
                'actual_phi': float(phi_ref_actual)
            })

    def scale_amplitude(self, scale_factor: float) -> None:
        """
        Scale the amplitude of the pattern by a constant factor.
        
        Args:
            scale_factor: Factor to scale amplitudes by (linear, not dB)
            
        Note:
            This modifies the pattern in-place.
        """
        from .pattern_operations import scale_amplitude
        
        self.data.e_theta.values = scale_amplitude(self.data.e_theta.values, scale_factor)
        self.data.e_phi.values = scale_amplitude(self.data.e_phi.values, scale_factor)
        
        # Recompute co/cx polarization
        self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'scale_amplitude',
                'scale_factor': scale_factor
            })

    def rotate(self, alpha: float, beta: float, gamma: float,
               method: str = 'linear') -> None:
        """
        Rotate the pattern rigidly, as if the antenna itself were rotated.

        The rotation is R = R_y(alpha) . R_x(-beta) . R_z(gamma), applied in the
        order roll (gamma, about z), elevation (beta), azimuth (alpha), with
        the matrices defined in ``pattern_operations._rotation_matrix``. The
        field at a direction r' after rotation is the rotated field the
        antenna radiated toward R^-1 r' before rotation:

            E'(r') = R . E(R^-1 r')

        so both the sampling direction and the field vector are rotated. The
        original boresight (+z) ends up at direction R.z, i.e.

            theta_0 = arccos(cos(alpha) cos(beta))
            phi_0   = atan2(sin(beta), sin(alpha) cos(beta))

        A positive ``alpha`` tilts the boresight toward +x (phi = 0), a
        positive ``beta`` toward +y (phi = 90 deg), and a positive ``gamma``
        rolls the pattern from +x toward +y about the z-axis.

        This is a rigid rotation of the antenna. It is not the same as
        ``shift_theta_origin`` / ``shift_phi_origin``, which re-zero the
        measured angle axes of each cut to correct positioner misalignment.

        The result is sampled on the pattern's existing (theta, phi) grid, in
        its existing coordinate format, by interpolating the Cartesian
        components of the original field over the sphere. Directions that fall
        outside the original angular coverage (partial-sphere patterns) are
        set to zero and a warning is logged.

        Args:
            alpha: Azimuth rotation about the y-axis, degrees
            beta: Elevation rotation about the x-axis, degrees
            gamma: Roll about the z-axis, degrees
            method: Interpolation method passed to
                ``scipy.interpolate.RegularGridInterpolator``
                ('linear', 'nearest', 'slinear', 'cubic', ...)

        Raises:
            NotImplementedError: If the pattern has non-uniform theta grids
            ValueError: If the pattern has fewer than two theta or phi samples
        """
        from scipy.interpolate import RegularGridInterpolator
        from .pattern_operations import _rotation_matrix

        self._require_uniform_theta('rotate')

        R = _rotation_matrix(alpha, beta, gamma)
        if np.allclose(R, np.eye(3)):
            return

        # --- Source: a sided, phi-normalised copy of the current data --------
        src = self.copy()
        src.transform_coordinates('sided', _preserve_polarization=True)
        th_s = np.asarray(src.theta_angles, dtype=float)
        ph_s = np.asarray(src.phi_angles, dtype=float)
        if th_s.size < 2 or ph_s.size < 2:
            raise ValueError("rotate requires at least two theta and two phi samples.")
        es_theta = np.asarray(src.data.e_theta.values, dtype=np.complex128)
        es_phi = np.asarray(src.data.e_phi.values, dtype=np.complex128)

        # Cartesian field components on the source grid: (freq, theta, phi, 3)
        th_hat_s, ph_hat_s = self._spherical_basis(th_s, ph_s)
        e_cart = (es_theta[..., None] * th_hat_s[None] + es_phi[..., None] * ph_hat_s[None])

        # Periodic padding in phi when the grid covers the full circle, so the
        # interpolator can wrap across the phi seam.
        dphi = np.diff(ph_s)
        full_circle = (ph_s.size > 1 and np.allclose(dphi, dphi[0], atol=1e-6)
                       and np.isclose(ph_s.size * dphi[0], 360.0, atol=1e-6))
        ph_grid = ph_s
        if full_circle:
            ph_grid = np.append(ph_s, ph_s[0] + 360.0)
            e_cart = np.concatenate([e_cart, e_cart[:, :, :1, :]], axis=2)

        # Interpolate real and imaginary parts as trailing dimensions:
        # values shape (theta, phi, freq, 3, 2)
        values = np.stack([e_cart.real, e_cart.imag], axis=-1).transpose(1, 2, 0, 3, 4)
        interp = RegularGridInterpolator((th_s, ph_grid), values, method=method,
                                         bounds_error=False, fill_value=np.nan)

        # --- Target: the pattern's own grid, in its own format ---------------
        th_t = np.asarray(self.theta_angles, dtype=float)
        ph_t = np.asarray(self.phi_angles, dtype=float)
        r_hat_t = self._direction(th_t, ph_t)                  # (theta, phi, 3)
        th_hat_t, ph_hat_t = self._spherical_basis(th_t, ph_t)

        # Where did the antenna radiate toward r' before rotation? R^-1 r'.
        r_src = r_hat_t @ R                                     # (R^T r')^T
        th_src = np.degrees(np.arccos(np.clip(r_src[..., 2], -1.0, 1.0)))
        ph_src = np.degrees(np.arctan2(r_src[..., 1], r_src[..., 0]))
        ph_src = ph_grid[0] + np.mod(ph_src - ph_grid[0], 360.0)
        if full_circle:
            # Values within tolerance of the padded end belong to the seam
            ph_src = np.where(ph_src > ph_grid[-1], ph_grid[-1], ph_src)

        sampled = interp(np.stack([th_src, ph_src], axis=-1))   # (theta, phi, freq, 3, 2)
        missing = np.isnan(sampled[..., 0, 0, 0])
        if missing.any():
            logger.warning(
                "rotate: %d of %d directions fall outside the pattern's angular "
                "coverage and were set to zero.", int(missing.sum()), missing.size)
            sampled = np.nan_to_num(sampled)
        e_src = sampled[..., 0] + 1j * sampled[..., 1]           # (theta, phi, freq, 3)

        # Rotate the field vectors and project onto the target basis
        e_rot = e_src @ R.T                                     # (theta, phi, freq, 3)
        e_theta_new = np.einsum('tpfc,tpc->ftp', e_rot, th_hat_t)
        e_phi_new = np.einsum('tpfc,tpc->ftp', e_rot, ph_hat_t)

        self.data['e_theta'].values = e_theta_new.astype(np.complex64)
        self.data['e_phi'].values = e_phi_new.astype(np.complex64)
        self.assign_polarization(self.polarization)
        self.clear_cache()

        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata.setdefault('operations', []).append({
                'type': 'rotate',
                'alpha': float(alpha), 'beta': float(beta), 'gamma': float(gamma),
                'method': method,
            })

    @staticmethod
    def _direction(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
        """Unit direction vectors r_hat on a (theta, phi) grid, shape (theta, phi, 3).

        Valid for negative theta (central format), where sin(theta) < 0 places
        the direction on the phi + 180 side.
        """
        th = np.radians(theta_deg)[:, None]
        ph = np.radians(phi_deg)[None, :]
        return np.stack([np.sin(th) * np.cos(ph),
                         np.sin(th) * np.sin(ph),
                         np.cos(th) * np.ones_like(ph)], axis=-1)

    @staticmethod
    def _spherical_basis(theta_deg: np.ndarray, phi_deg: np.ndarray):
        """theta_hat and phi_hat unit vectors on a (theta, phi) grid, each (theta, phi, 3).

        Uses the analytic expressions, which for negative theta give exactly the
        basis the central-format field components are referred to.
        """
        th = np.radians(theta_deg)[:, None]
        ph = np.radians(phi_deg)[None, :]
        ones = np.ones_like(th * ph)
        th_hat = np.stack([np.cos(th) * np.cos(ph),
                           np.cos(th) * np.sin(ph),
                           -np.sin(th) * ones], axis=-1)
        ph_hat = np.stack([-np.sin(ph) * ones,
                           np.cos(ph) * ones,
                           np.zeros_like(ones)], axis=-1)
        return th_hat, ph_hat

    def unwrap_phase(self, component: str = 'e_co', axis: int = 1) -> np.ndarray:
        """
        Unwrap phase discontinuities for a component.

        Args:
            component: Field component to unwrap
            axis: Axis along which to unwrap (default: 1 for theta)

        Returns:
            Unwrapped phase in radians
        """
        from .pattern_operations import unwrap_phase

        field = self.data[component].values
        phase = np.angle(field)

        return unwrap_phase(phase, axis=axis)

    def mirror_pattern(self) -> None:
        """
        Mirror the pattern across the theta=0 plane.

        Copies the theta > 0 data of every phi cut onto the matching theta < 0
        samples of the same cut, with E_theta negated and E_phi unchanged.
        Useful for completing a central-format pattern that was only measured
        on one side of boresight.

        Raises:
            ValueError: If the pattern is not in central format, does not
                include theta=0, or its theta grid is not symmetric about 0
        """
        self._require_uniform_theta('mirror_pattern')
        theta = np.asarray(self.theta_angles, dtype=float)
        if theta.min() >= 0:
            raise ValueError("mirror_pattern requires a central-format pattern (negative theta)")
        if not np.any(np.isclose(theta, 0.0, atol=1e-6)):
            raise ValueError("Pattern must include theta=0 to mirror")

        neg = np.where(theta < -1e-6)[0]
        rows = self._match_rows(-theta[neg], theta)
        if np.any(rows < 0):
            raise ValueError("mirror_pattern requires a theta grid symmetric about 0")

        e_theta = self.data.e_theta.values
        e_phi = self.data.e_phi.values
        e_theta[:, neg, :] = -e_theta[:, rows, :]
        e_phi[:, neg, :] = e_phi[:, rows, :]

        self.assign_polarization(self.polarization)
        self.clear_cache()

        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata.setdefault('operations', []).append({'type': 'mirror_pattern'})

    def interpolate_frequency(self, new_frequencies: np.ndarray,
                            kind: str = 'linear') -> 'FarFieldSpherical':
        """
        Interpolate pattern to new frequency points.

        Args:
            new_frequencies: Array of new frequencies in Hz
            kind: Interpolation type ('linear', 'cubic', etc.)

        Returns:
            New FarFieldSpherical object at interpolated frequencies
        """
        self._require_uniform_theta('interpolate_frequency')

        # Interpolate complex fields
        e_theta_interp = np.zeros((len(new_frequencies), len(self.theta_angles),
                                   len(self.phi_angles)), dtype=np.complex64)
        e_phi_interp = np.zeros_like(e_theta_interp)
        
        for i, theta_idx in enumerate(self.theta_angles):
            for j, phi_idx in enumerate(self.phi_angles):
                # Interpolate e_theta
                f_real = interp1d(self.frequencies, 
                                 self.data.e_theta.values[:, i, j].real, 
                                 kind=kind, fill_value='extrapolate')
                f_imag = interp1d(self.frequencies, 
                                 self.data.e_theta.values[:, i, j].imag, 
                                 kind=kind, fill_value='extrapolate')
                e_theta_interp[:, i, j] = f_real(new_frequencies) + 1j * f_imag(new_frequencies)
                
                # Interpolate e_phi
                f_real = interp1d(self.frequencies, 
                                 self.data.e_phi.values[:, i, j].real, 
                                 kind=kind, fill_value='extrapolate')
                f_imag = interp1d(self.frequencies, 
                                 self.data.e_phi.values[:, i, j].imag, 
                                 kind=kind, fill_value='extrapolate')
                e_phi_interp[:, i, j] = f_real(new_frequencies) + 1j * f_imag(new_frequencies)
        
        # Create new pattern
        from .farfield import FarFieldSpherical
        return FarFieldSpherical(
            theta=self.theta_angles,
            phi=self.phi_angles,
            frequency=new_frequencies,
            e_theta=e_theta_interp,
            e_phi=e_phi_interp,
            polarization=self.polarization,
            metadata={'source': 'interpolated', 'original_metadata': self.metadata}
        )
    
    # ------------------------------------------------------------------
    # Coordinate-format transforms
    # ------------------------------------------------------------------

    _ANGLE_TOL = 1e-6

    @classmethod
    def _normalize_phi(cls, phi: np.ndarray, *fields: np.ndarray):
        """
        Map phi into [0, 360), sort ascending and merge duplicate cuts.

        A cut duplicated after wrapping (for example -180 and +180, or 0 and
        360) keeps its first occurrence. ``fields`` are (frequency, theta, phi)
        arrays reordered alongside phi.
        """
        phi = np.asarray(phi, dtype=float)
        phi_mod = np.mod(phi, 360.0)
        phi_mod = np.where(np.isclose(phi_mod, 360.0, atol=cls._ANGLE_TOL), 0.0, phi_mod)
        order = np.argsort(phi_mod, kind='stable')
        phi_sorted = phi_mod[order]
        keep = np.ones(len(phi_sorted), dtype=bool)
        keep[1:] = ~np.isclose(np.diff(phi_sorted), 0.0, atol=cls._ANGLE_TOL)
        idx = order[keep]
        return phi_sorted[keep], [f[:, :, idx] for f in fields]

    @classmethod
    def _match_rows(cls, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Index in ascending ``grid`` of each value (within tolerance), or -1."""
        values = np.asarray(values, dtype=float)
        grid = np.asarray(grid, dtype=float)
        result = np.full(values.shape, -1, dtype=int)
        if grid.size == 0:
            return result
        pos = np.clip(np.searchsorted(grid, values), 0, grid.size - 1)
        for cand in (pos, np.clip(pos - 1, 0, grid.size - 1)):
            hit = (result < 0) & np.isclose(grid[cand], values, atol=cls._ANGLE_TOL)
            result[hit] = cand[hit]
        return result

    @classmethod
    def _cut_key(cls, keys: List[float], value: float) -> float:
        """Return the existing key within tolerance of ``value``, else ``value`` itself."""
        for k in keys:
            if abs(k - value) <= cls._ANGLE_TOL:
                return k
        return value

    @classmethod
    def _regroup_cuts(cls, n_freq: int, n_theta: int, placements):
        """
        Assemble phi cuts from a list of placements.

        Each placement is ``(phi, rows, e_theta, e_phi)`` with ``rows`` the
        target theta indices and the fields shaped (frequency, len(rows)).
        Later placements overwrite earlier ones on the rows they cover.

        Returns (phi, e_theta, e_phi, filled) sorted by phi, where ``filled``
        is a (theta, phi) boolean mask of rows that received data.
        """
        keys: List[float] = []
        cuts = {}
        for phi_c, rows, et, ep in placements:
            phi_c = 0.0 if np.isclose(phi_c, 360.0, atol=cls._ANGLE_TOL) else float(phi_c)
            key = cls._cut_key(keys, phi_c)
            if key not in cuts:
                keys.append(key)
                cuts[key] = (np.zeros((n_freq, n_theta), dtype=np.complex64),
                             np.zeros((n_freq, n_theta), dtype=np.complex64),
                             np.zeros(n_theta, dtype=bool))
            c_et, c_ep, c_ok = cuts[key]
            c_et[:, rows] = et
            c_ep[:, rows] = ep
            c_ok[rows] = True

        keys.sort()
        phi_out = np.array(keys, dtype=float)
        e_theta = np.stack([cuts[k][0] for k in keys], axis=2)
        e_phi = np.stack([cuts[k][1] for k in keys], axis=2)
        filled = np.stack([cuts[k][2] for k in keys], axis=1)
        return phi_out, e_theta, e_phi, filled

    def _central_to_sided(self, theta, phi, e_theta, e_phi):
        """Central (theta +/-, phi 0..180) -> sided (theta >= 0, phi 0..360)."""
        n_freq = e_theta.shape[0]
        pos = theta >= -self._ANGLE_TOL
        new_theta = theta[pos].copy()
        if np.isclose(new_theta[0], 0.0, atol=self._ANGLE_TOL):
            new_theta[0] = 0.0
        n_theta = len(new_theta)

        neg_idx = np.where(~pos)[0][::-1]                # ascending |theta|
        neg_rows = self._match_rows(-theta[neg_idx], new_theta)
        valid = neg_rows >= 0
        if not valid.all():
            logger.warning(
                "transform_coordinates: %d negative-theta samples have no matching "
                "positive-theta sample and were dropped.", int((~valid).sum()))
        neg_idx, neg_rows = neg_idx[valid], neg_rows[valid]
        has_zero = new_theta[0] == 0.0

        placements = []
        # Mirrored half first: theta < 0 at phi maps to theta > 0 at phi + 180,
        # with both spherical components negated. The boresight sample is shared
        # and gets the same sign flip so the cut is continuous through theta = 0.
        for j, phi_c in enumerate(phi):
            rows, et, ep = neg_rows, -e_theta[:, neg_idx, j], -e_phi[:, neg_idx, j]
            if has_zero:
                rows = np.concatenate([[0], rows])
                et = np.concatenate([-e_theta[:, pos, j][:, :1], et], axis=1)
                ep = np.concatenate([-e_phi[:, pos, j][:, :1], ep], axis=1)
            placements.append((phi_c + 180.0 if phi_c < 180.0 else phi_c - 180.0, rows, et, ep))
        # Direct half last so measured data wins where both exist
        all_rows = np.arange(n_theta)
        for j, phi_c in enumerate(phi):
            placements.append((phi_c, all_rows, e_theta[:, pos, j], e_phi[:, pos, j]))

        new_phi, new_e_theta, new_e_phi, filled = self._regroup_cuts(n_freq, n_theta, placements)
        return new_theta, new_phi, new_e_theta, new_e_phi, filled

    def _sided_to_central(self, theta, phi, e_theta, e_phi):
        """Sided (theta 0..180, phi 0..360) -> central (theta +/-, phi 0..180)."""
        n_freq = e_theta.shape[0]
        if theta[0] > self._ANGLE_TOL:
            raise ValueError("Input theta must start at 0 when transforming to central")
        theta = theta.copy()
        theta[0] = 0.0
        n_pos = len(theta)
        new_theta = np.concatenate((-theta[1:][::-1], theta))
        n_theta = len(new_theta)
        neg_rows = np.arange(n_pos - 1)                  # rows for -theta[1:], flipped
        pos_rows = np.arange(n_pos - 1, n_theta)

        placements = []
        # Cuts at phi >= 180 supply the negative-theta half of the cut at phi - 180
        for j, phi_c in enumerate(phi):
            if phi_c >= 180.0 - self._ANGLE_TOL:
                rows = np.concatenate([neg_rows, [n_pos - 1]])
                et = np.concatenate([-e_theta[:, 1:, j][:, ::-1], -e_theta[:, :1, j]], axis=1)
                ep = np.concatenate([-e_phi[:, 1:, j][:, ::-1], -e_phi[:, :1, j]], axis=1)
                placements.append((phi_c - 180.0, rows, et, ep))
        for j, phi_c in enumerate(phi):
            if phi_c < 180.0 - self._ANGLE_TOL:
                placements.append((phi_c, pos_rows, e_theta[:, :, j], e_phi[:, :, j]))

        new_phi, new_e_theta, new_e_phi, filled = self._regroup_cuts(n_freq, n_theta, placements)
        return new_theta, new_phi, new_e_theta, new_e_phi, filled

    def _canonicalize_central(self, theta, phi, e_theta, e_phi):
        """Central input: fold any cut with phi outside [0, 180) back into range."""
        n_freq, n_theta = e_theta.shape[0], len(theta)
        high = phi >= 180.0 - self._ANGLE_TOL
        if not high.any():
            return theta, phi, e_theta, e_phi, np.ones((n_theta, len(phi)), dtype=bool)

        rows = self._match_rows(-theta, theta)           # theta -> -theta row
        valid = rows >= 0
        if not valid.all():
            logger.warning(
                "transform_coordinates: theta grid is not symmetric about 0; %d samples "
                "of folded phi cuts were dropped.", int((~valid).sum()))
        src_rows = np.where(valid)[0]
        dst_rows = rows[valid]

        placements = []
        for j, phi_c in enumerate(phi):
            if high[j]:
                placements.append((phi_c - 180.0, dst_rows,
                                   -e_theta[:, src_rows, j], -e_phi[:, src_rows, j]))
        all_rows = np.arange(n_theta)
        for j, phi_c in enumerate(phi):
            if not high[j]:
                placements.append((phi_c, all_rows, e_theta[:, :, j], e_phi[:, :, j]))

        new_phi, new_e_theta, new_e_phi, filled = self._regroup_cuts(n_freq, n_theta, placements)
        return theta, new_phi, new_e_theta, new_e_phi, filled

    def transform_coordinates(self, format: str = 'sided', _preserve_polarization: bool = False) -> None:
        """
        Transform pattern coordinates to conform to a specified theta/phi convention.

        This function rearranges the existing pattern data to match one of two standard
        coordinate conventions without interpolation:

        - 'sided': theta 0:180, phi 0:360 (spherical convention)
        - 'central': theta -180:180, phi 0:180 (more common for antenna patterns)

        Phi is always normalised to [0, 360) and sorted, and duplicate cuts
        (for example both -180 and +180, or both 0 and 360) are merged. Cuts are
        paired by their phi *values*: the theta < 0 half of the cut at phi and
        the theta > 0 half of the cut at phi + 180 describe the same
        directions, with both spherical field components negated because the
        local theta_hat and phi_hat reverse across boresight. Positive-theta
        (directly sampled) data wins where both a direct and a mirrored cut land
        on the same output cut. Output samples with no source data (partial
        spheres, unpaired cuts, asymmetric theta ranges) are zero-filled and a
        warning is logged.

        Calling with the pattern already in the requested format still
        normalises phi, and for central input folds any cut outside
        0 <= phi < 180 back into that range.

        Args:
            format: Target coordinate format ('sided' or 'central')
            _preserve_polarization: Internal flag to skip polarization recalculation

        Raises:
            ValueError: If format is not 'sided' or 'central', or if a sided
                pattern whose theta does not start at 0 is transformed to central
            NotImplementedError: If pattern has non-uniform theta grids
        """
        self._require_uniform_theta('transform_coordinates')

        if format not in ['sided', 'central']:
            raise ValueError("Format must be 'sided' or 'central'")

        theta = np.asarray(self.theta_angles, dtype=float)
        phi = np.asarray(self.phi_angles, dtype=float)
        e_theta = self.data.e_theta.values.copy()
        e_phi = self.data.e_phi.values.copy()
        frequencies = self.frequencies

        theta_min, theta_max = float(np.min(theta)), float(np.max(theta))
        phi_min, phi_max = float(np.min(phi)), float(np.max(phi))

        phi, (e_theta, e_phi) = self._normalize_phi(phi, e_theta, e_phi)
        is_central = theta_min < -0.5

        if format == 'sided':
            if is_central:
                new_theta, new_phi, new_e_theta, new_e_phi, filled = \
                    self._central_to_sided(theta, phi, e_theta, e_phi)
            else:
                new_theta, new_phi, new_e_theta, new_e_phi = theta, phi, e_theta, e_phi
                filled = np.ones((len(theta), len(phi)), dtype=bool)
        else:
            if is_central:
                new_theta, new_phi, new_e_theta, new_e_phi, filled = \
                    self._canonicalize_central(theta, phi, e_theta, e_phi)
            else:
                new_theta, new_phi, new_e_theta, new_e_phi, filled = \
                    self._sided_to_central(theta, phi, e_theta, e_phi)

        if not filled.all():
            logger.warning(
                "transform_coordinates('%s'): %d of %d output samples have no source "
                "data (incomplete sphere) and were set to zero.",
                format, int((~filled).sum()), filled.size)

        self.data = xr.Dataset(
            data_vars={
                'e_theta': (('frequency', 'theta', 'phi'), np.asarray(new_e_theta, dtype=np.complex64)),
                'e_phi': (('frequency', 'theta', 'phi'), np.asarray(new_e_phi, dtype=np.complex64)),
            },
            coords={
                'theta': new_theta,
                'phi': new_phi,
                'frequency': frequencies,
            }
        )

        if not _preserve_polarization:
            self.assign_polarization(self.polarization)

        self.clear_cache()

        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata.setdefault('operations', []).append({
                'type': 'transform_coordinates',
                'format': format,
                'old_theta_range': [theta_min, theta_max],
                'old_phi_range': [phi_min, phi_max],
                'new_theta_range': [float(np.min(new_theta)), float(np.max(new_theta))],
                'new_phi_range': [float(np.min(new_phi)), float(np.max(new_phi))]
            })

    def normalize_at_boresight(self) -> None:
        """
        Normalize the pattern at boresight using Ludwig's III (e_x, e_y) components.

        Each phi cut is scaled so that all cuts have the same amplitude and phase
        at boresight (theta=0). The reference amplitude is the median magnitude
        across all phi cuts, and the reference phase is from the first phi cut.
        """
        # Get underlying numpy arrays
        frequency = self.data.frequency.values
        theta = self.data.theta.values
        phi = self.data.phi.values
        e_theta = self.data.e_theta.values.copy()
        e_phi = self.data.e_phi.values.copy()

        # Find boresight index
        theta0_idx = np.argmin(np.abs(theta))

        # Process each frequency separately
        for f_idx in range(len(frequency)):
            # Convert all theta, phi points to x, y
            e_x, e_y = polarization_tp2xy(phi, e_theta[f_idx], e_phi[f_idx])

            # Get boresight values for all phi cuts
            e_x_boresight = e_x[theta0_idx, :]
            e_y_boresight = e_y[theta0_idx, :]

            # Calculate median magnitude at boresight
            e_x_med_mag = np.median(np.abs(e_x_boresight))
            e_y_med_mag = np.median(np.abs(e_y_boresight))

            # Get reference phase from median phase across phi cuts
            e_x_ref_phase = np.median(np.angle(e_x_boresight))
            e_y_ref_phase = np.median(np.angle(e_y_boresight))

            # Normalize each phi cut
            for p_idx in range(len(phi)):
                # Create reference values (median magnitude and phase)
                e_x_ref = e_x_med_mag * np.exp(1j * e_x_ref_phase)
                e_y_ref = e_y_med_mag * np.exp(1j * e_y_ref_phase)

                # Calculate correction factors (avoid division by zero)
                if np.abs(e_x_boresight[p_idx]) > 1e-30:
                    e_x_correction = e_x_ref / e_x_boresight[p_idx]
                else:
                    e_x_correction = 1.0
                if np.abs(e_y_boresight[p_idx]) > 1e-30:
                    e_y_correction = e_y_ref / e_y_boresight[p_idx]
                else:
                    e_y_correction = 1.0

                # Apply correction to all theta values for this phi
                e_x[:, p_idx] *= e_x_correction
                e_y[:, p_idx] *= e_y_correction

            # Convert back to e_theta, e_phi
            e_theta_new, e_phi_new = polarization_xy2tp(phi, e_x, e_y)
            e_theta[f_idx] = e_theta_new
            e_phi[f_idx] = e_phi_new
        
        # Update pattern data
        self.data['e_theta'].values = e_theta
        self.data['e_phi'].values = e_phi
        
        # Recalculate derived components
        self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'normalize_at_boresight',
            })

    def apply_mars(self, maximum_radial_extent: float) -> None:
        """
        Apply Mathematical Absorber Reflection Suppression technique.

        Args:
            maximum_radial_extent: Maximum radial extent of the antenna in meters
        """
        self._require_uniform_theta('apply_mars')

        if maximum_radial_extent <= 0:
            raise ValueError("Maximum radial extent must be positive")
        
        frequency = self.data.frequency.values
        theta = self.data.theta.values
        phi = self.data.phi.values
        e_theta = self.data.e_theta.values.copy()
        e_phi = self.data.e_phi.values.copy()
        
        # Initialize outputs
        e_theta_new = np.empty_like(e_theta)
        e_phi_new = np.empty_like(e_phi)
        
        # Apply MARS algorithm
        for f_idx, f in enumerate(frequency):
            # Calculate wavenumber and coefficients range
            wavenumber = 2 * np.pi * f / lightspeed
            max_coefficients = int(np.floor(wavenumber * maximum_radial_extent))
            coefficients = np.arange(-max_coefficients, max_coefficients + 1, 1)
            
            # Create arrays for theta in radians
            theta_rad = np.radians(theta)
            
            # Initialize storage arrays for cylindrical coefficients
            CMC_1_sum = np.zeros_like(e_theta[f_idx, :, :], dtype=complex)
            CMC_2_sum = np.zeros_like(e_phi[f_idx, :, :], dtype=complex)
            
            # Precompute exponential terms for efficiency
            exp_terms = np.zeros((len(coefficients), len(theta)), dtype=complex)
            for n_idx, n in enumerate(coefficients):
                exp_terms[n_idx, :] = np.exp(-1j * n * theta_rad)
            
            # Process each coefficient
            for n_idx, n in enumerate(coefficients):
                # Compute mode coefficient for theta component
                CMC_1 = (
                    -1 * ((-1j) ** (-n)) / (4 * np.pi * wavenumber) *
                    np.trapezoid(
                        (e_theta[f_idx, :, :].transpose() * exp_terms[n_idx, :]).transpose(),
                        theta_rad, axis=0
                    )
                )
                
                # Compute mode coefficient for phi component
                CMC_2 = (
                    -1j * ((-1j) ** (-n)) / (4 * np.pi * wavenumber) *
                    np.trapezoid(
                        (e_phi[f_idx, :, :].transpose() * exp_terms[n_idx, :]).transpose(),
                        theta_rad, axis=0
                    )
                )
                
                # Sum the modes
                CMC_1_term = np.outer(exp_terms[n_idx, :], (-1j) ** n * CMC_1)
                CMC_2_term = np.outer(exp_terms[n_idx, :], (-1j) ** n * CMC_2)
                
                CMC_1_sum += CMC_1_term
                CMC_2_sum += CMC_2_term
            
            # Compute final field components
            e_phi_new[f_idx, :, :] = 2 * 1j * wavenumber * CMC_2_sum
            e_theta_new[f_idx, :, :] = -2 * wavenumber * CMC_1_sum
        
        # Flip the theta axis because of coordinate system difference from reference
        e_theta_flipped = np.flip(e_theta_new, axis=1)
        e_phi_flipped = np.flip(e_phi_new, axis=1)
        
        # Update the pattern data directly
        self.data['e_theta'].values = e_theta_flipped
        self.data['e_phi'].values = e_phi_flipped
        
        # Recalculate derived components e_co and e_cx
        self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata if needed
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'apply_mars',
                'maximum_radial_extent': maximum_radial_extent
            })

    def swap_polarization_axes(self) -> None:
        """
        Swap vertical and horizontal polarization ports.

        Exchanges the X and Y Ludwig-3 components by converting to the X/Y basis,
        swapping them, then converting back to theta/phi. This is equivalent to
        physically rotating the antenna feed by 90 degrees.

        Note:
            This modifies the pattern in-place.
        """
        # Get data
        phi = self.data.phi.values
        e_theta = self.data.e_theta.values
        e_phi = self.data.e_phi.values

        # Convert to x/y, swap axes, convert back
        e_x, e_y = polarization_tp2xy(phi, e_theta, e_phi)
        e_theta_new, e_phi_new = polarization_xy2tp(phi, e_y, e_x)

        # Update the pattern data
        self.data['e_theta'].values = e_theta_new
        self.data['e_phi'].values = e_phi_new

        # Recompute co/cx polarization
        self.assign_polarization(self.polarization)

        # Clear cache
        self.clear_cache()

        # Update metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'swap_polarization_axes'
            })
        
        # Recalculate derived components e_co and e_cx
        self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata if needed
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'swap_polarization_axes'
            })

    def shift_theta_origin(self, theta_offset: float) -> None:
        """
        Shift the origin of the theta axis of every phi cut (measurement correction).

        This compensates a positioner or mounting offset: the sample that was
        recorded at ``theta`` is moved to ``theta - theta_offset`` on the same
        phi cut, so a pattern whose peak was measured at ``theta = +2`` is
        brought onto boresight with ``theta_offset = 2``. Each cut is shifted
        along its own theta axis, so this is *not* a rigid rotation of the
        antenna; use ``rotate`` for that.

        The shift is performed in central format, where each phi cut is a
        closed great circle: a sided pattern is converted, shifted, and mapped
        back onto its own theta/phi grid. Cuts that span the full circle are
        wrapped periodically, so no data is lost at the ends; partial cuts are
        extended with their end values. Amplitude and unwrapped phase are
        interpolated separately with cubic splines.

        Args:
            theta_offset: Shift in degrees. Positive moves the pattern toward
                negative theta (central) / toward the phi + 180 side (sided).

        Raises:
            NotImplementedError: If the pattern has non-uniform theta grids
        """
        self._require_uniform_theta('shift_theta_origin')

        theta = np.asarray(self.theta_angles, dtype=float)
        is_central = theta.min() < -0.5

        if is_central:
            e_theta, e_phi = self._shift_cuts_along_theta(
                theta, self.data.e_theta.values, self.data.e_phi.values, theta_offset)
        elif np.isclose(theta[0], 0.0, atol=self._ANGLE_TOL):
            # Sided: work on a closed-circle (central) copy, then map back
            # onto this pattern's own grid so the caller's layout is kept.
            work = self.copy()
            work.transform_coordinates('central', _preserve_polarization=True)
            et, ep = self._shift_cuts_along_theta(
                np.asarray(work.theta_angles, dtype=float),
                work.data.e_theta.values, work.data.e_phi.values, theta_offset)
            work.data['e_theta'].values = et.astype(np.complex64)
            work.data['e_phi'].values = ep.astype(np.complex64)
            work.transform_coordinates('sided', _preserve_polarization=True)

            phi_norm = np.mod(np.asarray(self.phi_angles, dtype=float), 360.0)
            phi_norm = np.where(np.isclose(phi_norm, 360.0, atol=self._ANGLE_TOL), 0.0, phi_norm)
            phi_idx = self._match_rows(phi_norm, np.asarray(work.phi_angles, dtype=float))
            theta_idx = self._match_rows(theta, np.asarray(work.theta_angles, dtype=float))
            if np.any(phi_idx < 0) or np.any(theta_idx < 0):
                raise RuntimeError("shift_theta_origin: could not map the shifted pattern "
                                   "back onto the original grid")
            e_theta = work.data.e_theta.values[:, theta_idx, :][:, :, phi_idx]
            e_phi = work.data.e_phi.values[:, theta_idx, :][:, :, phi_idx]
        else:
            logger.warning(
                "shift_theta_origin: sided pattern does not start at theta = 0; "
                "shifting each cut with end-value extension instead of wrapping.")
            e_theta, e_phi = self._shift_cuts_along_theta(
                theta, self.data.e_theta.values, self.data.e_phi.values, theta_offset)

        self.data['e_theta'].values = np.asarray(e_theta, dtype=np.complex64)
        self.data['e_phi'].values = np.asarray(e_phi, dtype=np.complex64)

        self.assign_polarization(self.polarization)
        self.clear_cache()

        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata.setdefault('operations', []).append({
                'type': 'shift_theta_origin',
                'theta_offset': float(theta_offset)
            })

    @classmethod
    def _shift_cuts_along_theta(cls, theta, e_theta, e_phi, theta_offset):
        """
        Resample (frequency, theta, phi) fields so new(theta) = old(theta + offset).

        Cuts that span a full 360 degrees are treated as periodic; otherwise
        the end values are extended. Returns (e_theta, e_phi) as complex128.
        """
        theta = np.asarray(theta, dtype=float)
        n = len(theta)
        span = theta[-1] - theta[0]
        step = span / (n - 1) if n > 1 else 0.0
        duplicate_end = np.isclose(span, 360.0, atol=cls._ANGLE_TOL)
        full_circle = duplicate_end or np.isclose(span + step, 360.0, atol=1e-6)

        base_t = theta[:-1] if duplicate_end else theta
        if full_circle:
            x = np.concatenate([base_t - 360.0, base_t, base_t + 360.0])
        else:
            x = theta

        def resample(field):
            f = np.asarray(field, dtype=np.complex128)
            y = f[:, :-1, :] if duplicate_end else f
            if full_circle:
                y = np.concatenate([y, y, y], axis=1)
            amp = np.abs(y)
            phase = np.unwrap(np.angle(y), axis=1)
            kind = 'cubic' if len(x) >= 4 else 'linear'
            common = dict(kind=kind, axis=1, bounds_error=False, assume_sorted=True)
            amp_i = interp1d(x, amp, fill_value=(amp[:, 0, :], amp[:, -1, :]), **common)
            ph_i = interp1d(x, phase, fill_value=(phase[:, 0, :], phase[:, -1, :]), **common)
            target = theta + theta_offset
            if full_circle:
                # keep the query inside the padded range
                target = x[0] + np.mod(target - x[0], 360.0 * 3)
                target = np.where(target > x[-1], target - 360.0, target)
            return amp_i(target) * np.exp(1j * ph_i(target))

        return resample(e_theta), resample(e_phi)

    def shift_phi_origin(self, phi_offset: float) -> None:
        """
        Rotates the pattern in phi by adding an offset to phi coordinates.

        The phi values are shifted by the offset, wrapped to 0-360, and the data
        is reordered so phi starts at 0 (or the minimum value).

        Args:
            phi_offset: Angle in degrees to add to phi coordinates.
        """
        phi = self.data.phi.values.copy()

        if len(phi) < 2:
            return

        # Add offset and wrap to 0-360
        new_phi = np.mod(phi + phi_offset, 360.0)

        # Find sort indices to put phi back in ascending order
        sort_idx = np.argsort(new_phi)
        sorted_phi = new_phi[sort_idx]

        # Reorder all field components along phi axis (axis=2)
        self.data['e_theta'].values = self.data.e_theta.values[:, :, sort_idx]
        self.data['e_phi'].values = self.data.e_phi.values[:, :, sort_idx]

        if 'e_co' in self.data:
            self.data['e_co'].values = self.data.e_co.values[:, :, sort_idx]
        if 'e_cx' in self.data:
            self.data['e_cx'].values = self.data.e_cx.values[:, :, sort_idx]

        # Update phi coordinates
        self.data = self.data.assign_coords({'phi': sorted_phi})

        # Clear cache
        self.clear_cache()

        # Update metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'shift_phi_origin',
                'phi_offset': float(phi_offset)
            })

    def subsample(self,
                  theta_range: Optional[Tuple[float, float]] = None,
                  theta_step: Optional[float] = None,
                  phi_range: Optional[Tuple[float, float]] = None,
                  phi_step: Optional[float] = None) -> 'FarFieldSpherical':
        """
        Create a subsampled version of the pattern by selecting nearest available points.

        Args:
            theta_range: Optional (min, max) theta range in degrees. If None, use full range.
            theta_step: Optional theta step size in degrees. If None, use original spacing.
            phi_range: Optional (min, max) phi range in degrees. If None, use full range.
            phi_step: Optional phi step size in degrees. If None, use original spacing.

        Returns:
            FarFieldSpherical: New pattern with reduced resolution

        Example:
            # Reduce to theta -150:2:150, phi 0:15:360
            reduced = pattern.subsample(
                theta_range=(-150, 150),
                theta_step=2.0,
                phi_range=(0, 360),
                phi_step=15.0
            )
        """
        self._require_uniform_theta('subsample')

        # Get original coordinates
        orig_theta = self.theta_angles
        orig_phi = self.phi_angles
        orig_freq = self.frequencies
        
        # Determine target theta array
        if theta_range is None:
            theta_min, theta_max = orig_theta.min(), orig_theta.max()
        else:
            theta_min, theta_max = theta_range
            # Validate range is within original data
            if theta_min < orig_theta.min() or theta_max > orig_theta.max():
                available_range = (orig_theta.min(), orig_theta.max())
                raise ValueError(f"Requested theta range {theta_range} exceeds available range {available_range}")
        
        if theta_step is None:
            # Use original theta angles within the range
            target_theta = orig_theta[(orig_theta >= theta_min) & (orig_theta <= theta_max)]
        else:
            # Create new theta array with specified step
            target_theta = np.arange(theta_min, theta_max + theta_step/2, theta_step)
        
        # Determine target phi array
        if phi_range is None:
            phi_min, phi_max = orig_phi.min(), orig_phi.max()
        else:
            phi_min, phi_max = phi_range
            # Validate range is within original data (with wraparound consideration)
            if phi_max - phi_min > 360:
                raise ValueError("Phi range cannot exceed 360 degrees")
        
        if phi_step is None:
            # Use original phi angles within the range
            if phi_range is None:
                target_phi = orig_phi
            else:
                # Handle wraparound for phi angles
                if phi_min < orig_phi.min() or phi_max > orig_phi.max():
                    # Check if range wraps around 0/360
                    phi_wrapped = orig_phi.copy()
                    if phi_min < 0:
                        phi_wrapped = np.concatenate([phi_wrapped - 360, phi_wrapped])
                    if phi_max > 360:
                        phi_wrapped = np.concatenate([phi_wrapped, phi_wrapped + 360])
                    
                    target_phi = phi_wrapped[(phi_wrapped >= phi_min) & (phi_wrapped <= phi_max)]
                    target_phi = np.mod(target_phi, 360)  # Normalize back to 0-360
                    target_phi = np.unique(target_phi)  # Remove duplicates
                else:
                    target_phi = orig_phi[(orig_phi >= phi_min) & (orig_phi <= phi_max)]
        else:
            # Create new phi array with specified step
            target_phi = np.arange(phi_min, phi_max + phi_step/2, phi_step)
            target_phi = np.mod(target_phi, 360)  # Normalize to 0-360
        
        # Find nearest indices for each target angle
        theta_indices = []
        actual_theta = []
        for target_t in target_theta:
            _, idx = find_nearest(orig_theta, target_t)
            theta_indices.append(idx)
            actual_theta.append(orig_theta[idx])
        
        phi_indices = []
        actual_phi = []
        for target_p in target_phi:
            # Handle wraparound for phi
            phi_diffs = np.abs(orig_phi - target_p)
            phi_diffs_wrapped = np.minimum(phi_diffs, 360 - phi_diffs)
            idx = np.argmin(phi_diffs_wrapped)
            phi_indices.append(idx)
            actual_phi.append(orig_phi[idx])
        
        # Convert to numpy arrays
        theta_indices = np.array(theta_indices)
        phi_indices = np.array(phi_indices)
        actual_theta = np.array(actual_theta)
        actual_phi = np.array(actual_phi)
        
        # Extract data using fancy indexing
        # Create index grids for 3D array indexing [freq, theta, phi]
        freq_grid = np.arange(len(orig_freq))[:, np.newaxis, np.newaxis]
        theta_grid = theta_indices[np.newaxis, :, np.newaxis]
        phi_grid = phi_indices[np.newaxis, np.newaxis, :]
        
        # Extract field components
        new_e_theta = self.data.e_theta.values[freq_grid, theta_grid, phi_grid]
        new_e_phi = self.data.e_phi.values[freq_grid, theta_grid, phi_grid]
        
        # Create new xarray Dataset
        new_data = xr.Dataset(
            data_vars={
                'e_theta': (('frequency', 'theta', 'phi'), new_e_theta),
                'e_phi': (('frequency', 'theta', 'phi'), new_e_phi),
            },
            coords={
                'theta': actual_theta,
                'phi': actual_phi,
                'frequency': orig_freq,
            }
        )
        
        # Create new FarFieldSpherical instance using the same type as self
        new_pattern = type(self).__new__(type(self))
        new_pattern.data = new_data
        new_pattern.polarization = self.polarization
        
        # Copy metadata and add operation record
        if hasattr(self, 'metadata') and self.metadata is not None:
            new_pattern.metadata = self.metadata.copy()
        else:
            new_pattern.metadata = {}
        
        if 'operations' not in new_pattern.metadata:
            new_pattern.metadata['operations'] = []
        
        new_pattern.metadata['operations'].append({
            'type': 'subsample',
            'theta_range': theta_range,
            'theta_step': theta_step,
            'phi_range': phi_range,
            'phi_step': phi_step,
            'original_shape': [len(orig_freq), len(orig_theta), len(orig_phi)],
            'new_shape': [len(orig_freq), len(actual_theta), len(actual_phi)]
        })
        
        # Initialize derived components
        new_pattern.assign_polarization(self.polarization)
        
        return new_pattern