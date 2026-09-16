"""
Mixin class that contains operations for FarFieldSpherical objects.
This class is designed to be mixed into the FarFieldSpherical class.
"""

import copy

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
        of the phase center. The phase shift is frequency-dependent:

            E'(theta, phi) = E(theta, phi) * exp(-j k (r_hat . d))

        with r_hat the unit direction and d the translation. The direction uses
        the signed theta, so central-format patterns are handled correctly.

        Args:
            translation: 3D translation vector [x, y, z] in meters

        Raises:
            ValueError: If translation is not three numbers
            NotImplementedError: If the pattern has non-uniform theta grids

        Note:
            This modifies the pattern in-place.
            Use normalize_phase() separately if phase normalization is needed.
        """
        self._require_uniform_theta('translate')

        # Validate before touching the data: this used to raise on the metadata
        # record at the end, after the fields had already been modified.
        translation = np.asarray(translation, dtype=float).ravel()
        if translation.size != 3:
            raise ValueError(
                f"translate expects a 3-element [x, y, z] vector in metres, "
                f"got {translation.size} value(s).")

        theta_rad = np.radians(np.asarray(self.theta_angles, dtype=float))[:, None]
        phi_rad = np.radians(np.asarray(self.phi_angles, dtype=float))[None, :]
        sin_theta = np.sin(theta_rad)

        # r_hat . d over the grid, shape (theta, phi)
        path_length = (translation[0] * sin_theta * np.cos(phi_rad)
                       + translation[1] * sin_theta * np.sin(phi_rad)
                       + translation[2] * np.cos(theta_rad) * np.ones_like(phi_rad))

        wavenumber = 2 * np.pi * np.asarray(self.frequencies, dtype=float) / lightspeed
        phase_shift = np.exp(-1j * wavenumber[:, None, None] * path_length[None, :, :])

        # Multiply the complex field directly. The previous implementation took
        # np.angle, shifted it and rebuilt magnitude * exp(j phase) per
        # frequency in a Python loop, which cost precision and discarded the
        # distinction between a true zero and a very small value.
        self.data['e_theta'].values = (self.data.e_theta.values * phase_shift).astype(np.complex64)
        self.data['e_phi'].values = (self.data.e_phi.values * phase_shift).astype(np.complex64)

        # Recompute co/cx polarization
        self.assign_polarization(self.polarization)

        # Clear cache
        self.clear_cache()

        # Update metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata.setdefault('operations', []).append({
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
        self._require_uniform_theta('normalize_phase')

        
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
        Scale the amplitude of the pattern by a constant number of dB.
        
        Args:
            scale_factor: Amplitude scale in dB. The field is multiplied by
                10**(scale_factor/20), so 6.0 is a factor of two in amplitude
                and 0.0 leaves the pattern unchanged.
            
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

        if len(self.frequencies) < 2:
            raise ValueError(
                "interpolate_frequency needs at least two frequencies to "
                f"interpolate between (this pattern has {len(self.frequencies)}).")

        new_frequencies = np.atleast_1d(np.asarray(new_frequencies, dtype=float))
        f_min, f_max = float(np.min(self.frequencies)), float(np.max(self.frequencies))
        if np.any(new_frequencies < f_min) or np.any(new_frequencies > f_max):
            logger.warning(
                "interpolate_frequency: requested frequencies extend outside the "
                "measured band %.4g to %.4g Hz; those values are extrapolated.",
                f_min, f_max)

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

    def normalize_at_boresight(self, weak_component_ratio: float = 0.1) -> None:
        """
        Remove cut-to-cut gain and phase offsets using the boresight sample.

        Every phi cut passes through boresight (theta = 0), which is a single
        physical direction, so the Ludwig-3 components e_x(0, phi) and
        e_y(0, phi) should be identical for all phi. Any spread across cuts at
        boresight is a per-cut measurement offset (a gain or phase drift
        between cuts), and each cut is scaled by one complex factor so that
        its boresight sample lands on a common reference.

        The reference magnitude is the median across cuts and the reference
        phase is the circular (vector) mean of the boresight samples, so a set
        of phases straddling +/-180 degrees is handled correctly; a linear
        median of wrapped angles would put the reference near 0 and rotate the
        whole pattern.

        For each cut the correction is derived from the dominant component,
        the one with the larger median boresight magnitude. A component whose
        median boresight magnitude is below ``weak_component_ratio`` times
        the dominant one (10 %, i.e. 20 dB down, by default) is the cross-pol
        of a linearly polarized antenna and its boresight value is noise;
        deriving a correction from it would divide by that noise and apply an
        arbitrary complex gain to the whole cut. Such a component borrows the
        dominant component's correction instead. When both components are
        significant (dual-pol or circular), each is corrected independently.

        Args:
            weak_component_ratio: Magnitude ratio below which a component
                borrows the other one's correction.
        """
        self._require_uniform_theta('normalize_at_boresight')

        frequency = self.data.frequency.values
        theta = np.asarray(self.data.theta.values, dtype=float)
        phi = np.asarray(self.data.phi.values, dtype=float)
        e_theta = np.asarray(self.data.e_theta.values, dtype=np.complex128)
        e_phi = np.asarray(self.data.e_phi.values, dtype=np.complex128)

        theta0_idx = int(np.argmin(np.abs(theta)))

        def cut_corrections(boresight):
            """One complex factor per cut mapping its boresight sample to the reference."""
            magnitude = np.abs(boresight)
            reference = np.median(magnitude) * np.exp(1j * np.angle(np.sum(boresight)))
            with np.errstate(divide='ignore', invalid='ignore'):
                correction = np.where(magnitude > 1e-30, reference / boresight, 1.0)
            return correction

        for f_idx in range(len(frequency)):
            e_x, e_y = polarization_tp2xy(phi, e_theta[f_idx], e_phi[f_idx])

            x_boresight = e_x[theta0_idx, :]
            y_boresight = e_y[theta0_idx, :]
            x_level = np.median(np.abs(x_boresight))
            y_level = np.median(np.abs(y_boresight))

            x_correction = cut_corrections(x_boresight)
            y_correction = cut_corrections(y_boresight)

            # A weak component's boresight value is noise: borrow the other's.
            dominant = max(x_level, y_level)
            if x_level < weak_component_ratio * dominant:
                x_correction = y_correction
            elif y_level < weak_component_ratio * dominant:
                y_correction = x_correction

            e_x = e_x * x_correction[None, :]
            e_y = e_y * y_correction[None, :]

            e_theta[f_idx], e_phi[f_idx] = polarization_xy2tp(phi, e_x, e_y)

        self.data['e_theta'].values = e_theta.astype(np.complex64)
        self.data['e_phi'].values = e_phi.astype(np.complex64)

        # Recalculate derived components
        self.assign_polarization(self.polarization)

        # Clear cache
        self.clear_cache()

        # Update metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata.setdefault('operations', []).append({
                'type': 'normalize_at_boresight'
            })

    def apply_mars(self, maximum_radial_extent: float, taper: int = 0) -> None:
        """
        Apply Mathematical Absorber Reflection Suppression (MARS).

        Each phi cut is treated as a closed circle in theta and expanded in
        cylindrical modes, i.e. a Fourier series in theta:

            c_n = (1/2pi) int_0^{2pi} E(theta) e^{-j n theta} dtheta
            E_filtered(theta) = sum_n w_n c_n e^{+j n theta}

        An antenna of maximum radial extent D about the measurement origin
        radiates only modes with |n| <= k D, so higher orders can only be
        range reflections and are removed. n_max = floor(k D) per frequency.

        The weights w_n are 1 for |n| <= n_max and 0 beyond n_max + taper.
        With ``taper`` = 0 the filter is a brick wall, which rings (the
        residual of a removed reflection spreads with slowly decaying
        sidelobes in theta). A positive ``taper`` rolls the weights off with a
        raised cosine over the ``taper`` orders above n_max, which trades a
        little extra retained reflection for much less ringing.

        The expansion needs each phi cut to be a closed great circle, so the
        pattern is processed in central format: a sided pattern is converted
        on a copy, filtered, and the result mapped back onto its own grid. The
        theta grid must be uniform with a step that divides 360 degrees.

        A cut that spans less than 360 degrees (a sector such as -100..100
        from a far-field or compact range) is zero-padded to a full circle,
        filtered, and read back on the sector. This is the sector processing
        of far-field MARS. The truncation makes the padded cut discontinuous,
        so the filtered result rings near the sector edges; a warning is
        logged and the outermost samples should be treated with caution.

        For the filter to act on reflections rather than the antenna's own
        pattern, the antenna should first be translated so that the
        measurement origin is the pattern's phase reference; see ``translate``.

        Args:
            maximum_radial_extent: Maximum radial extent of the antenna in
                metres, measured from the origin the pattern is referred to
            taper: Number of mode orders over which the cutoff rolls off with
                a raised cosine. 0 (the default) is a brick-wall filter.

        Raises:
            ValueError: If the extent is not positive, the taper is negative,
                the theta grid is not uniform or its step does not divide 360,
                or a sided pattern's theta does not start at 0
            NotImplementedError: If the pattern has non-uniform theta grids
        """
        self._require_uniform_theta('apply_mars')

        if maximum_radial_extent <= 0:
            raise ValueError("Maximum radial extent must be positive")
        taper = int(taper)
        if taper < 0:
            raise ValueError("taper must be zero or a positive number of modes")

        frequencies = np.asarray(self.frequencies, dtype=float)

        def filter_modes(theta_deg, e_theta, e_phi):
            theta_deg = np.asarray(theta_deg, dtype=float)
            n_samples = len(theta_deg)
            steps = np.diff(theta_deg)
            if n_samples < 4 or not np.allclose(steps, steps[0], atol=1e-6):
                raise ValueError("apply_mars requires a uniform theta grid.")
            step = float(steps[0])
            n_circle = 360.0 / step
            if not np.isclose(n_circle, np.round(n_circle), atol=1e-6):
                raise ValueError(
                    f"apply_mars requires a theta step that divides 360 degrees "
                    f"(found {step:g}).")
            n_circle = int(np.round(n_circle))
            span = theta_deg[-1] - theta_deg[0]
            duplicate_end = np.isclose(span, 360.0, atol=1e-6)
            full_circle = duplicate_end or np.isclose(span + step, 360.0, atol=1e-6)
            if not duplicate_end and span + step > 360.0 + 1e-6:
                raise ValueError(
                    f"apply_mars: theta spans more than one circle ({span:g} degrees).")
            if not full_circle:
                logger.warning(
                    "apply_mars: theta cuts span %g degrees, not a full circle. The "
                    "sector is zero-padded to 360 degrees before filtering (far-field "
                    "MARS sector processing); expect ringing within a few beamwidths "
                    "of the sector edges at theta = %g and %g degrees.",
                    span, theta_deg[0], theta_deg[-1])

            # Samples used for the analysis: the unique samples of a closed
            # circle (a duplicated +/-180 endpoint is dropped and reproduced
            # on synthesis), or every sample of a sector. Samples missing from
            # a sector are zeros on the implied full circle, so they simply do
            # not contribute to the sum.
            n_analysis = n_samples - 1 if duplicate_end else n_samples
            theta_rad = np.radians(theta_deg)
            theta_analysis = theta_rad[:n_analysis]
            d_theta = np.radians(step)

            e_theta_new = np.empty_like(e_theta, dtype=np.complex128)
            e_phi_new = np.empty_like(e_phi, dtype=np.complex128)
            nyquist = n_circle // 2

            for f_idx, f in enumerate(frequencies):
                wavenumber = 2 * np.pi * f / lightspeed
                n_max = int(np.floor(wavenumber * maximum_radial_extent))
                if n_max >= nyquist:
                    logger.warning(
                        "apply_mars: at %.4g Hz the mode limit k*D = %d is at or above "
                        "the theta sampling limit (%d); the filter removes nothing.",
                        f, n_max, nyquist)
                    n_max = nyquist
                n_stop = min(n_max + taper, nyquist)
                orders = np.arange(-n_stop, n_stop + 1)
                weights = self._mars_weights(orders, n_max, n_stop - n_max)

                # Periodic rectangular rule, exact for a band-limited periodic
                # field: analysis A[n, theta] and synthesis B[theta, n].
                analysis = np.exp(-1j * np.outer(orders, theta_analysis)) * (d_theta / (2 * np.pi))
                synthesis = np.exp(1j * np.outer(theta_rad, orders)) * weights[None, :]

                e_theta_new[f_idx] = synthesis @ (analysis @ e_theta[f_idx, :n_analysis, :])
                e_phi_new[f_idx] = synthesis @ (analysis @ e_phi[f_idx, :n_analysis, :])

            return e_theta_new, e_phi_new

        e_theta_new, e_phi_new = self._apply_on_central_grid(filter_modes, 'apply_mars')

        self.data['e_theta'].values = np.asarray(e_theta_new, dtype=np.complex64)
        self.data['e_phi'].values = np.asarray(e_phi_new, dtype=np.complex64)

        # Recalculate derived components e_co and e_cx
        self.assign_polarization(self.polarization)

        # Clear cache
        self.clear_cache()

        # Update metadata if needed
        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata.setdefault('operations', []).append({
                'type': 'apply_mars',
                'maximum_radial_extent': maximum_radial_extent,
                'taper': taper,
            })

    @staticmethod
    def _mars_weights(orders, n_max, taper):
        """
        Mode weights for apply_mars: 1 up to |n| = n_max, a raised-cosine
        roll-off to 0 at |n| = n_max + taper, 0 beyond.
        """
        magnitude = np.abs(orders).astype(float)
        weights = np.where(magnitude <= n_max, 1.0, 0.0)
        if taper > 0:
            in_taper = (magnitude > n_max) & (magnitude <= n_max + taper)
            x = (magnitude[in_taper] - n_max) / (taper + 1)
            weights[in_taper] = 0.5 * (1 + np.cos(np.pi * x))
        return weights

    def swap_polarization_axes(self) -> None:
        """
        Swap the two Ludwig-3 linear polarization components.

        Converts to the Ludwig-3 x/y basis, exchanges e_x and e_y, and converts
        back to theta/phi. Use it when the two linear ports of a measurement
        were recorded under each other's name, so the pattern labelled x is
        the y port and vice versa.

        This is an exchange of axes, not a rotation of the feed: the map
        (e_x, e_y) -> (e_y, e_x) is a reflection about the x = y plane, with
        determinant -1. It is its own inverse (two calls restore the pattern),
        and for a circularly polarized pattern it exchanges the handedness,
        because reflecting the basis reverses the sense of rotation. A rigid
        90 degree rotation of the feed about boresight would instead be
        (e_x, e_y) -> (-e_y, e_x), which preserves handedness; use
        ``rotate(0, 0, 90)`` for that.

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


    def _apply_on_central_grid(self, operation, name: str, fallback=None):
        """
        Evaluate an operation that needs closed theta cuts and write the result
        back on this pattern's own grid.

        ``operation(theta_deg, e_theta, e_phi) -> (e_theta, e_phi)`` receives
        central-format arrays (theta -180..180, every phi cut a closed great
        circle). A central pattern is passed through directly. A sided pattern
        whose theta starts at 0 is converted to central on a copy, operated on,
        converted back and mapped onto the original theta/phi grid by value,
        so the caller's layout is preserved. A sided pattern whose theta does
        not start at 0 cannot be closed; ``fallback`` is used if given,
        otherwise ValueError is raised.

        Returns (e_theta, e_phi) on this pattern's grid, not yet stored.
        """
        theta = np.asarray(self.theta_angles, dtype=float)
        if theta.min() < -0.5:
            return operation(theta, self.data.e_theta.values, self.data.e_phi.values)

        if not np.isclose(theta[0], 0.0, atol=self._ANGLE_TOL):
            if fallback is not None:
                return fallback(theta, self.data.e_theta.values, self.data.e_phi.values)
            raise ValueError(
                f"{name} needs each phi cut to be a closed circle in theta, which "
                f"requires central format or a sided pattern starting at theta = 0 "
                f"(this one starts at {theta[0]:g} degrees).")

        work = self.copy()
        work.transform_coordinates('central', _preserve_polarization=True)
        et, ep = operation(np.asarray(work.theta_angles, dtype=float),
                           work.data.e_theta.values, work.data.e_phi.values)
        work.data['e_theta'].values = np.asarray(et, dtype=np.complex64)
        work.data['e_phi'].values = np.asarray(ep, dtype=np.complex64)
        work.transform_coordinates('sided', _preserve_polarization=True)

        phi_norm = np.mod(np.asarray(self.phi_angles, dtype=float), 360.0)
        phi_norm = np.where(np.isclose(phi_norm, 360.0, atol=self._ANGLE_TOL), 0.0, phi_norm)
        phi_idx = self._match_rows(phi_norm, np.asarray(work.phi_angles, dtype=float))
        theta_idx = self._match_rows(theta, np.asarray(work.theta_angles, dtype=float))
        if np.any(phi_idx < 0) or np.any(theta_idx < 0):
            raise RuntimeError(f"{name}: could not map the result back onto the original grid")
        return (work.data.e_theta.values[:, theta_idx, :][:, :, phi_idx],
                work.data.e_phi.values[:, theta_idx, :][:, :, phi_idx])

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

        def shifted(theta_deg, e_theta, e_phi):
            return self._shift_cuts_along_theta(theta_deg, e_theta, e_phi, theta_offset)

        def extend_ends(theta_deg, e_theta, e_phi):
            logger.warning(
                "shift_theta_origin: sided pattern does not start at theta = 0; "
                "shifting each cut with end-value extension instead of wrapping.")
            return shifted(theta_deg, e_theta, e_phi)

        e_theta, e_phi = self._apply_on_central_grid(
            shifted, 'shift_theta_origin', fallback=extend_ends)

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
        Shift the origin of the phi axis (measurement correction).

        Adds ``phi_offset`` to every phi coordinate, wraps to [0, 360), sorts
        the cuts and merges any that coincide after wrapping. The spherical
        components e_theta / e_phi travel with their cut unchanged, but e_co
        and e_cx are Ludwig-3 components referred to the fixed x/y axes, so
        they are recomputed from the new phi labels.

        This is a measurement correction for a positioner or mounting offset in
        azimuth. It is equivalent to a rigid rotation about the z-axis; use
        ``rotate(0, 0, gamma)`` when you mean to reorient the antenna, so that
        the intent is recorded in the pattern's history.

        Args:
            phi_offset: Angle in degrees to add to the phi coordinates.
        """
        phi = np.asarray(self.data.phi.values, dtype=float)
        if len(phi) == 0:
            return

        e_theta = self.data.e_theta.values
        e_phi = self.data.e_phi.values
        new_phi, (e_theta, e_phi) = self._normalize_phi(phi + phi_offset, e_theta, e_phi)

        self.data = xr.Dataset(
            data_vars={
                'e_theta': (('frequency', 'theta', 'phi'), np.asarray(e_theta, dtype=np.complex64)),
                'e_phi': (('frequency', 'theta', 'phi'), np.asarray(e_phi, dtype=np.complex64)),
            },
            coords={
                'theta': self.data.theta.values,
                'phi': new_phi,
                'frequency': self.frequencies,
            }
        )

        # e_co / e_cx depend on phi, so they must be re-derived, not reordered.
        self.assign_polarization(self.polarization)
        self.clear_cache()

        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata.setdefault('operations', []).append({
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

        # A request such as phi_range=(0, 360) ends on 360, which wraps onto 0
        # and would otherwise leave the phi axis non-monotonic and duplicated.
        target_phi = np.unique(np.mod(target_phi, 360.0))
        
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
        
        # Build a fully initialised instance (going through __init__ so that
        # _theta_grid, the cache and the co/cross components all exist).
        new_pattern = type(self)(
            theta=actual_theta,
            phi=actual_phi,
            frequency=orig_freq,
            e_theta=new_e_theta,
            e_phi=new_e_phi,
            polarization=self.polarization,
            metadata=copy.deepcopy(self.metadata) if self.metadata else None,
        )
        if new_pattern.metadata is None:
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