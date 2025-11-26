"""
Mixin class that contains operations for FarFieldSpherical objects.
This class is designed to be mixed into the FarFieldSpherical class.
"""

import numpy as np
import xarray as xr
from typing import Tuple, Union, Optional, List, Any, Callable
from scipy.interpolate import interp1d

from .utilities import lightspeed, frequency_to_wavelength, find_nearest
from .polarization import polarization_tp2xy, polarization_tp2rl, polarization_xy2tp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .farfield import FarFieldSpherical

class FarFieldOperationsMixin:
    """Mixin class providing operations for far-field patterns."""
    
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
        
        # Get reference angle values (for logging)
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

    def rotate(self, alpha: float, beta: float, gamma: float) -> None:
        """
        Rotate the pattern using Euler angles (isometric rotation).
        
        Args:
            alpha: First rotation angle in degrees (z-axis)
            beta: Second rotation angle in degrees (y-axis)  
            gamma: Third rotation angle in degrees (z-axis)
            
        Note:
            This modifies the pattern in-place.
        """
        from .pattern_operations import isometric_rotation
        
        # Apply rotation to each frequency
        for freq_idx in range(len(self.frequencies)):
            e_theta_rot, e_phi_rot = isometric_rotation(
                self.theta_angles,
                self.phi_angles,
                self.data.e_theta.values[freq_idx],
                self.data.e_phi.values[freq_idx],
                alpha, beta, gamma
            )
            
            self.data.e_theta.values[freq_idx] = e_theta_rot
            self.data.e_phi.values[freq_idx] = e_phi_rot
        
        # Recompute co/cx polarization
        self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'rotate',
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma
            })

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
        
        This function reflects the pattern data across the theta=0 plane,
        creating a symmetric pattern. Useful for completing partial patterns
        from measurements.
        
        Raises:
            ValueError: If pattern doesn't include theta=0
        """
        # Check if theta=0 exists
        if 0 not in self.theta_angles:
            raise ValueError("Pattern must include theta=0 to mirror")
        
        # Find theta=0 index
        theta_zero_idx = np.where(self.theta_angles == 0)[0][0]
        
        # Get positive and negative theta regions
        positive_theta_mask = self.theta_angles >= 0
        negative_theta_mask = self.theta_angles < 0
        
        # Mirror positive side to negative side
        for freq_idx in range(len(self.frequencies)):
            # E_theta changes sign when mirroring
            self.data.e_theta.values[freq_idx, negative_theta_mask, :] = \
                -self.data.e_theta.values[freq_idx, positive_theta_mask[::-1], :]
            
            # E_phi stays the same
            self.data.e_phi.values[freq_idx, negative_theta_mask, :] = \
                self.data.e_phi.values[freq_idx, positive_theta_mask[::-1], :]
        
        # Recompute co/cx polarization
        self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'mirror_pattern'
            })

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
        # Interpolate complex fields
        e_theta_interp = np.zeros((len(new_frequencies), len(self.theta_angles), 
                                   len(self.phi_angles)), dtype=complex)
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
    
    def transform_coordinates(self, format: str = 'sided') -> None:
        """
        Transform pattern coordinates to conform to a specified theta/phi convention.
        
        This function rearranges the existing pattern data to match one of two standard
        coordinate conventions without interpolation:
        
        - 'sided': theta 0:180, phi 0:360 (spherical convention)
        - 'central': theta -180:180, phi 0:180 (more common for antenna patterns)
        
        Args:
            format: Target coordinate format ('sided' or 'central')
                
        Raises:
            ValueError: If format is not 'sided' or 'central'
        """
        
        if format not in ['sided', 'central']:
            raise ValueError("Format must be 'sided' or 'central'")
        
        # Get current coordinates
        theta = self.theta_angles
        phi = self.phi_angles
        
        # Get field components
        e_theta = self.data.e_theta.values.copy()
        e_phi = self.data.e_phi.values.copy()
        frequencies = self.frequencies
        
        # NORMALIZE PHI to 0-360 range first
        if np.any(phi < 0):
            # Phi has negative values, normalize to 0-360
            phi_normalized = np.mod(phi, 360)
            
            # Sort to maintain ascending order
            sort_indices = np.argsort(phi_normalized)
            phi = phi_normalized[sort_indices]
            
            # Reorder field components along phi axis
            e_theta = e_theta[:, :, sort_indices]
            e_phi = e_phi[:, :, sort_indices]

        # Check current format
        theta_min, theta_max = np.min(theta), np.max(theta)
        phi_min, phi_max = np.min(phi), np.max(phi)
        
        is_sided = theta_min >= 0 and theta_max <= 180 and phi_min >= 0 and phi_max <= 360
        is_central = theta_min >= -180 and theta_max <= 180 and phi_min >= 0 and phi_max <= 180
        
        # If already in the correct format, return
        if (format == 'sided' and is_sided) or (format == 'central' and is_central):
            return
        
        # Apply transformation based on target format
        if format == 'sided':
            # Target: theta 0:180, phi 0:360
            
            # Ensure phi is within central range
            if np.max(phi) >= 180:
                raise ValueError("Input phi must be less than 180 when transforming to sided")
            
            # Find theta = 0 index
            theta0_idx = np.argmin(np.abs(theta))
            
            # Create new theta vector (positive values only)
            new_theta = theta[theta0_idx:]
            
            # Create new phi vector
            new_phi = np.concatenate((phi, phi+180))
            
            # Initialize new electric field arrays
            new_e_theta = np.zeros((frequencies.size, len(new_theta), len(new_phi)), dtype=np.complex128)
            new_e_phi = np.zeros((frequencies.size, len(new_theta), len(new_phi)), dtype=np.complex128)
            
            # Fill new electric field arrays
            # First half of phi range - copy from positive theta
            new_e_theta[:, :, :len(phi)] = e_theta[:, theta0_idx:, :]
            new_e_phi[:, :, :len(phi)] = e_phi[:, theta0_idx:, :]
            
            # Second half of phi range - copy from negative theta (flipped)
            if theta0_idx > 0:  # Only if we have negative theta values
                # Get the flipped negative theta section
                flipped_e_theta = -np.flip(e_theta[:, :theta0_idx, :], axis=1)
                flipped_e_phi = -np.flip(e_phi[:, :theta0_idx, :], axis=1)
                
                # Calculate how many values to copy (minimum of available and needed)
                n_values = min(flipped_e_theta.shape[1], new_e_theta.shape[1])
                
                # Assign only as many values as will fit
                new_e_theta[:, :n_values, len(phi):] = flipped_e_theta[:, :n_values, :]
                new_e_phi[:, :n_values, len(phi):] = flipped_e_phi[:, :n_values, :]
                
                # If we didn't fill all values, fill the rest with the last value
                if n_values < new_e_theta.shape[1]:
                    if n_values > 0:  # Make sure we have at least one value to repeat
                        for i in range(n_values, new_e_theta.shape[1]):
                            new_e_theta[:, i, len(phi):] = flipped_e_theta[:, n_values-1, :]
                            new_e_phi[:, i, len(phi):] = flipped_e_phi[:, n_values-1, :]
                    else:
                        # No negative values to use, fill with zeros
                        new_e_theta[:, n_values:, len(phi):] = 0
                        new_e_phi[:, n_values:, len(phi):] = 0
    
        elif format == 'central':
            # Target: theta -180:180, phi 0:180
            
            # Ensure theta starts at 0
            if theta[0] != 0:
                raise ValueError("Input theta must start at 0 when transforming to central")
            
            # Generate new theta array
            new_theta = np.concatenate((-np.flip(theta[1:]), theta))
            
            # Find phi 180 crossing index
            phi180_idx = np.searchsorted(phi, 180, side='left')
            
            # Generate new phi vector (only 0-180)
            new_phi = phi[:phi180_idx]
            if len(new_phi) == 0:
                # If no phi values are < 180, use all phi values
                new_phi = phi
            
            # Initialize new electric field arrays
            new_e_theta = np.zeros((frequencies.size, len(new_theta), len(new_phi)), dtype=np.complex128)
            new_e_phi = np.zeros((frequencies.size, len(new_theta), len(new_phi)), dtype=np.complex128)
            
            # Fill new electric field arrays
            # Positive theta section (original data)
            new_e_theta[:, len(theta)-1:, :] = e_theta[:, :, :len(new_phi)]
            new_e_phi[:, len(theta)-1:, :] = e_phi[:, :, :len(new_phi)]
            
            # Negative theta section (with phi+180 from original data)
            if phi180_idx < len(phi):  # Only if we have phi values >= 180
                # Extract the high phi section - only use as many as we have in new_phi
                phi_high_indices = np.arange(phi180_idx, min(len(phi), phi180_idx + len(new_phi)))
                
                if len(phi_high_indices) > 0:
                    n_neg_theta = len(theta) - 1  # Number of negative theta values
                    n_phi_high = len(phi_high_indices)  # Number of high phi values
                    n_phi_to_use = min(n_phi_high, len(new_phi))  # Number of phi values to use
                    
                    # Flip the theta axis for the negative theta values
                    flipped_e_theta = -np.flip(e_theta[:, 1:, phi_high_indices], axis=1)
                    flipped_e_phi = -np.flip(e_phi[:, 1:, phi_high_indices], axis=1)
                    
                    # Only use as many phi values as we have in the output
                    new_e_theta[:, :n_neg_theta, :n_phi_to_use] = flipped_e_theta[:, :, :n_phi_to_use]
                    new_e_phi[:, :n_neg_theta, :n_phi_to_use] = flipped_e_phi[:, :, :n_phi_to_use]
        
        # Now create a completely new Dataset with the new coordinates and data
        new_data = xr.Dataset(
            data_vars={
                'e_theta': (('frequency', 'theta', 'phi'), new_e_theta), 
                'e_phi': (('frequency', 'theta', 'phi'), new_e_phi), 
            },
            coords={
                'theta': new_theta,
                'phi': new_phi,
                'frequency': frequencies,
            }
        )
        
        # Replace the pattern's data with the new dataset
        self.data = new_data
        
        # Recalculate derived components e_co and e_cx
        self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata if needed
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'transform_coordinates',
                'format': format,
                'old_theta_range': [float(theta_min), float(theta_max)],
                'old_phi_range': [float(phi_min), float(phi_max)],
                'new_theta_range': [float(np.min(new_theta)), float(np.max(new_theta))],
                'new_phi_range': [float(np.min(new_phi)), float(np.max(new_phi))]
            })

    def normalize_at_boresight(self) -> None:
        """
        Normalize the pattern at boresight using Ludwig's III (e_x, e_y) components.
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
            
            # Calculate average magnitude at boresight
            e_x_avg_mag = np.mean(np.abs(e_x_boresight))
            e_y_avg_mag = np.mean(np.abs(e_y_boresight))
            
            # Get reference phase from first phi cut
            e_x_ref_phase = np.angle(e_x_boresight[0])
            e_y_ref_phase = np.angle(e_y_boresight[0])
            
            # Normalize each phi cut
            for p_idx in range(len(phi)):
                # Create reference values (same phase across all phi)
                e_x_ref = e_x_avg_mag * np.exp(1j * e_x_ref_phase)
                e_y_ref = e_y_avg_mag * np.exp(1j * e_y_ref_phase)
                
                # Calculate correction factors
                e_x_correction = e_x_ref / e_x_boresight[p_idx]
                e_y_correction = e_y_ref / e_y_boresight[p_idx]
                
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
                    np.trapz(
                        (e_theta[f_idx, :, :].transpose() * exp_terms[n_idx, :]).transpose(),
                        theta_rad, axis=0
                    )
                )
                
                # Compute mode coefficient for phi component
                CMC_2 = (
                    -1j * ((-1j) ** (-n)) / (4 * np.pi * wavenumber) *
                    np.trapz(
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
        """
        
        # Get data
        phi = self.data.phi.values
        e_theta = self.data.e_theta.values 
        e_phi = self.data.e_phi.values
        
        # Convert to x/y and back to swap the axes
        e_x, e_y = polarization_tp2xy(phi, e_theta, e_phi)
        e_theta_new, e_phi_new = polarization_xy2tp(phi, e_y, e_x)  # Note: x and y are swapped
        
        # Update the pattern data directly
        self.data['e_theta'].values = e_theta_new
        self.data['e_phi'].values = e_phi_new
        
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
        Shifts the origin of the theta coordinate axis for all phi cuts.
        
        This is useful for aligning measurement data when the mechanical 
        antenna rotation axis doesn't align with the desired coordinate 
        system (e.g., antenna boresight).
        
        The function preserves the original theta grid while shifting 
        the pattern data through interpolation along each phi cut.
        
        Args:
            theta_offset: Angle in degrees to shift the theta origin.
                         Positive values move theta=0 to the right (positive theta),
                         negative values move theta=0 to the left (negative theta).
                         
        Notes:
            - This performs interpolation along the theta axis for each phi cut
            - Complex field components are interpolated separately for amplitude and phase
              to avoid interpolation issues with complex numbers
            - Phase discontinuities are handled by unwrapping before interpolation
        """
        # Get underlying numpy arrays
        frequency = self.data.frequency.values
        theta = self.data.theta.values
        phi = self.data.phi.values
        e_theta = self.data.e_theta.values.copy()
        e_phi = self.data.e_phi.values.copy()
        
        # Create shifted theta array for original data position
        # Positive theta_offset means data shifts left, grid points stay the same
        shifted_theta = theta - theta_offset
        
        # Initialize output arrays with same shape as input
        e_theta_new = np.zeros_like(e_theta, dtype=complex)
        e_phi_new = np.zeros_like(e_phi, dtype=complex)
        
        # Process each frequency and phi cut separately
        for f_idx in range(len(frequency)):
            for p_idx in range(len(phi)):
                # For each component, separate amplitude and phase for interpolation
                
                # Process e_theta
                amp_theta = np.abs(e_theta[f_idx, :, p_idx])
                phase_theta = np.unwrap(np.angle(e_theta[f_idx, :, p_idx]))
                
                # Create interpolation functions for amplitude and phase
                amp_interp_theta = interp1d(
                    shifted_theta, 
                    amp_theta, 
                    kind='cubic', 
                    bounds_error=False, 
                    fill_value=(amp_theta[0], amp_theta[-1])
                )
                
                phase_interp_theta = interp1d(
                    shifted_theta, 
                    phase_theta, 
                    kind='cubic', 
                    bounds_error=False, 
                    fill_value=(phase_theta[0], phase_theta[-1])
                )
                
                # Interpolate onto original grid
                amp_new_theta = amp_interp_theta(theta)
                phase_new_theta = phase_interp_theta(theta)
                
                # Combine amplitude and phase back to complex
                e_theta_new[f_idx, :, p_idx] = amp_new_theta * np.exp(1j * phase_new_theta)
                
                # Process e_phi
                amp_phi = np.abs(e_phi[f_idx, :, p_idx])
                phase_phi = np.unwrap(np.angle(e_phi[f_idx, :, p_idx]))
                
                # Create interpolation functions for amplitude and phase
                amp_interp_phi = interp1d(
                    shifted_theta, 
                    amp_phi, 
                    kind='cubic', 
                    bounds_error=False, 
                    fill_value=(amp_phi[0], amp_phi[-1])
                )
                
                phase_interp_phi = interp1d(
                    shifted_theta, 
                    phase_phi, 
                    kind='cubic', 
                    bounds_error=False, 
                    fill_value=(phase_phi[0], phase_phi[-1])
                )
                
                # Interpolate onto original grid
                amp_new_phi = amp_interp_phi(theta)
                phase_new_phi = phase_interp_phi(theta)
                
                # Combine amplitude and phase back to complex
                e_phi_new[f_idx, :, p_idx] = amp_new_phi * np.exp(1j * phase_new_phi)
        
        # Update the pattern data
        self.data['e_theta'].values = e_theta_new
        self.data['e_phi'].values = e_phi_new
        
        # Recalculate derived components e_co and e_cx
        self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata if needed
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'shift_theta_origin',
                'theta_offset': float(theta_offset)
            })

    def shift_phi_origin(self, phi_offset: float) -> None:
        """
        Shifts the origin of the phi coordinate axis for the pattern by adding the rotation
        to all phi values while maintaining the original phi range and sorting order.
        
        Args:
            phi_offset: Angle in degrees to shift the phi origin.
                   Positive values rotate phi clockwise,
                   negative values rotate phi counterclockwise.
        """
        # Get theta array
        theta = self.data.theta.values
        
        # Check if we have a central or sided pattern
        is_central = np.any(theta < 0)
        
        # If central, first convert to sided
        if is_central:
            # Transform to sided coordinates in place
            self.transform_coordinates('sided')
            
            # Update theta array
            theta = self.data.theta.values

        # Get underlying numpy arrays
        phi = self.data.phi.values
        e_theta = self.data.e_theta.values.copy()
        e_phi = self.data.e_phi.values.copy()
        
        # Apply the phi offset to all phi values
        new_phi = phi + phi_offset
        
        # For a proper coordinate system, phi should always be normalized to 0-360
        # regardless of the original phi range
        while np.any(new_phi < 0):
            new_phi[new_phi < 0] += 360
        while np.any(new_phi >= 360):
            new_phi[new_phi >= 360] -= 360
        
        # Sort indices to maintain ascending order
        sort_indices = np.argsort(new_phi)
        
        # Reorder phi and field components
        sorted_phi = new_phi[sort_indices]
        sorted_e_theta = e_theta[:, :, sort_indices]
        sorted_e_phi = e_phi[:, :, sort_indices]
        
        # Update the pattern data
        self.data = self.data.assign_coords({
            'phi': sorted_phi
        })
        self.data['e_theta'].values = sorted_e_theta
        self.data['e_phi'].values = sorted_e_phi
        
        # Recalculate derived components
        self.assign_polarization(self.polarization)

        # If originally central, transform back
        if is_central:
            self.transform_coordinates('central')
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata if needed
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
            AntennaPattern: New pattern with reduced resolution
            
        Example:
            # Reduce to theta -150:2:150, phi 0:15:360
            reduced = pattern.subsample(
                theta_range=(-150, 150), 
                theta_step=2.0,
                phi_range=(0, 360),
                phi_step=15.0
            )
        """
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
        
        # Create new AntennaPattern instance using the same type as self
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