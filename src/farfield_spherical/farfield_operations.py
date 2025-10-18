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

    def translate(self, translation: np.ndarray, normalize: bool = True) -> None:
        """
        Shifts the antenna phase pattern to place the origin at the location defined by the shift.
        
        This applies a linear phase shift to the pattern corresponding to a translation
        of the phase center. The phase shift is frequency-dependent.
        
        Args:
            translation: 3D translation vector [x, y, z] in meters
            normalize: If True, normalize the boresight gain to 0 dB after translation
            
        Note:
            This modifies the pattern in-place.
        """
        from .pattern_operations import phase_pattern_translate
        
        # Apply translation to each frequency
        for freq_idx, freq in enumerate(self.frequencies):
            wavelength = lightspeed / freq
            k = 2 * np.pi / wavelength
            
            # Apply phase shift
            self.data.e_theta.values[freq_idx] = phase_pattern_translate(
                self.theta_angles,
                self.phi_angles,
                self.data.e_theta.values[freq_idx],
                translation,
                k
            )
            
            self.data.e_phi.values[freq_idx] = phase_pattern_translate(
                self.theta_angles,
                self.phi_angles,
                self.data.e_phi.values[freq_idx],
                translation,
                k
            )
        
        # Recompute co/cx polarization
        self.assign_polarization(self.polarization)
        
        # Normalize if requested
        if normalize:
            # Find boresight (theta=0)
            theta_idx = np.argmin(np.abs(self.theta_angles))
            boresight_gain = np.abs(self.data.e_co.values[:, theta_idx, 0])**2
            
            # Normalize each frequency
            for freq_idx in range(len(self.frequencies)):
                norm_factor = np.sqrt(boresight_gain[freq_idx])
                if norm_factor > 0:
                    self.data.e_theta.values[freq_idx] /= norm_factor
                    self.data.e_phi.values[freq_idx] /= norm_factor
            
            # Recompute co/cx after normalization
            self.assign_polarization(self.polarization)
        
        # Clear cache
        self.clear_cache()
        
        # Update metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'translate',
                'translation': translation.tolist(),
                'normalize': normalize
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