"""
FarFieldSpherical package - Core functionality for spherical far-field antenna patterns.

This package provides the FarFieldSpherical class and related functions for
working with antenna radiation patterns in spherical coordinates, including 
reading/writing patterns, polarization conversions, and analysis tools.

This package was formerly known as antenna_pattern, with the main class renamed
from AntennaPattern to FarFieldSpherical for better descriptive naming.
"""

__version__ = '1.0.0'
__author__ = 'Justin Long'
__email__ = 'justinwlong1@gmail.com'

# Main class
from .farfield import FarFieldSpherical

# I/O functions
from .io.readers import read_cut, read_ffd, read_ticra_sph
from .io.writers import write_cut, write_ffd, write_ticra_sph
from .io.npz_format import load_pattern_npz, save_pattern_npz

# Polarization functions
from .polarization import (
    polarization_tp2xy,
    polarization_xy2tp,
    polarization_tp2rl,
    polarization_rl2xy,
    polarization_rl2tp
)

# Pattern operations (standalone functions)
from .pattern_operations import (
    unwrap_phase,
    phase_pattern_translate,
    scale_amplitude,
    transform_tp2uvw,
    transform_uvw2tp,
    isometric_rotation
)

# Analysis functions
from .analysis import (
    calculate_phase_center,
    principal_plane_phase_center,
    get_axial_ratio,
    calculate_directivity,
    detect_coordinate_format
)

# Utilities
from .utilities import (
    find_nearest,
    frequency_to_wavelength,
    wavelength_to_frequency,
    lightspeed,
    freespace_permittivity,
    freespace_impedance,
    db_to_linear,
    linear_to_db,
    interpolate_crossing
)

# Package-level functions
from .package_functions import (
    average_patterns,
    difference_patterns
)

__all__ = [
    'FarFieldSpherical',
    'read_cut',
    'read_ffd',
    'read_ticra_sph',
    'write_cut',
    'write_ffd',
    'write_ticra_sph',
    'load_pattern_npz',
    'save_pattern_npz',
    'polarization_tp2xy',
    'polarization_xy2tp',
    'polarization_tp2rl',
    'polarization_rl2xy',
    'polarization_rl2tp',
    'unwrap_phase',
    'phase_pattern_translate',
    'scale_amplitude',
    'transform_tp2uvw',
    'transform_uvw2tp',
    'isometric_rotation',
    'calculate_phase_center',
    'principal_plane_phase_center',
    'get_axial_ratio',
    'calculate_directivity',
    'detect_coordinate_format',
    'find_nearest',
    'frequency_to_wavelength',
    'wavelength_to_frequency',
    'lightspeed',
    'freespace_permittivity',
    'freespace_impedance',
    'db_to_linear',
    'linear_to_db',
    'interpolate_crossing',
    'average_patterns',
    'difference_patterns'
]