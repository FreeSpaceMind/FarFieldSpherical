# FarFieldSpherical

A Python library for working with antenna far-field patterns in spherical coordinates.

## Overview

**FarFieldSpherical** provides a comprehensive framework for representing, manipulating, and analyzing antenna far-field radiation patterns. The library handles patterns in spherical coordinates (theta, phi) with full support for complex field values, multiple frequencies, and various polarization types.

> **Note:** This package was formerly known as `antenna_pattern` with the main class named `AntennaPattern`. It has been renamed to `FarFieldSpherical` for better descriptive clarity.

## Features

- **Comprehensive Pattern Representation**: Store and manipulate far-field patterns with theta/phi angular coverage and multiple frequencies
- **Polarization Support**: Full support for linear (X/Y), circular (RHCP/LHCP), and spherical (theta/phi) polarizations with automatic conversions
- **File I/O**: Read and write patterns in multiple formats:
  - GRASP .cut files
  - NSI .ffd files
  - TICRA .sph files (spherical wave expansion)
  - ATAMS measurement files
  - NumPy .npz format (efficient native format)
- **Pattern Operations**: 
  - Phase center translation
  - Amplitude scaling
  - Isometric rotation
  - Pattern mirroring
  - Frequency interpolation
- **Analysis Tools**:
  - Phase center calculation
  - Axial ratio computation
  - Directivity calculation
  - Gain and phase extraction
- **Pattern Arithmetic**: Average and difference multiple patterns
- **Spherical Wave Expansion**: Calculate spherical mode coefficients from far-field data

## Installation

### From source
```bash
cd farfield-spherical
pip install -e .
```

### With optional dependencies
```bash
# For spherical wave expansion support
pip install farfield-spherical[swe]

# For development
pip install farfield-spherical[dev]
```

## Quick Start

### Creating a Pattern

```python
import numpy as np
from farfield_spherical import FarFieldSpherical

# Define angular grid
theta = np.linspace(-180, 180, 361)  # degrees
phi = np.linspace(0, 360, 73)        # degrees
frequency = np.array([1.0e9])        # Hz

# Define field components (example: simple pattern)
e_theta = np.ones((1, len(theta), len(phi)), dtype=complex)
e_phi = np.ones((1, len(theta), len(phi)), dtype=complex) * 0.1

# Create pattern object
pattern = FarFieldSpherical(
    theta=theta,
    phi=phi, 
    frequency=frequency,
    e_theta=e_theta,
    e_phi=e_phi,
    polarization='theta'  # or 'x', 'y', 'rhcp', 'lhcp', 'phi'
)

print(f"Pattern has {len(pattern.frequencies)} frequencies")
print(f"Angular coverage: theta=[{pattern.theta_angles.min()}, {pattern.theta_angles.max()}]")
```

### Reading Patterns from Files

```python
from farfield_spherical import read_cut, read_ffd, load_pattern_npz

# Read GRASP .cut file (frequency_start and frequency_end are required)
pattern_cut = read_cut('antenna.cut', frequency_start=1e9, frequency_end=1e9)

# Read NSI .ffd file
pattern_ffd = read_ffd('measurement.ffd')

# Load from NPZ format (returns a (pattern, metadata) tuple)
pattern_npz, metadata = load_pattern_npz('saved_pattern.npz')
```

### Accessing Pattern Data

```python
# Get gain in dB for co-polarized component
gain_db = pattern.get_gain_db('e_co')  # Shape: (frequency, theta, phi)

# Get phase in degrees
phase_deg = pattern.get_phase('e_co', unwrapped=True)

# Get axial ratio
axial_ratio_db = pattern.get_axial_ratio()

# Access raw field data
e_theta_data = pattern.data.e_theta.values
e_phi_data = pattern.data.e_phi.values
```

### Pattern Operations

```python
# Translate pattern (shift phase center)
translation = np.array([0.1, 0.0, 0.0])  # meters [x, y, z]
pattern.translate(translation)

# Change polarization
pattern.change_polarization('rhcp')

# Rotate pattern (rigid rotation of the antenna, degrees; +alpha tilts boresight toward +x, +beta toward +y)
# Boresight moves to theta0 = acos(cos(alpha) cos(beta)), phi0 = atan2(sin(beta), sin(alpha) cos(beta))
pattern.rotate(alpha=10, beta=20, gamma=0)
pattern.rotate(alpha=10, beta=20, gamma=0, method='cubic')  # more accurate on coarse grids

# Scale amplitude
pattern.scale_amplitude(2.0)  # Linear scale factor

# Make a copy before modifications
pattern_copy = pattern.copy()
```

### Working with Multiple Frequencies

```python
# Access single frequency using context manager
with pattern.at_frequency(2.4e9) as single_freq:
    gain = single_freq.get_gain_db('e_co')
    # Work with single frequency pattern
    
# Interpolate to new frequencies
new_freqs = np.linspace(1e9, 2e9, 11)
pattern_interp = pattern.interpolate_frequency(new_freqs)
```

### Pattern Arithmetic

```python
from farfield_spherical import average_patterns, difference_patterns

# Average multiple patterns
patterns = [pattern1, pattern2, pattern3]
averaged = average_patterns(patterns, weights=[0.5, 0.3, 0.2])

# Compute pattern difference (ratio)
diff_pattern = difference_patterns(measured_pattern, simulated_pattern)
```

### Saving Patterns

```python
# Save to NPZ format (recommended for Python workflows)
pattern.save_pattern_npz('output.npz')

# Write to GRASP .cut format
pattern.write_cut('output.cut')

# Write to NSI .ffd format  
pattern.write_ffd('output.ffd')
```

## Polarization Types

The library supports the following polarization conventions:

- **`'theta'`**: Spherical theta component as co-pol
- **`'phi'`**: Spherical phi component as co-pol
- **`'x'` or `'l3x'`**: Ludwig-3 X (horizontal) linear polarization
- **`'y'` or `'l3y'`**: Ludwig-3 Y (vertical) linear polarization
- **`'rhcp'`, `'rh'`, or `'r'`**: Right-hand circular polarization
- **`'lhcp'`, `'lh'`, or `'l'`**: Left-hand circular polarization

Polarization can be changed at any time:
```python
pattern.change_polarization('rhcp')
```

## Analysis Functions

```python
from farfield_spherical import (
    calculate_phase_center,
    principal_plane_phase_center,
    calculate_directivity
)

# Calculate 3D phase center (theta_angle defines beam cone for optimization)
phase_center = calculate_phase_center(pattern, theta_angle=10.0, frequency=1e9)
print(f"Phase center: {phase_center} meters")

# Analytic phase center from three phase measurements on a principal plane
# Args: frequency, theta1, theta2, theta3 (radians), phase1, phase2, phase3 (radians)
import numpy as np
planar, zaxis = principal_plane_phase_center(
    1e9,
    np.radians(-5), np.radians(0), np.radians(5),
    phase1, phase2, phase3
)

# Calculate peak directivity
peak_dir_db, peak_theta, peak_phi = calculate_directivity(pattern, frequency=1e9)
print(f"Peak directivity: {peak_dir_db:.2f} dBi at theta={peak_theta:.1f}, phi={peak_phi:.1f}")
```

### Feed Cross-Polarization Metrics

Feed-level cross-polarization figures over a reflector illumination cone
`0 <= theta <= theta_e`, as defined in *Cross-Polarization Metrics for a
Reflector Feed*. All functions work on `e_co` / `e_cx` as currently assigned
(Ludwig-3 requires polarization `'x'` or `'y'`; anything else logs a warning),
on a copy of the pattern in sided format, and return xarray objects indexed by
frequency.

| Function | Quantity |
|----------|----------|
| `integrated_xpd(pattern, theta_e)` | `XPD_int = 10 log10 [ ∫∫ |E_co|² sinθ dθ dφ / ∫∫ |E_cx|² sinθ dθ dφ ]` over the cone. Screening gate on total cross-polarized power delivered to the reflector. |
| `n0_crosspol_level(pattern, theta_e)` | `L0 = 20 log10 ( max|E_co| / max_θ |c_0(θ)| )`, where `c_0(θ) = (1/2π) ∫ E_cx(θ,φ) dφ` is the azimuthally symmetric part of the cross-pol field. Controls system boresight cross-polarization. |
| `azimuthal_modes(pattern, theta_e, n_max, component)` | Full azimuthal spectrum `c_n(θ) = (1/2π) ∫ E e^{-jnφ} dφ` via FFT along φ, with the power in each `|n|` over the cone (`n ≥ 1` sums the `+n` and `−n` bins) relative to the total power of the component in the cone. |
| `point_xpd(pattern, theta_e)` | `xpd_worst_db`: minimum over the cone of `20 log10(|E_co|/|E_cx|)` at the same angle. `xpol_peak_db`: `20 log10(max|E_co| / max_cone|E_cx|)`. For comparison only. |
| `edge_taper(pattern, theta_e)` | φ-averaged `|E_co|` at the θ sample nearest `theta_e`, relative to peak, in dB. Sanity check on the feed / illumination-angle pairing. |
| `crosspol_report(pattern, theta_e, n_max=6)` | All of the above in one Dataset. Also available as `pattern.crosspol_report(theta_e)`. |
| `check_requirements(report, xpd_int_min_db, n0_min_db, bands_hz)` | Adds per-frequency margins and pass/fail flags, plus `in_band`; out-of-band frequencies are reported but excluded from `attrs['all_pass']`. |

Quadrature is rectangular with weights `sinθ Δθ Δφ`, as in the requirement
text. The φ grid must be uniform and cover a full 360° exactly once
(a duplicated endpoint such as −180/+180 or 0/360 is dropped); θ spacing must
be uniform inside the cone. A `ValueError` is raised otherwise, so a
half-plane measurement (φ 0–180 in sided form) cannot be evaluated.

```python
from farfield_spherical import read_ffd, crosspol_report, check_requirements

pattern = read_ffd("feed.ffd")
report = crosspol_report(pattern, theta_e=35.0)
result = check_requirements(report, xpd_int_min_db=20, n0_min_db=40,
                            bands_hz=[(8e9, 11e9), (13e9, 15e9)])
print(result[["xpd_int_db", "n0_level_db", "in_band"]].to_dataframe())
print("PASS" if result.attrs["all_pass"] else "FAIL")
```

Note that `n0_level_db` on a horn-only, azimuthally symmetric simulation sits
at the simulation's numerical floor (typically 65–80 dB) and is not
representative of assembly performance; the n = 0 requirement is meaningful
for measured feeds or for models that include the asymmetric structure.

## Utilities

```python
from farfield_spherical import (
    frequency_to_wavelength,
    wavelength_to_frequency,
    db_to_linear,
    linear_to_db,
    find_nearest
)

# Convert between frequency and wavelength
wavelength = frequency_to_wavelength(1e9)  # Returns ~0.3 meters
freq = wavelength_to_frequency(0.3)         # Returns ~1e9 Hz

# Convert between dB and linear
linear_value = db_to_linear(20)  # Returns 100
db_value = linear_to_db(100)      # Returns 20

# Find nearest value in array
value, index = find_nearest(pattern.theta_angles, 45.0)
```

## Data Structure

The `FarFieldSpherical` class uses xarray.Dataset internally for efficient storage:

```python
print(pattern.data)
# Output shows:
# Dimensions: (frequency, theta, phi)
# Data variables:
#   e_theta: complex field component
#   e_phi: complex field component  
#   e_co: co-polarized component
#   e_cx: cross-polarized component
```

## Requirements

- Python >= 3.9
- numpy >= 1.21.0
- scipy >= 1.7.0
- xarray >= 0.19.0

Optional:
- swe >= 0.1.0 (for spherical wave expansion)

## Related Packages

- **AntennaPatternViewer**: GUI application for visualizing far-field patterns (separate package)

## License

MIT License - see LICENSE file for details

## Changelog

### Version 1.0.0
- Initial release
- Renamed from `antenna_pattern` to `farfield_spherical`
- Main class renamed from `AntennaPattern` to `FarFieldSpherical`
- Reorganized I/O functions into submodule structure
- Removed GUI components (moved to separate `antenna_pattern_viewer` package)