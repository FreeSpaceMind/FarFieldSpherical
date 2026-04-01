# Coordinate Systems & Conventions

## Spherical Coordinates

The FarFieldSpherical package uses the standard physics convention for spherical coordinates:

- $r$ — radial distance from the origin
- $\theta$ — polar angle measured from the $+z$ axis ($0° \leq \theta \leq 180°$)
- $\phi$ — azimuthal angle measured from the $+x$ axis in the $xy$-plane ($0° \leq \phi < 360°$)

The far-field electric field is represented by its two transverse components $E_\theta$ and $E_\phi$, stored as complex values (magnitude and phase) at each $(\theta, \phi)$ grid point.

## Coordinate Formats

The package works with two angular conventions for representing antenna pattern data. Both describe the same physical fields on the sphere, but organize the angular grid differently.

### Sided Format (Standard Spherical)

- $\theta \in [0°, 180°]$, $\phi \in [0°, 360°]$
- $\theta = 0°$ corresponds to the $+z$ axis (boresight for a $z$-pointing antenna)
- $\theta = 180°$ is the back lobe direction ($-z$ axis)
- This is the standard format used by file I/O routines (CUT, FFD formats)

In this format, every direction on the sphere is addressed exactly once. The full $360°$ range of $\phi$ is required to cover the sphere.

### Central Format (Antenna Pattern Convention)

- $\theta \in [-180°, 180°]$, $\phi \in [0°, 180°]$
- $\theta = 0°$ is boresight (the $+z$ axis)
- Positive $\theta$ sweeps toward $+x$ (for $\phi = 0°$)
- Negative $\theta$ sweeps toward $-x$ (for $\phi = 0°$), i.e., the back hemisphere
- Only half the $\phi$ range is needed because negative $\theta$ covers the other half

This format is more intuitive for antenna pattern visualization because the pattern is symmetric about boresight ($\theta = 0°$), and phi cuts sweep continuously from $-180°$ to $+180°$ without discontinuity.

### Conversion Between Formats

**Sided to Central:**

The conversion maps data from the $(\theta, \phi)$ grid with $\theta \in [0°, 180°]$ and $\phi \in [0°, 360°]$ onto a grid with $\theta \in [-180°, 180°]$ and $\phi \in [0°, 180°]$.

For the front hemisphere ($\theta \geq 0°$ in central format):
- Data is taken directly from the sided format at the same $(\theta, \phi)$

For the back hemisphere ($\theta < 0°$ in central format):
- The negative-$\theta$ data at azimuth $\phi$ comes from the sided-format point at $(\theta_{sided}, \phi + 180°)$
- Specifically, a central-format point at $(-\theta_0, \phi_0)$ maps to the sided-format point at $(\theta_0, \phi_0 + 180°)$
- The field components must be negated: $E'_\theta = -E_\theta$, $E'_\phi = -E_\phi$

**Why the fields are negated:**

A measurement at direction $(\theta, \phi + 180°)$ in sided format observes the same physical point on the sphere as $(-\theta, \phi)$ in central format. However, the spherical unit vectors $\hat{\theta}$ and $\hat{\phi}$ both reverse sign at the antipodal azimuth:

$$\hat{\theta}(\theta, \phi + 180°) = -\hat{\theta}(-\theta, \phi)$$
$$\hat{\phi}(\theta, \phi + 180°) = -\hat{\phi}(-\theta, \phi)$$

Therefore the field components, which are projections onto these unit vectors, must be negated to represent the same physical electric field.

**Central to Sided:**

The inverse transformation extracts the front hemisphere directly and reconstructs the back hemisphere by reversing the negation and azimuthal offset.

### Format Detection

Format auto-detection examines the minimum value of $\theta$ in the dataset:

- If $\theta_{min} < -0.5°$ then the data is in **central format**
- Otherwise the data is in **sided format**

The $-0.5°$ threshold provides robustness against small numerical offsets while reliably distinguishing the two conventions (central format always has significantly negative $\theta$ values).

## Direction Cosines

Direction cosines $(u, v, w)$ provide an alternative Cartesian representation of directions on the unit sphere:

$$u = \sin\theta \cos\phi, \quad v = \sin\theta \sin\phi, \quad w = \cos\theta$$

These satisfy the constraint $u^2 + v^2 + w^2 = 1$.

The inverse transformation recovers spherical angles:

$$\theta = \arccos(w), \quad \phi = \arctan2(v, u)$$

Direction cosines are particularly useful for rotation operations because rotations in 3D are simple matrix multiplications:

$$\begin{pmatrix} u' \\ v' \\ w' \end{pmatrix} = R \begin{pmatrix} u \\ v \\ w \end{pmatrix}$$

where $R$ is a $3 \times 3$ rotation matrix. This avoids the complications of rotating directly in spherical coordinates, where the transformation is nonlinear.

## Dual Sphere Data

Some antenna measurement systems capture data over $\phi \in [0°, 360°]$ with the antenna rotated between two measurement sweeps, effectively measuring two overlapping half-spheres. This provides redundant coverage that can be used for error analysis or averaging.

### Detection Criteria

A pattern is identified as dual-sphere data when:

1. The $\phi$ range spans approximately $0°$ to $360°$
2. There are equal numbers of phi cuts on each side of $\phi = 180°$
3. The phi grids align: $\phi_2 - 180° \approx \phi_1$ (the second set of cuts mirrors the first, offset by $180°$)

### Splitting into Two Spheres

- **Sphere 1**: Data with $\phi \in [0°, 180°)$ is used directly as a complete half-sphere measurement
- **Sphere 2**: Data with $\phi \in [180°, 360°)$ is processed as follows:
  1. $\phi$ is remapped to $[0°, 180°)$ by subtracting $180°$
  2. Field components are negated: $E'_\theta = -E_\theta$, $E'_\phi = -E_\phi$
  3. The $\theta$ axis is flipped (reversed in order)

### Physics of the Splitting

The field negation and theta flip arise from the same geometric principle as the sided-to-central conversion. A measurement at direction $(\theta, \phi + 180°)$ observes the same physical direction as $(-\theta, \phi)$. Because the spherical unit vectors $\hat{\theta}$ and $\hat{\phi}$ both reverse sign between these two representations, the field components must be negated to preserve the physical electric field vector.

After splitting, each sphere independently covers the $\phi \in [0°, 180°]$ range and can be converted to central format or analyzed separately.

### Important Implementation Note

The `transform_coordinates` function includes a guard condition that checks $\max(\phi) > 185°$ before performing sided-to-central conversion. This prevents the conversion from being applied to dual-sphere data that has not yet been split, which would produce incorrect results.
