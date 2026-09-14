# Pattern Operations Reference

This document describes the mathematical operations available for manipulating antenna far-field patterns in the FarFieldSpherical package.

## Phase Center Translation

Translating the phase reference point of a far-field pattern applies a linear phase gradient across the angular domain. This operation modifies only the phase; the amplitude pattern remains unchanged.

### Mathematical Formulation

Given a translation vector $(x, y, z)$ in meters, the phase shift at each direction $(\theta, \phi)$ is:

$$\Delta\psi(\theta, \phi) = k\left(x\cos\phi\sin\theta + y\sin\phi\sin\theta + z\cos\theta\right)$$

where $k = 2\pi f / c$ is the free-space wavenumber, $f$ is the frequency, and $c$ is the speed of light.

The translated field is:

$$E'(\theta, \phi) = E(\theta, \phi) \cdot e^{j\Delta\psi(\theta, \phi)}$$

### Physical Interpretation

In vector notation, the phase shift is the dot product of the wavenumber vector and the translation:

$$\Delta\psi = \mathbf{k} \cdot \mathbf{d} = k\,\hat{r} \cdot \mathbf{d}$$

where $\hat{r} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$ is the unit vector in the observation direction and $\mathbf{d} = (x, y, z)$ is the translation vector.

This is equivalent to physically moving the antenna by $\mathbf{d}$ and observing the resulting phase change in the far field. A translation along $z$ primarily affects the phase variation with $\theta$, while translations along $x$ or $y$ create $\phi$-dependent phase gradients.

## Phase Center Optimization

The optimal phase center is the point in space from which the far-field radiation appears to originate — the location that minimizes phase variation across the main beam.

### 3D Optimization Algorithm

The implementation uses SciPy's `basinhopping` global optimizer with Nelder-Mead local minimization to find the optimal phase center.

**Cost function:**

$$C(x, y, z) = \text{std}\left[\psi_{unwrapped}(\theta, \phi) \;\big|\; |\theta| \leq \theta_{cone}\right]$$

where $\psi_{unwrapped}$ is the unwrapped phase of the field after applying the translation $(x, y, z)$, and $\theta_{cone}$ defines the angular extent of the main beam region used for optimization.

**Optimizer parameters:**

- **Step size**: $\lambda / 20$ (wavelength-scaled for physical relevance)
- **Bounds**: Maximum displacement of $\pm 2$ meters in each axis
- **Iterations**: Configurable number of basinhopping iterations (default: 10)
- **Temperature**: Controls the acceptance probability for basinhopping's Metropolis criterion

The basinhopping algorithm is used because the phase standard deviation can have local minima, especially for complex antenna patterns. The global optimizer helps escape these local traps.

### Principal Plane Method

An analytical alternative that uses three phase measurements on a principal plane to compute the phase center displacement in that plane.

Given three measured phase values $\psi_1, \psi_2, \psi_3$ at angles $\theta_1, \theta_2, \theta_3$ along a principal plane:

$$d_{planar} = \frac{1}{k} \cdot \frac{(\psi_2 - \psi_1)(\cos\theta_2 - \cos\theta_3) - (\psi_2 - \psi_3)(\cos\theta_2 - \cos\theta_1)}{(\cos\theta_2 - \cos\theta_3)(\sin\theta_2 - \sin\theta_1) - (\cos\theta_2 - \cos\theta_1)(\sin\theta_2 - \sin\theta_3)}$$

This method is fast but only provides the phase center in the plane of the selected cut. It is most useful for quick estimates or when the phase center is known to lie on a principal plane.

## Isometric Rotation

`rotate(alpha, beta, gamma)` rotates the pattern rigidly, equivalent to physically rotating the antenna about the coordinate origin. Both the sampling directions and the field vectors are rotated.

### Rotation Convention

The rotation is parameterized by three angles $(\alpha, \beta, \gamma)$ applied as successive rotations about the coordinate axes:

$$R = R_y(\alpha) \cdot R_x(\beta) \cdot R_z(\gamma)$$

i.e. roll $\gamma$ about $z$ first, then elevation $\beta$ about $x$, then azimuth $\alpha$ about $y$. The individual rotation matrices are:

$$R_z(\gamma) = \begin{pmatrix} \cos\gamma & -\sin\gamma & 0 \\ \sin\gamma & \cos\gamma & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

$$R_x(\beta) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\beta & -\sin\beta \\ 0 & \sin\beta & \cos\beta \end{pmatrix}$$

$$R_y(\alpha) = \begin{pmatrix} \cos\alpha & 0 & -\sin\alpha \\ 0 & 1 & 0 \\ \sin\alpha & 0 & \cos\alpha \end{pmatrix}$$

The same matrices are used by the standalone `isometric_rotation` helper. The original boresight $(+z)$ moves to $R\hat{z}$:

$$\theta_0 = \arccos(\cos\alpha\cos\beta), \qquad \phi_0 = \operatorname{atan2}(-\sin\beta,\; -\sin\alpha\cos\beta)$$

Note the signs: with these matrices a positive $\alpha$ tilts the boresight toward $\phi = 180°$ ($-x$) and a positive $\beta$ toward $\phi = 270°$ ($-y$). Negate the angles for the opposite sense.

### Process

The rotated pattern is evaluated on the pattern's own $(\theta, \phi)$ grid, in its own coordinate format, so the grid and format are preserved.

1. **Source field in Cartesian components.** A sided, $\phi$-normalised copy of the pattern is converted to Cartesian field vectors at every sample, $\mathbf{E} = E_\theta\hat{\theta} + E_\phi\hat{\phi}$. Cartesian components are continuous across the poles, which spherical components are not, so they interpolate cleanly.

2. **Inverse-rotate the target directions.** For every grid direction $\hat{r}'$ of the output, the direction the antenna radiated toward before rotation is $\hat{r} = R^{-1}\hat{r}'$. For central-format grids $\hat{r}'$ is formed directly from the signed $\theta$, so negative $\theta$ needs no special handling.

3. **Interpolate.** The three Cartesian components (real and imaginary parts) are interpolated at $\hat{r}$ with `scipy.interpolate.RegularGridInterpolator` (`method='linear'` by default; `'cubic'` is available and noticeably more accurate on coarse grids). When the $\phi$ grid covers the full circle it is padded periodically so the interpolation wraps across the seam. Directions that fall outside a partial-sphere pattern's coverage are set to zero and a warning is logged.

4. **Rotate the field vector and project.** $\mathbf{E}'(\hat{r}') = R\,\mathbf{E}(\hat{r})$, then $E'_\theta = \mathbf{E}'\cdot\hat{\theta}'$ and $E'_\phi = \mathbf{E}'\cdot\hat{\phi}'$ using the basis at the output direction. Co- and cross-polarised components are recomputed afterwards.

Because the field is interpolated, a rotation is only as accurate as the sampling: at a $1° \times 2°$ grid the linear-interpolation error on a smoothly varying unit-amplitude field is a few $10^{-3}$, and a rotation by a multiple of the $\phi$ step about $z$ is exact. Patterns with a phase centre far from the origin vary quickly in phase between samples and should be translated to their phase centre before rotating.

### Mirroring

`mirror_pattern()` copies the $\theta > 0$ half of every cut of a central-format pattern onto the matching $\theta < 0$ samples with $E_\theta$ negated and $E_\phi$ unchanged. It requires a central-format pattern whose $\theta$ grid includes $0°$ and is symmetric about it.

## MARS (Mathematical Absorber Reflection Suppression)

MARS is a signal processing technique that removes multipath reflections from antenna range measurements using mode filtering in the cylindrical harmonic domain.

### Physical Basis

In a well-designed measurement range, the dominant source of error is reflections from the absorber walls. These reflections create high-spatial-frequency ripple in the measured pattern that does not correspond to physical radiation from the antenna.

The key insight is that an antenna of maximum radial extent $D$ can only support cylindrical harmonic modes up to order:

$$n_{max} = \lfloor k \cdot D \rfloor$$

where $k = 2\pi / \lambda$ is the wavenumber. Modes with $|n| > n_{max}$ are evanescent and cannot represent physical radiation — they must be artifacts of the measurement environment.

### Algorithm

1. **Decompose**: Compute the cylindrical harmonic spectrum of the measured field along the azimuthal ($\phi$) direction using the FFT
2. **Filter**: Retain only modes with $|n| \leq n_{max}$, zeroing all higher-order modes
3. **Reconstruct**: Inverse-transform the filtered spectrum back to the spatial domain

The result is a pattern with multipath ripple suppressed while preserving the physical antenna radiation.

### Parameters

- **Diameter** $D$: The maximum physical extent of the antenna (in meters). This determines $n_{max}$ and thus the aggressiveness of the filtering.
- A larger $D$ retains more modes (less filtering). Setting $D$ too small removes physical content; setting it too large leaves reflections.

## Amplitude Normalization

Three methods are provided for normalizing the pattern amplitude.

### Peak Normalization

Normalizes the pattern so that the peak total field magnitude is $0\;dB$:

$$E'(\theta, \phi) = \frac{E(\theta, \phi)}{\max_{\theta,\phi}\sqrt{|E_{co}|^2 + |E_{cx}|^2}}$$

After normalization, all field values are $\leq 0\;dB$. This is useful for comparing patterns of different absolute gain levels.

### Boresight Normalization

Normalizes the pattern to the field value at boresight ($\theta = 0°$, $\phi = 0°$):

$$E'(\theta, \phi) = \frac{E(\theta, \phi)}{|E(\theta_0, \phi_0)|}$$

where $(\theta_0, \phi_0)$ is the grid point closest to boresight. After normalization, the boresight value is $0\;dB$.

### Mean Normalization

Normalizes to the RMS field level:

$$E'(\theta, \phi) = \frac{E(\theta, \phi)}{\sqrt{\langle|E|^2\rangle}}$$

where $\langle|E|^2\rangle$ is the mean squared magnitude over all grid points.

## Phase Normalization

Sets the phase at a reference point to zero, removing an arbitrary phase offset:

$$E'(\theta, \phi) = E(\theta, \phi) \cdot e^{-j\angle E(\theta_{ref}, \phi_{ref})}$$

where $\angle E(\theta_{ref}, \phi_{ref})$ is the phase of the field at the reference direction. This shifts all phase values uniformly, preserving relative phase relationships.

## Boresight Normalization (Per-Cut)

This operation ensures that all phi cuts pass through the same amplitude and phase at boresight ($\theta = 0°$). It corrects for systematic measurement errors that cause cut-to-cut offsets.

### Method

For each phi cut $\phi_i$, the boresight field value $E(\theta=0, \phi_i)$ is extracted. A reference value is computed as the **median** across all cuts:

$$E_{ref} = \text{median}_i\{E(\theta=0, \phi_i)\}$$

The median is used instead of the mean to provide robustness against outlier cuts (e.g., a cut with a measurement glitch at boresight).

Each cut is then scaled:

$$E'(\theta, \phi_i) = E(\theta, \phi_i) \cdot \frac{E_{ref}}{E(\theta=0, \phi_i)}$$

This is a complex-valued scaling that adjusts both amplitude and phase of each cut to match the median boresight value.

## Pattern Mirroring

Mirrors the pattern across the $\theta = 0°$ plane, creating a symmetric pattern:

$$E_\theta(-\theta, \phi) = -E_\theta(\theta, \phi)$$
$$E_\phi(-\theta, \phi) = E_\phi(\theta, \phi)$$

### Why $E_\theta$ is Negated

The unit vector $\hat{\theta}$ reverses direction when $\theta \to -\theta$ (in central coordinates). Specifically, $\hat{\theta}$ points "away from the pole," so at $-\theta$ it points in the opposite direction compared to $+\theta$. To represent the same physical field, the $E_\theta$ component must be negated.

The unit vector $\hat{\phi}$ does not reverse under $\theta \to -\theta$ (it remains tangent to the circle of constant $\theta$ in the same sense), so $E_\phi$ is preserved.

## Frequency Interpolation

Interpolates the pattern data to new frequency points when multi-frequency data is available.

### Method

Uses `scipy.interpolate.interp1d` applied independently to the real and imaginary parts of the complex field:

$$\text{Re}[E'(f')] = \text{interp}\left(\text{Re}[E(f_1)], \text{Re}[E(f_2)], \ldots; f'\right)$$
$$\text{Im}[E'(f')] = \text{interp}\left(\text{Im}[E(f_1)], \text{Im}[E(f_2)], \ldots; f'\right)$$

Interpolating real and imaginary parts separately avoids discontinuity issues that arise when interpolating magnitude and phase directly (phase wrapping) or when interpolating the complex values directly across resonances.

