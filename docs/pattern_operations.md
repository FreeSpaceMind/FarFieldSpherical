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

$$R = R_y(\alpha) \cdot R_x(-\beta) \cdot R_z(\gamma)$$

i.e. roll $\gamma$ about $z$ first, then elevation $\beta$ about $x$, then azimuth $\alpha$ about $y$. The signs are chosen so that positive angles tilt the boresight toward the positive axes. The individual rotation matrices are:

$$R_z(\gamma) = \begin{pmatrix} \cos\gamma & -\sin\gamma & 0 \\ \sin\gamma & \cos\gamma & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

$$R_x(-\beta) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\beta & \sin\beta \\ 0 & -\sin\beta & \cos\beta \end{pmatrix}$$

$$R_y(\alpha) = \begin{pmatrix} \cos\alpha & 0 & \sin\alpha \\ 0 & 1 & 0 \\ -\sin\alpha & 0 & \cos\alpha \end{pmatrix}$$

The same matrices are used by the standalone `isometric_rotation` helper. The original boresight $(+z)$ moves to $R\hat{z}$:

$$\theta_0 = \arccos(\cos\alpha\cos\beta), \qquad \phi_0 = \operatorname{atan2}(\sin\beta,\; \sin\alpha\cos\beta)$$

A positive $\alpha$ tilts the boresight toward $+x$ ($\phi = 0°$), a positive $\beta$ toward $+y$ ($\phi = 90°$), and a positive $\gamma$ rolls the pattern about $z$ from $+x$ toward $+y$.

### Rotation versus measurement correction

`rotate` changes the antenna's orientation: the whole pattern, field vectors included, turns rigidly about the origin. It is the operation to use when the antenna will be mounted pointing somewhere other than $+z$.

`shift_theta_origin` and `shift_phi_origin` are **measurement corrections**. They re-zero the measured angle axes to compensate for a positioner or mounting offset: `shift_phi_origin` relabels the cuts, and `shift_theta_origin` slides each $\phi$ cut along its own $\theta$ axis. Because every cut is shifted along a different great circle, a theta-origin shift is not a rotation of the antenna and does not move the boresight to a definite $(\theta_0, \phi_0)$; it only makes sense for small offsets and for data whose true boresight was at $\theta = 0$ of the positioner.

### Theta Origin Shift

`shift_theta_origin(offset)` resamples every cut so that

$$E_{new}(\theta, \phi) = E_{old}(\theta + \delta, \phi)$$

The shift is carried out in central format, where each $\phi$ cut is a closed great circle covering $360°$ of $\theta$. A cut that spans the full circle is treated as periodic, so samples pushed past one end reappear at the other and nothing is lost; a partial cut is extended with its end values. Amplitude and unwrapped phase are interpolated separately with cubic splines. A sided-format pattern is converted to central, shifted, converted back, and mapped onto its own $\theta/\phi$ grid, so the caller's layout is unchanged; this is why a shift applied to a sided pattern shows up as data moving from the $\phi$ cuts onto the $\phi + 180°$ cuts across boresight rather than being clipped at $\theta = 0$.

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

Each $\phi$ cut is treated as a closed circle in $\theta$ (central format, $\theta$ from $-180°$ to $180°$) and expanded in cylindrical harmonics, which on a circle is a Fourier series in $\theta$:

$$c_n = \frac{1}{2\pi}\int_0^{2\pi} E(\theta)\,e^{-jn\theta}\,d\theta, \qquad E_{filtered}(\theta) = \sum_{|n| \le n_{max}} c_n\,e^{+jn\theta}$$

1. **Decompose** each cut with a periodic (rectangular-rule) transform, which is exact for a band-limited periodic field on a uniform grid.
2. **Filter**: keep $|n| \le n_{max}$ and discard the rest.
3. **Reconstruct** on the original grid.

The expansion needs every cut to span a full $360°$ of $\theta$, so `apply_mars` works in central format: a sided pattern is converted on a copy, filtered, and the result mapped back onto its own grid, the same as `shift_theta_origin`. A pattern that cannot be closed (a hemisphere, or a sided pattern whose $\theta$ does not start at $0°$) is rejected. If $n_{max}$ reaches the sampling limit of the $\theta$ grid the filter removes nothing and a warning is logged.

The mode limit $n_{max} = \lfloor kD \rfloor$ is meaningful only when the antenna is centred on the origin the pattern is referred to. Translate the pattern to its phase centre first (`translate`, or the Phase Center step of the viewer's processing pipeline, which runs before MARS) so that the filter acts on range reflections rather than on the antenna's own displaced-origin phase ramp.

### Parameters

- **Maximum radial extent** $D$: the largest distance from the origin to any part of the antenna (in metres). This determines $n_{max}$ and thus the aggressiveness of the filtering.
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

Every $\phi$ cut passes through boresight ($\theta = 0°$), which is a single physical direction, so the Ludwig-3 components $E_x(0, \phi_i)$ and $E_y(0, \phi_i)$ should be identical for every cut. Any spread across cuts at boresight is a per-cut measurement offset (a gain or phase drift between cuts), and this operation removes it by scaling each cut with one complex factor.

### Method

For each cut $\phi_i$ the boresight sample $E(0, \phi_i)$ is extracted and a common reference is formed from the magnitude median and the circular mean of the phase:

$$|E_{ref}| = \operatorname{median}_i |E(0, \phi_i)|, \qquad \arg E_{ref} = \arg \sum_i E(0, \phi_i)$$

The median magnitude is robust to an outlier cut. The circular (vector) mean of the phase is used because a linear median of wrapped angles fails when the phases straddle $\pm180°$: samples at $179°$, $-179°$, $178°$, $-178°$ have a linear median near $0°$, which would rotate the whole pattern by $180°$.

Each cut is then scaled:

$$E'(\theta, \phi_i) = E(\theta, \phi_i) \cdot \frac{E_{ref}}{E(0, \phi_i)}$$

### Which component sets the correction

The correction for a cut is derived from the dominant component, the one with the larger median boresight magnitude. A component whose median boresight magnitude is more than 20 dB below the dominant one is the cross-pol of a linearly polarized antenna, and its boresight value is noise; deriving a correction from it would divide by that noise and apply an arbitrary complex gain to the entire cut. Such a component borrows the dominant component's correction. When both components are significant (dual-polarized or circularly polarized antennas) each is corrected independently. The threshold is the `weak_component_ratio` argument, 0.1 by default.

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

