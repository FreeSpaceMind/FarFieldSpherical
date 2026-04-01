# Analysis Functions

This document describes the analysis and computation functions available in the FarFieldSpherical package for extracting quantitative metrics from antenna far-field patterns.

## Directivity Calculation

Directivity measures the antenna's ability to concentrate radiation in a particular direction relative to an isotropic radiator.

### Definition

$$D(\theta, \phi) = \frac{4\pi \cdot U(\theta, \phi)}{P_{total}}$$

where:
- $U(\theta, \phi) = |E(\theta, \phi)|^2$ is the radiation intensity (power per unit solid angle)
- $P_{total}$ is the total radiated power integrated over the full sphere

In decibels:

$$D_{dB}(\theta, \phi) = 10\log_{10}\left(\frac{4\pi \cdot U(\theta, \phi)}{P_{total}}\right)$$

### Total Power Integration

The total radiated power is computed by integrating the radiation intensity over the full sphere:

$$P_{total} = \int_0^{2\pi}\int_0^{\pi} U(\theta, \phi) \sin\theta \, d\theta \, d\phi$$

This integral is evaluated numerically using the trapezoid rule on the discrete $(\theta, \phi)$ grid.

**Central coordinate handling:** When the data is in central format (where $\theta$ can be negative), the solid angle element uses $\sin|\theta|$ to correctly account for the geometry:

$$d\Omega = \sin|\theta| \, d\theta \, d\phi$$

This ensures the integration weight is always positive regardless of the sign convention for $\theta$.

### Component Options

The radiation intensity $U(\theta, \phi)$ can be computed from different field components:

| Component | Radiation Intensity |
|-----------|-------------------|
| **Total** | $U = \|E_\theta\|^2 + \|E_\phi\|^2$ |
| **Co-pol** | $U = \|E_{co}\|^2$ |
| **Cross-pol** | $U = \|E_{cx}\|^2$ |
| **E-theta** | $U = \|E_\theta\|^2$ |
| **E-phi** | $U = \|E_\phi\|^2$ |

When computing directivity for a single component (e.g., co-pol only), the total power $P_{total}$ in the denominator still uses the **total** field ($|E_\theta|^2 + |E_\phi|^2$) to give the true partial directivity. This answers the question: "What fraction of the total radiated power goes into this component in this direction?"

### Partial Sphere Handling

Antenna measurements often do not cover the full $4\pi$ steradians. Near-field measurement systems with limited scan ranges, or measurement systems that only capture the forward hemisphere, produce patterns that cover less than the complete sphere.

When the measured solid angle is less than 80% of the full sphere, the unmeasured regions must be estimated to compute $P_{total}$ accurately. Two methods are available:

#### Far Sidelobe Method (Recommended)

Assumes the unmeasured region has a radiation intensity equal to the peak level reduced by a specified sidelobe level:

$$U_{unmeasured} = U_{peak} \cdot 10^{SLL_{dB}/10}$$

The total power contribution from the unmeasured region is:

$$P_{unmeasured} = U_{unmeasured} \cdot \Omega_{unmeasured}$$

where $\Omega_{unmeasured}$ is the unmeasured solid angle in steradians.

**Typical values:** For a well-designed antenna, $SLL_{dB} = -20$ to $-30\;dB$ is typical. More conservative estimates (less negative $SLL_{dB}$) give lower directivity values.

The total power becomes:

$$P_{total} = P_{measured} + P_{unmeasured}$$

#### Edge Extrapolation Method

Uses the measured field values at the boundary of the measured region, with an additional dB drop, to estimate the unmeasured contribution. This is useful when the edge of the measured region is in the sidelobe region and provides a data-driven estimate.

### Return Values

The directivity function can return different results depending on the query:

- **At a specified direction** $(\theta, \phi)$: returns $D_{dB}(\theta, \phi)$ at that point
- **Peak directivity**: returns a tuple $(D_{peak,dB}, \theta_{peak}, \phi_{peak})$ containing the maximum directivity value and the direction at which it occurs

## Phase Center Calculation

The phase center is the apparent point of origin of the spherical wavefront in the far field. It is the location from which the far-field phase appears most uniform.

### 3D Optimization Method

Finds the translation vector $\mathbf{d} = [x, y, z]$ that minimizes phase variation across the main beam region.

The optimization minimizes:

$$C(\mathbf{d}) = \text{std}\left[\psi(\theta, \phi; \mathbf{d}) \;\big|\; |\theta| \leq \theta_{cone}\right]$$

where $\psi(\theta, \phi; \mathbf{d})$ is the unwrapped phase of the field after applying a phase center translation by $\mathbf{d}$.

See [Pattern Operations - Phase Center Optimization](pattern_operations.md#phase-center-optimization) for full algorithmic details.

### Principal Plane Method

Uses three measured phase points on a principal plane to analytically solve for the phase center displacement in that plane.

See [Pattern Operations - Principal Plane Method](pattern_operations.md#principal-plane-method) for the formula.

### Practical Considerations

- The phase center generally depends on frequency and may differ between the E-plane and H-plane
- For reflector antennas, the phase center is typically near the focal point
- For horn antennas, the phase center is usually inside the horn aperture
- The optimization should use a $\theta_{cone}$ that covers the main beam but excludes sidelobes, where phase is noisy

## Axial Ratio

The axial ratio characterizes the polarization purity by quantifying the shape of the polarization ellipse at each point in the pattern.

### Computation

First, the circular polarization components $E_R$ and $E_L$ are obtained. If the pattern is not already in the circular basis, the Ludwig-3 conversion chain is applied:

$$E_\theta, E_\phi \xrightarrow{\text{Ludwig-3}} E_x, E_y \xrightarrow{\text{Circular}} E_R, E_L$$

Then the axial ratio is:

$$AR_{dB} = 20\log_{10}\left(\frac{|E_R| + |E_L|}{\max(||E_R| - |E_L||, \;\epsilon)}\right)$$

where $\epsilon = 10^{-15}$ prevents division by zero at points of perfect circular polarization.

### Interpretation

| $AR_{dB}$ | Polarization State |
|-----------|-------------------|
| $0\;dB$ | Perfect circular ($\|E_R\| = \|E_L\|$) |
| $3\;dB$ | Elliptical, major/minor axis ratio of $\sqrt{2}$ |
| $15\;dB$ | Nearly linear |
| $\to \infty$ | Perfect linear (one circular component is zero) |

### Relationship to Cross-Pol

For a nominally RHCP antenna, the axial ratio and cross-pol level are related. If $X_{dB}$ is the cross-pol discrimination ($|E_R|^2 / |E_L|^2$ in dB), then:

$$AR_{dB} = 20\log_{10}\left(\frac{10^{X_{dB}/20} + 1}{10^{X_{dB}/20} - 1}\right)$$

For example, $X_{dB} = 20\;dB$ (cross-pol 20 dB below co-pol) gives $AR \approx 1.7\;dB$.

## Pattern Averaging

Computes a weighted average of $N$ patterns that share identical angular and frequency grids.

### Formulation

$$E_{avg}(\theta, \phi) = \sum_{i=1}^{N} w_i \cdot E_i(\theta, \phi)$$

where the weights satisfy:

$$\sum_{i=1}^{N} w_i = 1$$

Default weights are uniform: $w_i = 1/N$.

### Properties

- The averaging is performed on the **complex** field values, preserving both amplitude and phase information
- Coherent averaging (complex) will reduce incoherent noise while preserving the coherent signal
- If the patterns have random phase errors, averaging $N$ patterns reduces the error by approximately $1/\sqrt{N}$
- All input patterns must have exactly the same $(\theta, \phi, f)$ grids

### Use Cases

- Averaging multiple measurement sweeps to reduce random noise
- Combining dual-sphere measurements with appropriate weighting
- Creating composite patterns from multiple test configurations

## Pattern Difference

Computes the complex ratio between two patterns to reveal gain and phase deviations.

### Formulation

$$E_{diff}(\theta, \phi) = \frac{E_1(\theta, \phi)}{E_2(\theta, \phi)}$$

### Phase Alignment

Before computing the ratio, the global phase of pattern 2 is aligned to pattern 1 at boresight:

$$E'_2(\theta, \phi) = E_2(\theta, \phi) \cdot e^{j[\angle E_1(0, 0) - \angle E_2(0, 0)]}$$

This removes systematic phase offsets (e.g., from different measurement setups or cable lengths) so that the resulting difference pattern shows only the true deviations.

### Numerical Protection

A floor of $10^{-30}$ is applied to the denominator to prevent division by zero in regions where pattern 2 has nulls:

$$E_{diff}(\theta, \phi) = \frac{E_1(\theta, \phi)}{\max(|E'_2(\theta, \phi)|, 10^{-30}) \cdot e^{j\angle E'_2(\theta, \phi)}}$$

### Interpretation

- **Amplitude of the difference**: $20\log_{10}|E_{diff}|$ shows the gain deviation in dB. A value of $0\;dB$ means the patterns have equal magnitude at that point.
- **Phase of the difference**: $\angle E_{diff}$ shows the phase deviation. A value of $0°$ means the patterns have identical phase.

### Use Cases

- Comparing measured patterns against simulation references
- Quantifying measurement repeatability between test runs
- Identifying systematic errors in measurement systems
- Before/after comparison when applying corrections (e.g., MARS filtering)
