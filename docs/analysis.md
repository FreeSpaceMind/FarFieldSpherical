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

See [Pattern Operations - Phase Center Optimization](pattern_operations.md#phase-center-optimization) for algorithmic details on the principal plane formula.

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

## Feed Cross-Polarization Metrics

These functions evaluate the cross-polarization requirements placed on a reflector feed, as defined in *Cross-Polarization Metrics for a Reflector Feed*. They live in `farfield_spherical.crosspol` and are exported from the package root; `crosspol_report` is also available as a method on `FarFieldSpherical`.

### Domain and Conventions

All quantities are evaluated over the illumination cone

$$0 \le \theta \le \theta_e, \qquad 0 \le \phi < 360^\circ$$

where $\theta_e$ is the half-angle subtended by the reflector at the feed. The fields used are `e_co` and `e_cx` as currently assigned on the pattern. Ludwig-3 co/cross definitions require the pattern polarization to be `'x'` or `'y'`; any other polarization is accepted with a logged warning, since the integrals are well defined for any co/cross pair.

Computation is done on a copy of the pattern transformed to sided format (the caller's pattern is never modified). Quadrature is rectangular, matching the requirement text:

$$\iint f \, d\Omega \approx \sum_{i,k} f(\theta_i, \phi_k)\, \sin\theta_i \, \Delta\theta \, \Delta\phi$$

Requirements on the grid:

- $\phi$ must be uniformly spaced and cover a full $360^\circ$ exactly once. A duplicated endpoint ($-180/+180$ or $0/360$) is dropped before integrating. Anything less than full coverage raises `ValueError`, so a half-plane measurement cannot be evaluated.
- $\theta$ must be uniformly spaced inside the cone, and $\theta_e$ must lie inside the pattern's $\theta$ range.

Because every metric is a ratio, absolute field scaling (for example the $1/\sqrt{60}$ applied by `read_ffd`) has no effect.

### Integrated XPD

$$\text{XPD}_\text{int} = 10\log_{10}\left[\frac{\iint_{\text{cone}} |E_\text{co}|^2 \sin\theta\, d\theta\, d\phi}{\iint_{\text{cone}} |E_\text{cx}|^2 \sin\theta\, d\theta\, d\phi}\right]$$

Ratio of co- to cross-polarized power delivered to the reflector. This is the screening gate: it bounds the total cross-polarized power in the aperture regardless of how it is distributed in azimuth. Computed by `integrated_xpd(pattern, theta_e)`; returned as `xpd_int_db`.

### Azimuthal Mode Content

The cross-polarized field is decomposed in azimuth:

$$c_n(\theta) = \frac{1}{2\pi}\int_0^{2\pi} E_\text{cx}(\theta, \phi)\, e^{-jn\phi}\, d\phi$$

which on a uniform $\phi$ grid equals the FFT along $\phi$ divided by the number of samples. The power carried by each order over the cone is

$$P_n = 2\pi \int_0^{\theta_e} |c_n(\theta)|^2 \sin\theta\, d\theta$$

For $n \ge 1$ the reported mode power is the sum of the $+n$ and $-n$ bins; $n = 0$ is the single DC bin. `mode_power_rel_db` expresses each $P_n$ relative to the total cross-polarized power in the cone (by Parseval, the sum over all bins), so it does not depend on the number of orders retained. Computed by `azimuthal_modes(pattern, theta_e, n_max, component)`, with `component` either `'e_cx'` (default) or `'e_co'`.

For a well-behaved linearly polarized feed, essentially all cross-pol power is in $n = 2$ (the $\cos 2\phi$ Ludwig-3 term), which integrates to zero on boresight of a symmetric reflector and therefore does not set the system boresight cross-polarization.

### n = 0 Cross-Polarization Level

$$L_0 = 20\log_{10}\left(\frac{\max|E_\text{co}|}{\max_{\theta \le \theta_e} |c_0(\theta)|}\right)$$

The azimuthally symmetric cross-pol component is the one that survives the reflector's azimuthal integration and appears on the system boresight. The peak co-polarized amplitude is taken over the whole pattern, not just the cone. Larger is better. Computed by `n0_crosspol_level(pattern, theta_e)`; returned as `n0_level_db`.

On a horn-only, azimuthally symmetric simulation this value sits at the numerical floor of the solver (typically 65–80 dB) and is not representative of the assembled feed; the requirement is meaningful for measured feeds or models that include the asymmetric structure around the horn.

### Supporting Quantities

| Name | Definition | Purpose |
|------|------------|---------|
| `xpd_worst_db` | $\min_{\text{cone}} 20\log_{10}\left(\lvert E_\text{co}\rvert / \lvert E_\text{cx}\rvert\right)$ at the same angle | Conventional point XPD; pessimistic because it is dominated by the cone edge where co-pol has rolled off |
| `xpol_peak_db` | $20\log_{10}\left(\max\lvert E_\text{co}\rvert / \max_{\text{cone}}\lvert E_\text{cx}\rvert\right)$ | Peak-referenced cross-pol, the number usually quoted on a datasheet |
| `edge_taper_db` | $20\log_{10}\left(\overline{\lvert E_\text{co}(\theta_e, \phi)\rvert}^{\,\phi} / \max\lvert E_\text{co}\rvert\right)$ | Confirms the feed and $\theta_e$ are a sensible pairing (about $-10$ to $-13$ dB for a typical design) |

`point_xpd` returns the first two; `edge_taper` returns the third using the $\theta$ sample nearest $\theta_e$ (reported in `attrs['theta_actual_deg']`).

### Report and Requirement Check

`crosspol_report(pattern, theta_e, n_max=6)` returns one Dataset with `xpd_int_db`, `n0_level_db`, `xpd_worst_db`, `xpol_peak_db`, `edge_taper_db` (dims `frequency`) and `mode_power_rel_db` (dims `frequency, n`).

`check_requirements(report, xpd_int_min_db=None, n0_min_db=None, bands_hz=None)` returns a copy of the report with, for each requirement given, a per-frequency margin and pass flag, plus an `in_band` flag when `bands_hz` (a list of `(lo, hi)` in Hz) is supplied. Out-of-band frequencies are still reported but excluded from `attrs['all_pass']` and from the worst-case attributes `worst_xpd_int_in_band_db` and `worst_n0_in_band_db`.

```python
from farfield_spherical import read_ffd, crosspol_report, check_requirements

pattern = read_ffd("feed.ffd")
report = crosspol_report(pattern, theta_e=35.0)
result = check_requirements(report, xpd_int_min_db=20, n0_min_db=40,
                            bands_hz=[(8e9, 11e9), (13e9, 15e9)])
print(result[["xpd_int_db", "n0_level_db", "in_band"]].to_dataframe())
print("PASS" if result.attrs["all_pass"] else "FAIL")
```

---

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
