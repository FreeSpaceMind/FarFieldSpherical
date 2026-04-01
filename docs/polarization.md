# Polarization Theory & Conversions

## Overview

The FarFieldSpherical package supports three polarization bases for representing antenna far-field patterns:

1. **Spherical** ($E_\theta$, $E_\phi$) — the native coordinate-system basis
2. **Ludwig-3 Linear** ($E_x$, $E_y$) — a co/cross-pol linear basis
3. **Ludwig-3 Circular** ($E_R$, $E_L$) — right- and left-hand circular components

All conversions between these bases are unitary transformations that preserve the total field magnitude:

$$|E_\theta|^2 + |E_\phi|^2 = |E_x|^2 + |E_y|^2 = |E_R|^2 + |E_L|^2$$

## Spherical Basis ($E_\theta$, $E_\phi$)

The spherical basis is defined by the unit vectors of the spherical coordinate system:

- $E_\theta$: component in the direction of increasing $\theta$ (polar angle)
- $E_\phi$: component in the direction of increasing $\phi$ (azimuthal angle)

These are the components stored natively in antenna pattern file formats (CUT, FFD) and form the starting point for all polarization conversions.

### Limitations

The spherical basis has a singularity at the poles ($\theta = 0°$ and $\theta = 180°$). At $\theta = 0°$, the direction of $\hat{\theta}$ depends on $\phi$, which means the mapping from $(E_\theta, E_\phi)$ to a physical polarization direction is ambiguous at boresight. This motivates the use of Ludwig-3 components.

## Ludwig-3 Linear Basis ($E_x$, $E_y$)

The Ludwig third definition provides a linearly-polarized decomposition that resolves the pole singularity and gives a "natural" co-pol/cross-pol split for linearly polarized antennas.

### Conversion from Spherical

$$\begin{pmatrix} E_x \\ E_y \end{pmatrix} = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix} \begin{pmatrix} E_\theta \\ E_\phi \end{pmatrix}$$

Expanding component-wise:

$$E_x = \cos\phi \cdot E_\theta - \sin\phi \cdot E_\phi$$
$$E_y = \sin\phi \cdot E_\theta + \cos\phi \cdot E_\phi$$

### Conversion to Spherical

The inverse transformation uses the transpose of the rotation matrix (since it is orthogonal):

$$\begin{pmatrix} E_\theta \\ E_\phi \end{pmatrix} = \begin{pmatrix} \cos\phi & \sin\phi \\ -\sin\phi & \cos\phi \end{pmatrix} \begin{pmatrix} E_x \\ E_y \end{pmatrix}$$

Expanding:

$$E_\theta = \cos\phi \cdot E_x + \sin\phi \cdot E_y$$
$$E_\phi = -\sin\phi \cdot E_x + \cos\phi \cdot E_y$$

### Physical Interpretation

At boresight ($\theta = 0°$), the Ludwig-3 components align with the Cartesian axes:

- $E_x$ corresponds to horizontally polarized radiation (along $\hat{x}$)
- $E_y$ corresponds to vertically polarized radiation (along $\hat{y}$)

Away from boresight, the Ludwig-3 basis vectors smoothly vary to maintain a consistent notion of "horizontal" and "vertical" polarization relative to the antenna's principal planes. This makes Ludwig-3 the preferred basis for analyzing linearly polarized antennas.

## Ludwig-3 Circular Basis ($E_R$, $E_L$)

Circular polarization components are derived from the Ludwig-3 linear components, representing right-hand and left-hand circular polarization.

### Conversion from Linear

$$E_R = \frac{E_x + iE_y}{\sqrt{2}}$$

$$E_L = \frac{E_x - iE_y}{\sqrt{2}}$$

where $i = \sqrt{-1}$.

### Conversion to Linear

$$E_x = \frac{E_R + E_L}{\sqrt{2}}$$

$$E_y = \frac{i(E_L - E_R)}{\sqrt{2}} = \frac{-i(E_R - E_L)}{\sqrt{2}}$$

### Convention

The package follows the IEEE convention for circular polarization:

- **RHCP** ($E_R$): the electric field vector rotates clockwise when viewed looking into the direction of propagation (from the perspective of the receiver)
- **LHCP** ($E_L$): the electric field vector rotates counter-clockwise when viewed in the same manner

## Complete Conversion Chain

The full conversion path from spherical to circular is:

$$E_\theta, E_\phi \xrightarrow{\text{Ludwig-3 rotation}} E_x, E_y \xrightarrow{\text{Circular decomposition}} E_R, E_L$$

Each step is independently invertible, so any basis can be converted to any other.

The combined spherical-to-circular transformation in a single step:

$$E_R = \frac{1}{\sqrt{2}}\left[(\cos\phi \cdot E_\theta - \sin\phi \cdot E_\phi) + i(\sin\phi \cdot E_\theta + \cos\phi \cdot E_\phi)\right]$$

$$E_L = \frac{1}{\sqrt{2}}\left[(\cos\phi \cdot E_\theta - \sin\phi \cdot E_\phi) - i(\sin\phi \cdot E_\theta + \cos\phi \cdot E_\phi)\right]$$

Which can be simplified to:

$$E_R = \frac{1}{\sqrt{2}}\left[E_\theta(\cos\phi + i\sin\phi) + E_\phi(-\sin\phi + i\cos\phi)\right]$$

$$E_R = \frac{1}{\sqrt{2}}\left[E_\theta \cdot e^{i\phi} + i \cdot E_\phi \cdot e^{i\phi}\right] = \frac{e^{i\phi}}{\sqrt{2}}(E_\theta + iE_\phi)$$

$$E_L = \frac{e^{-i\phi}}{\sqrt{2}}(E_\theta - iE_\phi)$$

## Co-Polarization and Cross-Polarization

The package assigns co-polarization ($E_{co}$) and cross-polarization ($E_{cx}$) based on the user's selected polarization reference. The co-pol component is the desired polarization; the cross-pol is the orthogonal component.

| Polarization Setting | $E_{co}$ | $E_{cx}$ | Typical Use Case |
|---------------------|----------|----------|-----------------|
| `theta` | $E_\theta$ | $E_\phi$ | Analyzing $\theta$-polarized fields directly |
| `phi` | $E_\phi$ | $E_\theta$ | Analyzing $\phi$-polarized fields directly |
| `x` (Ludwig-3 X) | $E_x$ | $E_y$ | Horizontally polarized antennas |
| `y` (Ludwig-3 Y) | $E_y$ | $E_x$ | Vertically polarized antennas |
| `rhcp` | $E_R$ | $E_L$ | Right-hand circularly polarized antennas |
| `lhcp` | $E_L$ | $E_R$ | Left-hand circularly polarized antennas |

The co-pol and cross-pol components are used throughout the package for directivity calculations, pattern normalization, and display purposes.

## Axial Ratio

The axial ratio quantifies the shape of the polarization ellipse, indicating how close the polarization is to circular or linear.

### Definition

$$AR_{dB} = 20 \log_{10}\left(\frac{|E_R| + |E_L|}{||E_R| - |E_L||}\right)$$

### Interpretation

- $AR = 0\;dB$: perfect circular polarization, where $|E_R| = |E_L|$ (the polarization ellipse is a circle)
- $AR = 3\;dB$: the major axis of the polarization ellipse is $\sqrt{2}$ times the minor axis
- $AR \to \infty\;dB$: pure linear polarization, where one circular component is zero (the ellipse degenerates to a line)

### Numerical Protection

In practice, the denominator $||E_R| - |E_L||$ can be extremely small or zero (at points of perfect circular polarization). The implementation uses a floor value of $\epsilon = 10^{-15}$ to prevent division by zero:

$$AR_{dB} = 20 \log_{10}\left(\frac{|E_R| + |E_L|}{\max(||E_R| - |E_L||, \epsilon)}\right)$$

### Sense of Rotation

The sign convention distinguishes RHCP-dominant from LHCP-dominant polarization:

- $|E_R| > |E_L|$: RHCP-dominant (positive sense)
- $|E_L| > |E_R|$: LHCP-dominant (negative sense)

The axial ratio magnitude alone does not indicate the sense of rotation; this must be determined by comparing the magnitudes of $E_R$ and $E_L$.
