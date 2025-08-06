# GRISMCONF Documentation: Generalized Grism Coordinate Transformations

GRISMCONF is a Python module that implements a generalized coordinate transformation approach for grism spectroscopy, addressing limitations in the traditional aXe configuration methodology used by HST instruments (NICMOS, ACS, WFC3).

Updated from the original code by Pirzkal and Ryan (2017) [npirzkal/GRISMCONF](https://github.com/npirzkal/GRISMCONF) and [WFC3-2017-01.pdf](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2017/WFC3-2017-01.pdf).

*Documentation based on HST Instrument Science Report WFC3 2017-01 by Nor Pirzkal & R. Ryan (2017) [(WFC3-2017-01.pdf)](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2017/WFC3-2017-01.pdf).*

## Introduction

The grism configuration file enables coordinate transformations between two systems: image coordinates $(x, y)$ + wavelength $(\lambda)$ and dispersed spectrum coordinates $(x', y')$.

This parallels WCS transformations (`pixtosky` and `skytopix`) that convert between sky coordinates and detector pixels while correcting for geometric distortions. For dispersed spectra, we need to determine where light at detector position $(x, y)$ will appear in the dispersed frame as a function of wavelength. The grism configuration provides the necessary transformation functions, abstracting implementation details from users.


## Core Aspects of the Generalized Parametric Approach

The new grism configuration approach generalizes the coordinate transformation process, allowing for more flexible and efficient spectral extraction and dispersion calculations. 

A key feature is the introduction of a generalized coordinate parameter $t$, which spans the range (0, 1) and describing the pathlength along the trace. This parameter is not directly tied to physical coordinates but serves as a flexible variable for defining spectral traces.
This parameter avoids the assumption that the dispersion is a function of the x-coordinate.

How the coordinate $t$ is related, or not, to physical quantities such as $x$, $y$ and  can be determined on an instrument/disperser basis and is something that remains transparent to the user. 

### Core Transformation Functions

The new approach uses a generalized parameter $t \in [0,1]$ and six fundamental functions:

**Forward Functions:**
$$
\begin{align}
\hat{x} = x^\prime - x = f_x(x, y; t) \quad \text{(DISPX)} \\
\hat{y} = y^\prime - y = f_y(x, y; t) \quad \text{(DISPY)} \\
\lambda = f_\lambda(x, y; t) \quad \text{(DISPL)}
\end{align}
$$

**Inverse Functions:**

$$
\begin{align}
t = f_x^{-1}(x, y; \hat{x}) \quad \text{(INVDISPX)} \\
t = f_y^{-1}(x, y; \hat{y}) \quad \text{(INVDISPY)} \\ 
t = f_\lambda^{-1}(x, y; \lambda) \quad \text{(INVDISPL)} 
\end{align}
$$
Where:
- $(x, y)$: Source coordinates in direct image
- $(x^\prime, y^\prime)$: Dispersed coordinates in grism image  
- $\lambda$: Wavelength
- $t$: Generalized parameter ($0 ≤ t ≤ 1$)

### Implementation as Polynomial Representation

Functions are implemented as 2D field-dependent polynomials up to order (2, 3):

$$
P_{2,3}(x, y, t) = \sum_{i=0}^{3} \sum_{j=0}^{9} a_{i,j} \cdot x^{p(j)} \cdot y^{q(j)} \cdot t^i
$$

## Usage Examples (from original package)

### 1. Spectral Extraction (Dispersed → Direct)

```python
import grismconf

# Load configuration
C = grismconf.Config("G102.conf")

# Extract wavelength from dispersed image
dx = xp - x0  # offset in dispersed image
t = C.INVDISPX("A", x0, y0, dx)  # compute parameter t
wavelength = C.DISPL("A", x0, y0, t)  # get wavelength
```

### 2. Spectral Dispersion (Direct → Dispersed)

```python
import grismconf

# Load configuration  
C = grismconf.Config("G102.conf")

# Generate dispersed coordinates for given wavelength
t = C.INVDISPL("A", x0, y0, wavelength)  # compute parameter t
dx = C.DISPX("A", x0, y0, t)  # x-displacement
dy = C.DISPY("A", x0, y0, t)  # y-displacement

# Final dispersed coordinates
xp = x0 + dx
yp = y0 + dy
```

## Key Advantages of this package

```{admonition} Benefits
:class: tip

- **Universal**: Handles both horizontal and vertical dispersion directions
- **Efficient**: Avoids expensive path length integrations
- **Flexible**: Accommodates filter wedge effects and complex optical systems
- **Bidirectional**: Equal precision for forward and inverse transformations
- **Scalable**: Supports vector operations for full spectral traces
```

## Technical Notes

- Analytical inverses are provided for polynomials up to order (1,3) for computational efficiency
- Independent calibration of forward and inverse functions prevents error propagation
- Configuration files support multiple spectral orders and detector configurations
- Framework extensible to other calibrations (sensitivity, flat-fielding)

---

*For detailed mathematical derivations and calibration procedures, refer to the original HST Instrument Science Report WFC3 2017-01.*
