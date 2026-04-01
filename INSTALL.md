# Installation Guide for FarFieldSpherical

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Installation Methods

### 1. Development Installation (Recommended for Contributors)

If you're developing or modifying the package:

```bash
# Navigate to the project directory
cd farfield-spherical

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

The `-e` flag installs the package in "editable" mode, meaning changes to the source code are immediately reflected without reinstalling.

### 2. Standard Installation from Source

```bash
# Navigate to the project directory
cd farfield-spherical

# Install the package
pip install .
```

### 3. Installation with Optional Dependencies

#### With Spherical Wave Expansion support:
```bash
pip install ".[swe]"
```

#### With Development tools:
```bash
pip install ".[dev]"
```

#### With Documentation tools:
```bash
pip install ".[docs]"
```

#### Install everything:
```bash
pip install ".[swe,dev,docs]"
```

### 4. Installation from PyPI (When Published)

Once the package is published to PyPI:

```bash
pip install farfield-spherical
```

## Verifying Installation

After installation, verify it works:

```python
python -c "from farfield_spherical import FarFieldSpherical; print('Success!')"
```

Or run a more complete test:

```python
import numpy as np
from farfield_spherical import FarFieldSpherical

# Create a simple pattern
theta = np.linspace(-180, 180, 361)
phi = np.linspace(0, 360, 73)
freq = np.array([1e9])
e_theta = np.ones((1, 361, 73), dtype=complex)
e_phi = np.ones((1, 361, 73), dtype=complex)

pattern = FarFieldSpherical(theta, phi, freq, e_theta, e_phi)
print(f"Created pattern with {len(pattern.frequencies)} frequencies")
print("Installation verified successfully!")
```

## Running Tests

If you installed with dev dependencies:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=farfield_spherical --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Building Documentation

If you installed with docs dependencies:

```bash
cd docs
make html
open _build/html/index.html
```

## Building Distribution Packages

To create distribution packages (wheel and source):

```bash
# Install build tools
pip install build

# Build the package
python -m build

# This creates:
# dist/farfield_spherical-1.0.0-py3-none-any.whl
# dist/farfield-spherical-1.0.0.tar.gz
```

## Uninstalling

```bash
pip uninstall farfield-spherical
```

## Troubleshooting

### ImportError: No module named 'farfield_spherical'

Make sure the package is installed:
```bash
pip list | grep farfield-spherical
```

If not found, reinstall using one of the methods above.

### Dependency Conflicts

If you encounter dependency conflicts, try creating a fresh virtual environment:

```bash
# Create a new virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install the package
pip install -e .
```

### NumPy/SciPy Build Issues

If you have trouble installing NumPy or SciPy, you may need to install system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-dev libopenblas-dev liblapack-dev gfortran
```

**macOS:**
```bash
brew install openblas lapack gcc
```

**Windows:**
Use pre-built wheels from: https://www.lfd.uci.edu/~gohlke/pythonlibs/

### Permission Errors

If you get permission errors during installation, either:

1. Use a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install .
   ```

2. Install for user only:
   ```bash
   pip install --user .
   ```

3. Use sudo (not recommended):
   ```bash
   sudo pip install .
   ```

## Upgrading

To upgrade to a newer version:

```bash
# From PyPI (when published)
pip install --upgrade farfield-spherical

# From source
cd farfield-spherical
git pull
pip install --upgrade .
```

## Virtual Environment Best Practices

Always use a virtual environment for development:

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install package
pip install -e ".[dev]"

# When done
deactivate
```

## Getting Help

If you encounter issues not covered here:

1. Check for existing issues
2. Report the problem with:
   - Your Python version: `python --version`
   - Your OS
   - Full error message
   - Steps to reproduce