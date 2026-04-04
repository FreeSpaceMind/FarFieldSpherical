"""
Basic usage example for the FarFieldSpherical library.

This script demonstrates the core workflow:
  1. Reading a GRASP CUT file
  2. Inspecting pattern metadata and dimensions
  3. Working with polarization
  4. Extracting gain and phase data
  5. Saving and loading in NPZ format

Run from the repo root:
    python examples/basic_usage.py

Requires example.cut in examples/data/ or adjust CUT_FILE below.
"""
import os
import numpy as np

from farfield_spherical.io.readers import read_cut
from farfield_spherical.io.npz_format import save_pattern_npz, load_pattern_npz

# ---------------------------------------------------------------------------
# Configuration — adjust paths as needed
# ---------------------------------------------------------------------------
CUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'example.cut')
NPZ_FILE = os.path.join(os.path.dirname(__file__), 'output_pattern.npz')

FREQUENCY = 8.0e9  # Hz


# ---------------------------------------------------------------------------
# 1. Read a GRASP CUT file
# ---------------------------------------------------------------------------
print("Reading CUT file...")
pattern = read_cut(CUT_FILE, frequency_start=FREQUENCY, frequency_end=FREQUENCY)

print(f"  Frequencies : {pattern.frequencies / 1e9} GHz")
print(f"  Theta range : {pattern.theta_angles[0]:.1f}° to {pattern.theta_angles[-1]:.1f}°"
      f"  ({len(pattern.theta_angles)} points)")
print(f"  Phi cuts    : {len(pattern.phi_angles)}"
      f"  ({pattern.phi_angles[0]:.1f}° to {pattern.phi_angles[-1]:.1f}°)")
print(f"  Polarization: {pattern.polarization}")
print(f"  Data shape  : {pattern.data.e_theta.shape}  (freq × theta × phi)")


# ---------------------------------------------------------------------------
# 2. Gain and phase extraction
# ---------------------------------------------------------------------------
print("\nExtracting gain and phase...")
gain_db = pattern.get_gain_db()          # 20*log10|E|, shape (freq, theta, phi)
phase_rad = pattern.get_phase()          # angle(E_co), shape (freq, theta, phi)
gain_co_db = pattern.get_gain_db(component='e_co')

peak_gain = np.nanmax(gain_db[0])
peak_theta_idx = np.unravel_index(np.nanargmax(gain_db[0]), gain_db[0].shape)[0]
peak_theta = pattern.theta_angles[peak_theta_idx]

print(f"  Peak total gain : {peak_gain:.2f} dB")
print(f"  Peak at theta   : {peak_theta:.2f}°")


# ---------------------------------------------------------------------------
# 3. Polarization conversion
# ---------------------------------------------------------------------------
print("\nChanging polarization to RHCP...")
pattern.change_polarization('rhcp')
print(f"  Polarization now: {pattern.polarization}")

# Change back to original
pattern.change_polarization('x')
print(f"  Reverted to     : {pattern.polarization}")


# ---------------------------------------------------------------------------
# 4. Save and reload in NPZ format
# ---------------------------------------------------------------------------
print(f"\nSaving to {NPZ_FILE}...")
save_pattern_npz(pattern, NPZ_FILE)

print("Reloading from NPZ...")
loaded, metadata = load_pattern_npz(NPZ_FILE)
print(f"  Loaded shape    : {loaded.data.e_theta.shape}")
print(f"  Polarization    : {loaded.polarization}")

# Verify fields match
max_diff = np.max(np.abs(loaded.data.e_theta.values - pattern.data.e_theta.values))
print(f"  Max |e_theta| difference after roundtrip: {max_diff:.2e}")

# Clean up output file
os.unlink(NPZ_FILE)

print("\nDone.")
