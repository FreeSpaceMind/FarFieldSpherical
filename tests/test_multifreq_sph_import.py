import numpy as np
import pytest

from farfield_spherical import FarFieldSpherical
from farfield_spherical.io.readers import scan_sph_frequencies
from farfield_spherical.io.swe_utils import create_pattern_from_swe
from swe import SphericalWaveExpansion


def _tiny_swe(frequencies=(8e9, 10e9)):
    blocks = []
    for freq in frequencies:
        scale = freq / 1e9
        blocks.append({
            "frequency": freq,
            "Q1_coeffs": {
                (1, 0): 1.0 + 0.01j * scale,
                (1, -1): 0.15j,
                (1, 1): 0.1,
            },
            "Q2_coeffs": {
                (1, 0): 0.25 - 0.02j * scale,
                (1, -1): 0.05,
                (1, 1): -0.08j,
            },
            "NMAX": 1,
            "MMAX": 1,
        })
    swe = SphericalWaveExpansion.from_frequency_data(blocks)
    swe.normalize_coefficients()
    return swe


def _write_sph(path, frequencies=(8e9, 10e9)):
    swe = _tiny_swe(frequencies)
    swe.to_sph_file(str(path), NTHE=3, NPHI=3, description="tiny test")
    return path


def test_scan_sph_frequencies_multi_and_single(tmp_path):
    multi_path = _write_sph(tmp_path / "multi.sph")
    assert scan_sph_frequencies(multi_path) == [8e9, 10e9]

    single_path = _write_sph(tmp_path / "single.sph", frequencies=(9e9,))
    assert scan_sph_frequencies(single_path) == [9e9]


def test_scan_sph_frequencies_no_header(tmp_path):
    path = tmp_path / "not_sph.sph"
    path.write_text("no frequency headers\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Freq"):
        scan_sph_frequencies(path)


def test_from_ticra_sph_loads_all_single_and_subset(tmp_path):
    path = _write_sph(tmp_path / "multi.sph")
    theta = np.array([0.0, 45.0])
    phi = np.array([0.0, 90.0])

    all_pattern = FarFieldSpherical.from_ticra_sph(
        path,
        theta_angles=theta,
        phi_angles=phi,
    )
    assert np.allclose(all_pattern.frequencies, [8e9, 10e9])
    assert all_pattern.data.e_theta.values.shape == (2, 2, 2)
    assert set(all_pattern.swe.keys()) == {8e9, 10e9}

    single_pattern = FarFieldSpherical.from_ticra_sph(
        path,
        frequency=8e9,
        theta_angles=theta,
        phi_angles=phi,
    )
    assert np.allclose(single_pattern.frequencies, [8e9])
    assert single_pattern.data.e_theta.values.shape == (1, 2, 2)

    subset_pattern = FarFieldSpherical.from_ticra_sph(
        path,
        frequency=[10e9, 8e9],
        theta_angles=theta,
        phi_angles=phi,
    )
    assert np.allclose(subset_pattern.frequencies, [10e9, 8e9])
    assert subset_pattern.data.e_theta.values.shape == (2, 2, 2)


def test_from_ticra_sph_reports_available_frequencies_on_mismatch(tmp_path):
    path = _write_sph(tmp_path / "multi.sph")

    with pytest.raises(ValueError, match="Available frequencies"):
        FarFieldSpherical.from_ticra_sph(
            path,
            frequency=12e9,
            theta_angles=np.array([0.0]),
            phi_angles=np.array([0.0]),
        )


def test_create_pattern_from_swe_preserves_requested_order():
    swe = _tiny_swe()

    pattern = create_pattern_from_swe(
        swe,
        theta_angles=np.array([0.0, 30.0]),
        phi_angles=np.array([0.0, 90.0]),
        frequencies=[10e9, 8e9],
    )

    assert np.allclose(pattern.frequencies, [10e9, 8e9])
    assert pattern.data.e_theta.values.shape == (2, 2, 2)
    assert list(pattern.swe.keys()) == [10e9, 8e9]
