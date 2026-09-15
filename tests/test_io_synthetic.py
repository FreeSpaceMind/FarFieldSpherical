"""
Round-trip tests for the file readers and writers using synthetic patterns.

The repository's real example files are not committed, so every test that
depends on tests/data/ skips. These tests build a pattern in memory, write it,
read it back and compare, so the CUT, FFD, CSV and NPZ paths are exercised
everywhere the suite runs.
"""
import numpy as np
import pytest

from farfield_spherical import (
    FarFieldSpherical,
    load_pattern_npz,
    read_cut,
    read_ffd,
    save_pattern_npz,
    write_cut,
    write_ffd,
    write_csv,
)
from .test_coordinate_transforms import analytic_fields

FREQ_START = 8e9
FREQ_END = 10e9
FREQS = np.array([FREQ_START, 9e9, FREQ_END])


def make_io_pattern(theta=None, phi=None, freqs=FREQS, polarization='x'):
    """A smooth pattern on a uniform grid, which is what both formats can store."""
    theta = np.arange(0, 181, 10.0) if theta is None else np.asarray(theta, float)
    phi = np.arange(0, 360, 30.0) if phi is None else np.asarray(phi, float)
    moments = np.array([[1.0 + 0.3j, 0.5 - 0.2j, 0.25 + 0.1j],
                        [0.4 - 0.1j, 1.0 + 0.2j, -0.3 + 0.5j],
                        [0.8 + 0.0j, -0.2 + 0.4j, 0.6 - 0.1j]])[:len(freqs)]
    e_theta, e_phi = analytic_fields(theta, phi, moments)
    return FarFieldSpherical(theta, phi, np.asarray(freqs, float),
                             e_theta, e_phi, polarization=polarization)


def assert_fields_close(a, b, atol=2e-4):
    np.testing.assert_allclose(a.theta_angles, b.theta_angles, atol=1e-6)
    np.testing.assert_allclose(a.phi_angles, b.phi_angles, atol=1e-6)
    np.testing.assert_allclose(a.data.e_theta.values, b.data.e_theta.values, atol=atol)
    np.testing.assert_allclose(a.data.e_phi.values, b.data.e_phi.values, atol=atol)


class TestFfdRoundTrip:
    def test_fields_and_grid_survive(self, tmp_path):
        pattern = make_io_pattern()
        path = tmp_path / "pattern.ffd"
        write_ffd(pattern, path)
        assert_fields_close(pattern, read_ffd(path))

    def test_absolute_scaling_survives(self, tmp_path):
        """read_ffd divides by sqrt(60) and write_ffd multiplies it back."""
        pattern = make_io_pattern()
        path = tmp_path / "pattern.ffd"
        write_ffd(pattern, path)
        peak_in = np.abs(pattern.data.e_theta.values).max()
        peak_out = np.abs(read_ffd(path).data.e_theta.values).max()
        assert peak_out == pytest.approx(peak_in, rel=1e-4)

    def test_frequencies_survive(self, tmp_path):
        pattern = make_io_pattern()
        path = tmp_path / "pattern.ffd"
        write_ffd(pattern, path)
        np.testing.assert_allclose(read_ffd(path).frequencies, FREQS, rtol=1e-9)

    def test_single_frequency(self, tmp_path):
        pattern = make_io_pattern(freqs=[FREQ_START])
        path = tmp_path / "one.ffd"
        write_ffd(pattern, path)
        result = read_ffd(path)
        assert len(result.frequencies) == 1
        assert_fields_close(pattern, result)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            read_ffd(tmp_path / "does_not_exist.ffd")

    def test_missing_frequency_line(self, tmp_path):
        """Some single-frequency exports omit the 'Frequency' line. Without a
        frequency supplied the first data row used to be parsed as one."""
        pattern = make_io_pattern(freqs=[FREQ_START])
        path = tmp_path / "one.ffd"
        write_ffd(pattern, path)
        lines = path.read_text().splitlines()
        assert lines[3].lower().startswith('freq')
        stripped = tmp_path / "nofreq.ffd"
        stripped.write_text("\n".join(lines[:3] + lines[4:]) + "\n")

        with pytest.raises(ValueError, match="frequency_hz"):
            read_ffd(stripped)

        result = read_ffd(stripped, frequency_hz=FREQ_START)
        assert result.frequencies[0] == pytest.approx(FREQ_START)
        assert_fields_close(pattern, result)


class TestWriterValidation:
    def test_cut_rejects_non_uniform_theta(self, tmp_path):
        """Both headers store start/step/count, so a non-uniform axis would be
        written onto angles it was never sampled at."""
        theta = np.concatenate([np.arange(-180, -20, 20.0), np.arange(-20, 181, 5.0)])
        pattern = make_io_pattern(theta=theta, phi=np.arange(0, 180, 30.0))
        with pytest.raises(ValueError, match='uniformly spaced'):
            write_cut(pattern, tmp_path / "bad.cut")

    def test_ffd_rejects_non_uniform_theta(self, tmp_path):
        theta = np.concatenate([np.arange(0, 60, 20.0), np.arange(60, 181, 5.0)])
        pattern = make_io_pattern(theta=theta)
        with pytest.raises(ValueError, match='uniformly spaced'):
            write_ffd(pattern, tmp_path / "bad.ffd")

    def test_ffd_rejects_non_uniform_phi(self, tmp_path):
        phi = np.array([0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0])
        pattern = make_io_pattern(phi=phi)
        with pytest.raises(ValueError, match='uniformly spaced'):
            write_ffd(pattern, tmp_path / "bad.ffd")


class TestCutRoundTrip:
    @pytest.mark.parametrize('polarization_format', [1, 3])
    def test_fields_and_grid_survive(self, tmp_path, polarization_format):
        # CUT stores one cut per phi over a central theta range
        pattern = make_io_pattern(theta=np.arange(-180, 181, 10.0),
                                  phi=np.arange(0, 180, 30.0))
        path = tmp_path / "pattern.cut"
        write_cut(pattern, path, polarization_format=polarization_format)
        result = read_cut(path, frequency_start=FREQ_START, frequency_end=FREQ_END)
        assert_fields_close(pattern, result)

    def test_frequencies_are_reconstructed(self, tmp_path):
        pattern = make_io_pattern(theta=np.arange(-180, 181, 10.0),
                                  phi=np.arange(0, 180, 30.0))
        path = tmp_path / "pattern.cut"
        write_cut(pattern, path)
        result = read_cut(path, frequency_start=FREQ_START, frequency_end=FREQ_END)
        np.testing.assert_allclose(result.frequencies, FREQS, rtol=1e-9)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_cut(tmp_path / "nope.cut", frequency_start=1e9, frequency_end=2e9)


class TestNpzRoundTrip:
    def test_fields_grid_and_polarization_survive(self, tmp_path):
        pattern = make_io_pattern()
        path = tmp_path / "pattern.npz"
        save_pattern_npz(pattern, path)
        result, _metadata = load_pattern_npz(path)
        assert_fields_close(pattern, result, atol=1e-6)
        assert result.polarization == pattern.polarization
        np.testing.assert_allclose(result.frequencies, pattern.frequencies)

    def test_metadata_survives(self, tmp_path):
        pattern = make_io_pattern()
        path = tmp_path / "pattern.npz"
        save_pattern_npz(pattern, path, metadata={'operator': 'test'})
        _result, metadata = load_pattern_npz(path)
        assert metadata.get('operator') == 'test'

    def test_central_format_survives(self, tmp_path):
        pattern = make_io_pattern(theta=np.arange(-180, 181, 10.0),
                                  phi=np.arange(0, 180, 30.0))
        path = tmp_path / "central.npz"
        save_pattern_npz(pattern, path)
        result, _ = load_pattern_npz(path)
        assert result.theta_angles.min() < 0
        assert_fields_close(pattern, result, atol=1e-6)


class TestCsv:
    def test_writes_one_row_per_sample(self, tmp_path):
        pattern = make_io_pattern()
        path = tmp_path / "pattern.csv"
        write_csv(pattern, path)
        lines = path.read_text().strip().splitlines()
        expected = len(pattern.frequencies) * len(pattern.theta_angles) * len(pattern.phi_angles)
        assert len(lines) == expected + 1          # + header
        assert 'theta' in lines[0].lower()

    def test_extension_is_added(self, tmp_path):
        pattern = make_io_pattern()
        write_csv(pattern, tmp_path / "noext")
        assert (tmp_path / "noext.csv").exists()


class TestFormatsAgree:
    def test_cut_and_ffd_describe_the_same_pattern(self, tmp_path):
        """Written to both formats and read back, the fields must match each
        other as well as the source."""
        pattern = make_io_pattern(theta=np.arange(-180, 181, 10.0),
                                  phi=np.arange(0, 180, 30.0))
        cut_path = tmp_path / "p.cut"
        write_cut(pattern, cut_path)
        from_cut = read_cut(cut_path, frequency_start=FREQ_START, frequency_end=FREQ_END)

        sided = pattern.copy()
        sided.transform_coordinates('sided')
        ffd_path = tmp_path / "p.ffd"
        write_ffd(sided, ffd_path)
        from_ffd = read_ffd(ffd_path)

        from_cut.transform_coordinates('sided')
        np.testing.assert_allclose(from_cut.data.e_theta.values,
                                   from_ffd.data.e_theta.values, atol=2e-4)
