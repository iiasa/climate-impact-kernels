from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from impacts.runner import run_impact_matrix


def _synthetic_loader(gcm: str, scenario: str) -> xr.Dataset:
    """Return a tiny climate dataset for testing."""
    times = pd.date_range("2001-01-01", periods=3, freq="MS")
    return xr.Dataset(
        {"tas": (["time", "lat", "lon"], np.full((3, 2, 2), 300.0))},
        coords={"time": times, "lat": [10.25, 11.25], "lon": [20.25, 21.25]},
    )


def _synthetic_kernel(mapping: pd.DataFrame, climate: xr.Dataset) -> xr.Dataset:
    """Trivial kernel: return mean temperature per entity."""
    n_entities = len(mapping)
    n_times = climate.sizes["time"]
    return xr.Dataset(
        {"capacity_factor": (["time", "entity"], np.ones((n_times, n_entities)))},
        coords={
            "time": climate.time,
            "entity": mapping["UNITID"].tolist(),
        },
    )


class RunnerContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = pd.DataFrame({
            "UNITID": [1, 2],
            "grid_lat": [10.25, 11.25],
            "grid_lon": [20.25, 21.25],
            "mw": [100.0, 200.0],
        })

    def test_run_impact_matrix_produces_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = run_impact_matrix(
                kernel=_synthetic_kernel,
                mapping=self.mapping,
                loader=_synthetic_loader,
                gcms=["GCM_A"],
                scenarios=["hist", "ssp370"],
                output_dir=output_dir,
            )
            self.assertEqual(len(paths), 2)
            for path in paths:
                self.assertTrue(path.exists())
                ds = xr.open_dataset(path)
                self.assertIn("capacity_factor", ds.data_vars)
                self.assertEqual(ds.sizes["entity"], 2)
                ds.close()

    def test_run_impact_matrix_skips_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Create a pre-existing output
            existing = output_dir / "GCM_A_hist.nc"
            xr.Dataset({"dummy": 1}).to_netcdf(existing)

            paths = run_impact_matrix(
                kernel=_synthetic_kernel,
                mapping=self.mapping,
                loader=_synthetic_loader,
                gcms=["GCM_A"],
                scenarios=["hist"],
                output_dir=output_dir,
            )
            # Should return the existing file without re-running
            self.assertEqual(len(paths), 1)
            ds = xr.open_dataset(paths[0])
            self.assertIn("dummy", ds.data_vars)
            self.assertNotIn("capacity_factor", ds.data_vars)
            ds.close()

    def test_run_impact_matrix_applies_post_process(self) -> None:
        def add_summary(ds: xr.Dataset) -> xr.Dataset:
            return ds.assign(mean_cf=ds["capacity_factor"].mean("time"))

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = run_impact_matrix(
                kernel=_synthetic_kernel,
                mapping=self.mapping,
                loader=_synthetic_loader,
                gcms=["GCM_A"],
                scenarios=["hist"],
                output_dir=Path(tmpdir),
                post_process=add_summary,
            )
            ds = xr.open_dataset(paths[0])
            self.assertIn("mean_cf", ds.data_vars)
            ds.close()

    def test_run_impact_matrix_creates_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c"
            paths = run_impact_matrix(
                kernel=_synthetic_kernel,
                mapping=self.mapping,
                loader=_synthetic_loader,
                gcms=["GCM_A"],
                scenarios=["hist"],
                output_dir=nested,
            )
            self.assertTrue(nested.exists())
            self.assertEqual(len(paths), 1)

    def test_run_impact_matrix_rebuilds_invalid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            existing = output_dir / "GCM_A_hist.nc"
            existing.write_text("not a netcdf file")

            paths = run_impact_matrix(
                kernel=_synthetic_kernel,
                mapping=self.mapping,
                loader=_synthetic_loader,
                gcms=["GCM_A"],
                scenarios=["hist"],
                output_dir=output_dir,
            )

            invalid_backups = sorted(output_dir.glob("GCM_A_hist.nc.invalid*"))
            self.assertEqual(len(invalid_backups), 1)
            self.assertTrue(invalid_backups[0].exists())

            ds = xr.open_dataset(paths[0])
            self.assertIn("capacity_factor", ds.data_vars)
            ds.close()

    def test_run_impact_matrix_cleans_up_temp_file_on_write_failure(self) -> None:
        def failing_write(dataset: xr.Dataset, path: str | Path, *args: object, **kwargs: object) -> None:
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch.object(xr.Dataset, "to_netcdf", new=failing_write):
                with self.assertRaisesRegex(OSError, "disk full"):
                    run_impact_matrix(
                        kernel=_synthetic_kernel,
                        mapping=self.mapping,
                        loader=_synthetic_loader,
                        gcms=["GCM_A"],
                        scenarios=["hist"],
                        output_dir=output_dir,
                    )

            self.assertFalse((output_dir / "GCM_A_hist.nc").exists())
            self.assertEqual(list(output_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
