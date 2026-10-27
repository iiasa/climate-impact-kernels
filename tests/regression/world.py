"""The made-up input data shared by the end-to-end tests: files on disk plus products derived once and reused.

``World`` writes the on-disk inputs (an ISIMIP3b-style tree, GMT tables, region
rasters, the cells of today's fleet, a common-definitions checkout) into one
temporary directory and derives the intermediate products later tests consume:
cooling and wet/dry mappings, three small site evaluations, and the split of the
wet capacity loss built from them. Derived products come from the library code
through the table of library functions (``entrypoints.py``), so a change in
behaviour upstream shows up in every test that depends on it as well as in the
test that checks it directly.
"""

from __future__ import annotations

import contextlib
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from . import entrypoints as ep
from . import fixtures
from .canon import Scrubber

GCM_B = "IPSL-CM6A-LR"


class World:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.scrub = Scrubber(self.root, str(ep.REPO_ROOT))
        self.work = self.root / "work"
        self.work.mkdir(parents=True, exist_ok=True)

    # --- inputs written to disk ------------------------------------------------

    @cached_property
    def definitions_root(self) -> Path:
        return fixtures.write_common_definitions(self.root / "common-definitions")

    def region_definitions(self):
        return fixtures.common_definitions(self.definitions_root)

    @cached_property
    def region_paths(self) -> dict[str, Path]:
        return fixtures.write_region_files(self.root / "regions")

    @cached_property
    def watxene_root(self) -> Path:
        return fixtures.write_isimip_tree(self.root / "watxene")

    @cached_property
    def counterclim_root(self) -> Path:
        return fixtures.write_counterclim_tree(self.root / "counterclim")

    @cached_property
    def gmt_root(self) -> Path:
        return fixtures.write_gmt(self.root / "gmt")

    @cached_property
    def fleet_files(self) -> tuple[Path, Path]:
        return fixtures.write_fleet_files(self.root / "fleet")

    @cached_property
    def isimip_paths(self):
        return ep.get("isimip.isimip3b_paths")(self.watxene_root)

    @cached_property
    def plants_parquet(self) -> Path:
        path = self.root / "plants.parquet"
        fixtures.plants().to_parquet(path, index=False)
        return path

    # --- derived fleet products --------------------------------------------------

    @cached_property
    def cooling(self) -> pd.DataFrame:
        return ep.get("cooling.build_unit_cooling_mapping")(fixtures.plants())

    @cached_property
    def wet_mapping(self) -> pd.DataFrame:
        imputed = ep.get("fleet.build_imputed_wet_unit_cell_mapping")(
            fixtures.plants(), self.cooling, random_seed=0
        )
        mapping = ep.get("wet.build_mapping")(imputed, self.cooling)
        return ep.get("wet.assign_saline_intakes")(mapping, fixtures.monthly_climate()["tos"])

    @cached_property
    def dry_mapping(self) -> pd.DataFrame:
        imputed = ep.get("fleet.build_imputed_dry_unit_cell_mapping")(
            fixtures.plants(), self.cooling, random_seed=0
        )
        return ep.get("dry.build_mapping")(imputed, self.cooling)

    # --- site evaluations (audit) --------------------------------------------------

    def audit_config(self, name: str, **overrides) -> dict:
        """Settings of a warming-level site evaluation on the made-up tree (see ``audit.run``)."""

        config = {
            "gcm": fixtures.GCM,
            "watxene_root": str(self.watxene_root),
            "gmt_root": str(self.gmt_root),
            "scenarios": list(fixtures.SCENARIOS),
            "gwl_sampling": {
                "bin_width": 2.0,
                "min_level": 1.0,
                "max_level": 3.5,
                "years_per_bin": 2,
                "seed": 730,
            },
            "region": {key: str(path) for key, path in self.region_paths.items()},
            "fleet": {"wet": str(self.fleet_files[0]), "dry": str(self.fleet_files[1])},
            "output": str(self.work / f"{name}.parquet"),
            "summary": {
                "bootstrap_replicates": 6,
                "bootstrap_seed": 17,
                "bad_tail_thresholds": {"capacity_factor": 0.9},
            },
        }
        config.update(overrides)
        return config

    @cached_property
    def audit_summary_run(self) -> dict[str, Path]:
        """Default responses with the per-region summary and the comparison of potential with existing sites."""

        config = self.audit_config("audit_summary")
        output = ep.get("audit.run")(config)
        return {
            "long": output,
            "summary": output.with_name(f"{output.stem}_summary{output.suffix}"),
            "verdict": output.with_name(f"{output.stem}_verdict{output.suffix}"),
        }

    _MONTHLY: ClassVar[dict[str, dict[str, str]]] = {
        "wet": {"capacity_factor": "1", "water_limited_capacity_ceiling_mw": "MW"},
        "wet_monthly": {
            "available_discharge_m3_per_s": "m3/s",
            "withdrawal_intensity_m3_per_mwh": "m3/MWh",
        },
        "dry": {"capacity_factor": "1"},
    }

    @cached_property
    def audit_monthly_run(self) -> Path:
        """Site evaluation carrying the monthly wet drivers; input of the split of the wet capacity loss."""

        config = self.audit_config(
            "audit_monthly", responses=self._MONTHLY, summary={"skip": True}
        )
        return ep.get("audit.run")(config)

    @cached_property
    def audit_baseline_run(self) -> Path:
        """Low-warming-level site evaluation that gives the reference capacity per unit flow for the split."""

        config = self.audit_config(
            "audit_baseline",
            responses=self._MONTHLY,
            summary={"skip": True},
            gwl_sampling={
                "target_levels": [1.0],
                "bin_width": 0.5,
                "years_per_bin": 2,
                "seed": 11,
            },
        )
        return ep.get("audit.run")(config)

    # --- the present-day reference calculation ----------------------------------------------------------------

    # The monthly drivers of the ``REF-PD`` site evaluation.
    _REF_PD_MONTHLY: ClassVar[dict[str, dict[str, str]]] = {
        "wet": {"capacity_factor": "1", "water_limited_capacity_ceiling_mw": "MW"},
        "wet_monthly": {
            "temperature_headroom_c": "C",
            "available_discharge_m3_per_s": "m3_per_s",
            "unit_available_discharge_m3_per_s": "m3_per_s",
            "withdrawal_intensity_m3_per_mwh": "m3_per_mwh",
            "wet_bulb_temperature": "C",
            "thermal_efficiency_fraction": "1",
            "heat_rejection_coefficient_fraction": "1",
            "evaporation_coefficient_fraction": "1",
        },
        "dry": {"capacity_factor": "1"},
    }

    @contextlib.contextmanager
    def short_reference_window(self):
        """Shorten the ``REF-PD`` window to the season-years the made-up tree holds.

        The present-day reference period starts in 1995 and a 20-year site
        evaluation takes minutes, so the first season-year is moved to
        ``fixtures.REFERENCE_FIRST_YEAR``, for the run driven by settings and for
        the library functions alike. Everything else about ``REF-PD`` is unchanged.
        """

        starts = ep.get("audit.REFERENCE_SAMPLE_STARTS")
        previous = starts["REF-PD"]
        starts["REF-PD"] = fixtures.REFERENCE_FIRST_YEAR
        try:
            yield
        finally:
            starts["REF-PD"] = previous

    def reference_pd_config(self, name: str) -> dict:
        """The settings of the ``REF-PD`` site evaluation on the made-up tree.

        The 1995-2014 historical window, wet cells taken from every scenario and the
        historical run, the existing sites of ssp585, the monthly drivers, and no
        summary. The made-up tree has no ssp370.
        """

        config = {
            "gcm": fixtures.GCM,
            "watxene_root": str(self.watxene_root),
            "reference_sampling": {"mode": "REF-PD"},
            "region": {key: str(path) for key, path in self.region_paths.items()},
            "universe": {"scenarios": [*fixtures.SCENARIOS, fixtures.HISTORICAL]},
            "fleet": {
                "gcm": fixtures.GCM,
                "scenario": "ssp585",
                "wet": str(self.fleet_files[0]),
                "dry": str(self.fleet_files[1]),
            },
            "responses": self._REF_PD_MONTHLY,
            "output": str(self.work / f"{name}.parquet"),
            "summary": {"skip": True},
        }
        # The order of the responses is the order of the rows.
        return config

    @cached_property
    def audit_reference_pd_run(self) -> Path:
        """The ``REF-PD`` monthly site evaluation, written from the ``REF-PD`` settings."""

        config = self.reference_pd_config(f"{fixtures.GCM}_REF-PD_monthly_census")
        with self.short_reference_window():
            return ep.get("audit.run")(config)

    @cached_property
    def reference_pd_decomposition_path(self) -> Path:
        """The ``REF-PD`` site evaluation with its wet capacity loss split, using itself as the reference.

        ``build_low_warming_reference`` on the site evaluation alone, then
        ``split_site_evaluations`` with the ``REF-PD`` label spelled out.
        """

        census = self.audit_reference_pd_run
        reference = ep.get("decomposition.build_low_gwl_reference")([census])
        return ep.get("decomposition.decompose_censuses")(
            [census],
            reference,
            self.work / f"{fixtures.GCM}_REF-PD_wet_channel_decomposition.parquet",
            reference="max_thermal_conductance_over_ref_pd_1995_2014",
        )

    # --- splitting the wet capacity loss, and the output files -------------------------------------------

    @cached_property
    def low_gwl_reference(self) -> pd.DataFrame:
        return ep.get("decomposition.build_low_gwl_reference")([self.audit_baseline_run])

    @cached_property
    def decomposition_path(self) -> Path:
        return ep.get("decomposition.decompose_censuses")(
            [self.audit_monthly_run],
            self.low_gwl_reference,
            self.work / f"{fixtures.GCM}_wet_channel_decomposition.parquet",
        )

    @cached_property
    def samples(self) -> dict[str, pd.DataFrame]:
        load = ep.get("export.load_decomposition_samples")
        return {
            population: load([self.decomposition_path], population=population)
            for population in ("candidate", "incumbent")
        }

    @cached_property
    def water_limits(self) -> pd.DataFrame:
        return ep.get("export.load_reach_water_ceilings")(
            [self.audit_monthly_run], population="candidate", preserve_year=True
        )

    @cached_property
    def topology_files(self) -> list[Path]:
        """One river-network parquet per scenario: a chain with a tributary and gaps."""

        cells = sorted(self.samples["candidate"]["cell_id"].unique())
        downstream: list[object] = []
        for position, _cell in enumerate(cells):
            if position == len(cells) - 1:
                downstream.append(None)
            elif position == 1:
                downstream.append(cells[position + 2] if position + 2 < len(cells) else None)
            elif position == 2:
                downstream.append("+999.000_+999.000")  # points outside the domain
            else:
                downstream.append(cells[position + 1])
        frame = pd.DataFrame(
            {
                "population": "candidate",
                "cell_id": cells,
                "downstream_cell_id": pd.Series(downstream, dtype=object),
            }
        )
        paths = []
        for scenario in fixtures.SCENARIOS:
            path = self.work / f"{fixtures.GCM}_{scenario}_wet_ucr_reach_topology.parquet"
            frame.to_parquet(path, index=False)
            paths.append(path)
        return paths

    @cached_property
    def topology(self) -> pd.DataFrame:
        return ep.get("export.load_topology")(self.topology_files, population="candidate")

    @cached_property
    def fixed_ordering(self) -> pd.DataFrame:
        return ep.get("export.build_fixed_cell_ordering")(self.samples["candidate"])

    @cached_property
    def fixed_ordering_path(self) -> Path:
        path = self.work / "fixed_wet_cell_ordering.parquet"
        self.fixed_ordering.to_parquet(path, index=False)
        return path

    # --- a second GCM, derived from the first ----------------------------------------

    def derived_gcm_inputs(self, gcm: str = GCM_B) -> dict[str, Path]:
        """Split wet capacity loss, site evaluation with water limits, and river network for another GCM.

        Values are scaled deterministically from the first GCM's so that the two
        GCMs rank cells differently once the water that existing sites take is
        subtracted. Only the ssp585 samples are kept because the residual-water
        order pairs one river-network file with one file of the split capacity loss.
        """

        rng = np.random.default_rng(31)
        directory = self.work / f"gcm_{gcm}"
        directory.mkdir(exist_ok=True)

        def keep(frame: pd.DataFrame) -> pd.DataFrame:
            return frame[frame["scenario"] == "ssp585"].reset_index(drop=True)

        decomposition = keep(pd.read_parquet(self.decomposition_path))
        cells = sorted(decomposition["cell_id"].unique())
        scale = dict(zip(cells, rng.uniform(0.6, 1.0, len(cells)), strict=True))
        factor = decomposition["cell_id"].map(scale)
        decomposition["gcm"] = gcm
        for column in ("thermal_loss_mw", "withdrawal_loss_mw"):
            decomposition[column] = decomposition[column] * (2.0 - factor)
        decomposition["usable_capacity_mw"] = decomposition["usable_capacity_mw"] * factor
        ceiling = keep(pd.read_parquet(self.audit_monthly_run))
        ceiling["gcm"] = gcm
        ceiling["value"] = ceiling["value"] * ceiling["cell_id"].map(scale)
        topology = pd.read_parquet(self.topology_files[1])
        paths = {
            "decomposition": directory / f"{gcm}_wet_channel_decomposition.parquet",
            "ceiling": directory / f"{gcm}_monthly_census.parquet",
            "topology": directory / f"{gcm}_ssp585_wet_ucr_reach_topology.parquet",
        }
        decomposition.to_parquet(paths["decomposition"], index=False)
        ceiling.to_parquet(paths["ceiling"], index=False)
        topology.to_parquet(paths["topology"], index=False)
        return paths

    @cached_property
    def first_gcm_ssp585_inputs(self) -> dict[str, Path]:
        """The first GCM's ssp585 inputs, filtered to the single-scenario form the residual-water order takes."""

        directory = self.work / f"gcm_{fixtures.GCM}"
        directory.mkdir(exist_ok=True)

        def keep(frame: pd.DataFrame) -> pd.DataFrame:
            return frame[frame["scenario"] == "ssp585"].reset_index(drop=True)

        paths = {
            "decomposition": directory / f"{fixtures.GCM}_wet_channel_decomposition.parquet",
            "ceiling": directory / f"{fixtures.GCM}_monthly_census.parquet",
            "topology": directory / f"{fixtures.GCM}_ssp585_wet_ucr_reach_topology.parquet",
        }
        keep(pd.read_parquet(self.decomposition_path)).to_parquet(paths["decomposition"], index=False)
        keep(pd.read_parquet(self.audit_monthly_run)).to_parquet(paths["ceiling"], index=False)
        pd.read_parquet(self.topology_files[1]).to_parquet(paths["topology"], index=False)
        return paths

    # --- inputs of the river-network and finalize steps ---------------------------------

    @cached_property
    def flow_direction_files(self) -> dict[str, Path]:
        return fixtures.write_flow_directions(self.root / "flowdir")

    @cached_property
    def finalize_task_layout(self) -> dict[str, Path]:
        """Five GCM task directories for each of the two calculations, for the finalize step."""

        return fixtures.write_finalize_task_layout(self.root / "finalize_tasks")
