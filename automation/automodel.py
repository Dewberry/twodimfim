"""Script for automating the creation of LISFLOOD models for an entire hydrofabric channel network."""

import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import requests
from shapely import unary_union

from twodimfim.consts import NLCD_WMS_URL, USGS_3DEP_URL
from twodimfim.utils.etl import DatasetMetadata
from twodimfim.utils.geospatial import BBox

os.environ["HYDROFABRIC_DIR"] = "/hydrofabric"

from twodimfim.models.data_models import (
    BoundaryCondition,
    HydraulicModel,
    HydraulicModelRun,
    ModelConnection,
    ModelDomain,
    Roughness,
    Terrain,
)
from twodimfim.utils.network import NetworkWalker

ROOT = 69041  # Winooski River at Lake Champlain
VPU = 2
RESOLUTION = 30
DATA_DIR = Path("/data")
HYDROFABRIC_PATH = f"/hydrofabric/{str(VPU).rjust(2, '0')}_commmunity_nextgen.gpkg"
DB_PATH = "model_runs.db"
RI_LIST = ["Q100"]
LARGEST_RI = "Q500"
PRELIM_RUN_ID = f"prelim_{LARGEST_RI}"
LISFLOOD_RUNNER_URL = "http://lisflood-runner:5000"
DATA_DIR = Path("/data")


def q2(da_sqkm: float) -> float:
    """Estimate Q100 (m3/s) from drainage area (km2) using USGS regression for Vermont."""
    da_sqmi = da_sqkm / 2.58999
    q_cfs = 52.6 * (da_sqmi**0.854)
    return q_cfs / 35.3147


def q10(da_sqkm: float) -> float:
    """Estimate Q100 (m3/s) from drainage area (km2) using USGS regression for Vermont."""
    da_sqmi = da_sqkm / 2.58999
    q_cfs = 113 * (da_sqmi**0.829)
    return q_cfs / 35.3147


def q100(da_sqkm: float) -> float:
    """Estimate Q100 (m3/s) from drainage area (km2) using USGS regression for Vermont."""
    da_sqmi = da_sqkm / 2.58999
    q_cfs = 224 * (da_sqmi**0.807)
    return q_cfs / 35.3147


def q500(da_sqkm: float) -> float:
    """Estimate Q100 (m3/s) from drainage area (km2) using USGS regression for Vermont."""
    da_sqmi = da_sqkm / 2.58999
    q_cfs = 330 * (da_sqmi**0.795)
    return q_cfs / 35.3147


RI_FUNC_MAP = {"Q100": q100, "Q500": q500}

### LOGGING ###


def initialize_logging():
    """Create logger that writes to file and console."""
    logger = logging.getLogger("automodel")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler("automodel.log")
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


initialize_logging()
logger = logging.getLogger("automodel")


### CUSTOM ERRORS ###


class NonConvergentModelError(Exception):
    """Raised when a model fails to converge within the maximum number of iterations."""


class WaterOnInvalidBoundaryError(Exception):
    """Raised when water is detected on an invalid boundary in the model."""


### DATABASE ###


class DataBaseConnection:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

        # Build tables
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS models (
                    model_id INT PRIMARY KEY,
                    model_path TEXT,
                    da_sqkm real,
                    ds_id TEXT
                )"""
            )
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS domains (
                    model_id INT,
                    domain_id TEXT,
                    domain_ready INTEGER,
                    processed_at TEXT,
                    PRIMARY KEY (model_id, domain_id),
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )"""
            )
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS model_runs (
                    model_id INT,
                    run_id TEXT,
                    processed_at TEXT,
                    status TEXT,
                    model_dependency INTEGER,
                    run_dependency TEXT,
                    PRIMARY KEY (model_id, run_id),
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )"""
            )
            conn.commit()

    def add_model(
        self, model_id: int, model_path: str, da_sqkm: float, ds_id: int | None = None
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO models (model_id, model_path, da_sqkm, ds_id)
                   VALUES (?, ?, ?, ?)""",
                (model_id, model_path, da_sqkm, ds_id),
            )
            conn.commit()

    def add_domain(self, model_id: str, domain_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO domains (model_id, domain_id, domain_ready, processed_at)
                   VALUES (?, ?, 0, datetime('now'))""",
                (model_id, domain_id),
            )
            conn.commit()

    def add_run(
        self,
        model_id: str,
        run_id: str,
        model_dependency: int | None = None,
        run_dependency: str | None = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO model_runs (model_id, run_id, processed_at, status, model_dependency, run_dependency)
                   VALUES (?, ?, datetime('now'), 'created', ?, ?)""",
                (model_id, run_id, model_dependency, run_dependency),
            )
            conn.commit()

    def get_next_run_ready(self, run_filter: str) -> tuple[int, str] | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT mr.model_id, mr.run_id
                   FROM model_runs mr
                   LEFT JOIN model_runs dep
                   ON mr.model_dependency = dep.model_id AND mr.run_dependency = dep.run_id
                   WHERE mr.status = 'created'
                   AND (mr.model_dependency IS NULL OR dep.status = 'success')
                   AND mr.run_id = ?
                   ORDER BY mr.model_id ASC, mr.run_id ASC
                   LIMIT 1""",
                (run_filter,),
            )
            return cursor.fetchone()

    def get_models(self) -> list[tuple[int, str, float, int]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT model_id, model_path, da_sqkm, ds_id FROM models")
            res = cursor.fetchall()
            # TEMP FIX LATER
            res = [
                (int(a), b, float(c), int(d) if d is not None else None)
                for a, b, c, d in res
            ]
            return res

    def update_run_status(
        self, model_id: str, run_id: str, status: Literal["success", "failure"]
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE model_runs
                   SET status = ?, processed_at = datetime('now')
                   WHERE model_id = ? AND run_id = ?""",
                (status, model_id, run_id),
            )
            conn.commit()


### AUTOMATION FUNCTIONS ###


def execute_runs(run_id: str, allow_domain_refinement: bool = True):
    """Traverse network, running each model for each reach."""
    db = DataBaseConnection(DB_PATH)
    while True:
        next_run = db.get_next_run_ready(run_id)
        if next_run is None:
            logger.info("No more runs to process.")
            break
        try:
            logger.info(f"Executing model {next_run[0]} run {next_run[1]}")
            execute_steady_state_run(next_run[0], next_run[1], tolerance=0.02)
            db.update_run_status(next_run[0], next_run[1], "success")
        except WaterOnInvalidBoundaryError as e:
            logger.info(f"Model {next_run[0]} run {next_run[1]} failed: {e}")
            if allow_domain_refinement:
                modify_domain_and_rerun(next_run[0], next_run[1])
            else:
                db.update_run_status(next_run[0], next_run[1], "failure")
        except Exception as e:
            logger.info(f"Model {next_run[0]} run {next_run[1]} failed with error: {e}")
            db.update_run_status(next_run[0], next_run[1], "failure")


def execute_run(run_path) -> bool:
    """Execute a single model run."""
    try:
        response = requests.post(
            f"{LISFLOOD_RUNNER_URL}/run_model", json={"model_dir": run_path}
        )
        if response.status_code == 200:
            result = response.json()
            return True
        else:
            logger.info(f"Model API returned {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.info(f"Error connecting to Lisflood API: {e}")
    return False


def execute_steady_state_run(
    model_id: int,
    run_id: str,
    tolerance: float = 0.05,
    inc: int = 900,
    max_iter: int = 60,
    raise_on_invalid_boundary: bool = True,
) -> bool:
    """Execute a steady state model run."""
    model = HydraulicModel.from_file(DATA_DIR / str(model_id) / "model.json")
    tmp_run = model.runs[run_id]
    tmp_run.mass_interval = 60  # 1 minutes
    tmp_run.save_interval = 300  # 5 minutes
    tmp_run.sim_time = inc
    model.add_run(tmp_run)

    run_path = str(model.runs[run_id].parfile_path)

    iter = 0
    while iter < max_iter:
        success = execute_run(run_path)
        if not success:
            logger.info("Model run failed.")
            return False

        # Check for water on invalid boundary, if necessary
        if (
            all(bc.geometry_vector != "all" for bc in tmp_run.boundary_conditions)
            and raise_on_invalid_boundary
        ):
            # NOTE: this will catch reaches that hit midway down the reach, which may not be ideal behavior
            # We may want to do something like all_us_ms_divides instead
            if "all_us_divides" in model.vectors:
                valid_pool_poly = unary_union(
                    [
                        model.vectors["all_ds_divides"].shape,
                        model.vectors["all_us_divides"].shape,
                    ]
                )
            else:
                valid_pool_poly = model.vectors["all_ds_divides"].shape
            if tmp_run.water_on_invalid_boundary(valid_pool_poly):
                raise WaterOnInvalidBoundaryError(
                    f"Water on invalid boundary in model {model_id} run {run_id}."
                )
        # Check for convergence
        if tmp_run.is_converged(tolerance, "max_depth_change"):
            logger.info(f"Model converged after {iter + 1} iterations.")
            return True

        iter += 1
        tmp_run = model.runs[run_id]
        tmp_run.sim_time += inc
        tmp_run.initial_state = tmp_run.depth_file_paths[-1]
        model.add_run(tmp_run)
    else:
        raise NonConvergentModelError(
            f"Model {model_id} run {run_id} did not converge after {max_iter} iterations."
        )


def modify_domain_and_rerun(model_id: int, run_id: str):
    """Modify the model domain to attempt to fix convergence issues and rerun."""
    logger.info(f"Modifying domain for model {model_id} to attempt to fix convergence.")
    model = HydraulicModel.from_file(DATA_DIR / str(model_id) / "model.json")
    run = model.runs[run_id]
    domain = model.domains[run.domain]

    # Find bad edges
    valid_pool_poly = unary_union(
        [
            model.vectors["all_ds_divides"].shape,
            model.vectors["all_us_divides"].shape,
        ]
    )
    edges = run.check_water_on_invalid_boundaries(valid_pool_poly)

    # Modify bbox
    bbox = domain.bbox
    for direction, has_water in edges.items():
        if has_water:
            if direction == "north":
                bbox.ymax += bbox.height * 0.5
            elif direction == "south":
                bbox.ymin -= bbox.height * 0.5
            elif direction == "west":
                bbox.xmin -= bbox.width * 0.5
            elif direction == "east":
                bbox.xmax += bbox.width * 0.5

    # Recreate domain
    new_domain = ModelDomain.from_bbox(
        run.domain, bbox, domain.resolution, model._context
    )
    new_domain.load_3dep_terrain()
    new_domain.load_nlcd_roughness()
    model.domains[run.domain] = new_domain

    # Recreate run
    run.initial_state = None
    shutil.rmtree(run.run_dir, ignore_errors=True)
    model.add_run(run)
    model.save()

    # Update database
    db = DataBaseConnection(DB_PATH)
    db.add_domain(model_id, run.domain)
    db.update_run_status(model_id, run_id, "created")


######


def setup_models(vpu: str, root: int):
    """Traverse network, generating models for each reach."""
    # Connect to data
    walker = NetworkWalker(HYDROFABRIC_PATH)
    db = DataBaseConnection(DB_PATH)

    # Get reaches
    reaches = walker.walk_network_us(root, 1e10)
    # reaches = walker.walk_network_us_ms(root, 1e10)  # temporary (limit to mainstem)
    reaches.insert(0, root)

    # Make models and log
    for i in reaches:
        da_sqkm = walker.network[i].da
        if i == root:
            ds_id = None
        else:
            ds_id = walker.network[i].ds
        model_path = DATA_DIR / str(i)

        if (model_path / "model.json").exists():
            logger.info(f"Model for reach {i} already exists, skipping.")
        else:
            logger.info(f"Creating model for reach {i}.")
            model = HydraulicModel.from_hydrofabric(vpu, i, RESOLUTION, model_path)
            del model.domains[f"hydrofabric_res_{RESOLUTION}"]  # All custom
            model.save()
        db.add_model(i, str(model_path / "model.json"), da_sqkm, ds_id)


def setup_default_domains(resolution: float = 10, buffer: float = 100):
    """Generate domains from hydrofabric bboxes plus a buffer."""
    db = DataBaseConnection(DB_PATH)
    model_ids = db.get_models()
    for model_id, model_path, _, _ in model_ids:
        logger.info(f"Processing default domain for model {model_id}")
        model = HydraulicModel.from_file(model_path)
        idx = f"default_{resolution}"
        if idx in model.domains:
            logger.info(f"Model {model_id} already has domain {idx}, skipping.")
            continue

        logger.info(f"Creating domain {idx} for model {model_id}.")
        base_bbox = BBox(
            *unary_union(
                [model.vectors["divide"].shape, model.vectors["us_bc_line"].shape]
            ).bounds
        )
        base_bbox.buffer(buffer)

        domain = ModelDomain.from_bbox(idx, base_bbox, resolution, model._context)
        domain.load_3dep_terrain()
        domain.load_nlcd_roughness()
        model.domains[idx] = domain
        model.save()
        db.add_domain(model_id, idx)


def setup_default_runs(overwrite: bool = False):
    """Create default model runs for each model in the database."""
    db = DataBaseConnection(DB_PATH)
    model_ids = db.get_models()
    for model_id, model_path, da_sqkm, ds_id in model_ids:
        logger.info(f"Creating default runs for model {model_id}")
        model = HydraulicModel.from_file(model_path)
        if PRELIM_RUN_ID in model.runs and not overwrite:
            logger.info(f"Model {model_id} already has run {PRELIM_RUN_ID}, skipping.")
            continue
        logger.info(f"Creating run {PRELIM_RUN_ID} for model {model_id}")
        qin = RI_FUNC_MAP[LARGEST_RI](da_sqkm)
        if "default_30" in model.domains:
            create_run(model_id, PRELIM_RUN_ID, "default_30", qin, ds_id, "open")
            db.add_run(model_id, PRELIM_RUN_ID)
        else:
            raise RuntimeError(
                f"Model {model_id} does not have default domain for preliminary run."
            )


def setup_and_execute_production_runs(
    ris: list[str] = RI_LIST, overwrite: bool = False
):
    ### This script is messy and will need a refactor later ###

    db = DataBaseConnection(DB_PATH)
    model_ids = {a: (a, b, c, d) for a, b, c, d in db.get_models()}
    cur_reach = ROOT
    while cur_reach is not None:
        model_id, model_path, da_sqkm, ds_id = model_ids[cur_reach]

        # Build runs
        logger.info(f"Creating production runs for model {model_id}")
        model = HydraulicModel.from_file(model_path)
        q = []
        for ri in ris:
            if ri in model.runs and not overwrite:
                logger.info(f"Model {model_id} already has run {ri}, skipping.")
                continue
            logger.info(f"Creating run {ri} for model {model_id}")
            qin = RI_FUNC_MAP[ri](da_sqkm)
            if "refined_10" in model.domains:
                create_run(
                    model_id,
                    ri,
                    "refined_10",
                    qin,
                    ds_id,
                    "transfer",
                    clear_previous=overwrite,
                )
                db.add_run(model_id, ri, ds_id, ri)
                q.append(ri)
            else:
                raise RuntimeError(
                    f"Model {model_id} does not have refined domain for production run."
                )

        # Execute runs
        for ri in q:
            logger.info(f"Executing production run {ri} for model {model_id}")
            try:
                execute_steady_state_run(model_id, ri, raise_on_invalid_boundary=False)
                db.update_run_status(model_id, ri, "success")
            except Exception as e:
                logger.info(f"Model {model_id} run {ri} failed with error: {e}")
                db.update_run_status(model_id, ri, "failure")

        # Move to downstream reach
        for model_id, _, _, ds_id in model_ids.values():
            if ds_id == cur_reach:
                cur_reach = model_id
                break
        else:
            cur_reach = None


def create_run(
    model_id: int,
    run_id: str,
    domain: str,
    qin: float,
    ds_model_id: int | None,
    ds_bc_type: Literal["open", "transfer"] = "transfer",
    clear_previous: bool = False,
):
    """Create and execute a single model run for a given reach."""
    model = HydraulicModel.from_file(DATA_DIR / str(model_id) / "model.json")

    # Establish connection with downstream model if applicable
    if ds_model_id is not None:
        model.add_connection(run_id, DATA_DIR / str(ds_model_id) / "model.json", run_id)

    # Establish boundary conditions
    if ds_bc_type == "open":
        bcs = open_ds_bc_2(qin)
    elif ds_bc_type == "transfer":
        bcs = bcs_from_ds_model(qin, ds_model_id, run_id, model)
    else:
        raise ValueError(f"Unknown ds_bc_type: {ds_bc_type}")

    run = HydraulicModelRun(
        run_id, "unsteady", domain, bcs, sim_time=10000, _context=model._context
    )
    # run.use_cuda = False
    if clear_previous:
        shutil.rmtree(run.run_dir, ignore_errors=True)
    model.add_run(run)
    model.save()


def bcs_from_ds_model(
    qin: float, ds_model_id: int | None, run_id: str, model: HydraulicModel
) -> list[BoundaryCondition]:
    bcs = []
    bcs.append(BoundaryCondition("us_bc_line", "QFIX", qin))
    if ds_model_id is not None:
        if "transfer" in model.vectors:
            bcs.append(BoundaryCondition("transfer", "TRANSFER", run_id))
    if "transfer_offset" in model.vectors:
        bcs.append(BoundaryCondition("transfer_offset", "HFIX", -999))
    else:
        bcs.append(BoundaryCondition("all", "HFIX", -999))
    return bcs


def open_ds_bc(qin: float) -> list[BoundaryCondition]:
    return [
        BoundaryCondition("us_bc_line", "QFIX", qin),
        BoundaryCondition("all", "HFIX", -999),
    ]


def open_ds_bc_2(qin: float) -> list[BoundaryCondition]:
    return [
        BoundaryCondition("us_bc_line", "QFIX", qin),
        BoundaryCondition("transfer", "HFIX", -999),
    ]


def setup_refined_domains(resolution: float = 10, buffer: float = 100):
    """Generate refined domains for each model in the database."""
    db = DataBaseConnection(DB_PATH)
    model_ids = db.get_models()
    for model_id, model_path, _, _ in model_ids:
        logger.info(f"Processing refined domain for model {model_id}")
        model = HydraulicModel.from_file(model_path)
        idx = f"refined_{resolution}"
        if idx in model.domains:
            logger.info(f"Model {model_id} already has domain {idx}, skipping.")
            continue

        logger.info(f"Creating domain {idx} for model {model_id}.")
        tmp_run = model.runs[PRELIM_RUN_ID]
        with rasterio.open(tmp_run.depth_file_paths[-1]) as src:
            arr = src.read(1)
            transform = src.transform
        rows, cols = np.where(arr > 0.01)
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        xmin = min(xs) - buffer
        xmax = max(xs) + buffer
        ymin = min(ys) - buffer
        ymax = max(ys) + buffer
        refined_bbox = BBox(xmin, ymin, xmax, ymax)

        domain = ModelDomain.from_bbox(idx, refined_bbox, resolution, model._context)
        domain.load_3dep_terrain()
        domain.load_nlcd_roughness()
        model.domains[idx] = domain
        model.save()
        db.add_domain(model_id, idx)


def main():
    """Traverse network, generating LISFLOOD model for each reach and running it."""
    setup_models(VPU, ROOT)
    setup_default_domains(RESOLUTION)
    setup_default_runs(overwrite=False)
    execute_runs("prelim_Q500")
    setup_refined_domains(10)
    setup_and_execute_production_runs(RI_LIST, overwrite=True)


if __name__ == "__main__":
    main()
