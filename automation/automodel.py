"""Script for automating the creation of LISFLOOD models for an entire hydrofabric channel network."""

import os
import sqlite3
from pathlib import Path

import rasterio
import requests

from twodimfim.consts import NLCD_WMS_URL, USGS_3DEP_URL
from twodimfim.utils.etl import DatasetMetadata

os.environ["HYDROFABRIC_DIR"] = "/hydrofabric"

from twodimfim.models.data_models import (
    BoundaryCondition,
    HydraulicModel,
    HydraulicModelRun,
    ModelConnection,
    Roughness,
    Terrain,
)
from twodimfim.utils.network import NetworkWalker

ROOT = 69041  # Winooski River at Lake Champlain
VPU = 2
RESOLUTION = 10
DATA_DIR = Path("/data")
HYDROFABRIC_PATH = f"/hydrofabric/{str(VPU).rjust(2, '0')}_commmunity_nextgen.gpkg"
DB_PATH = "model_runs.db"
RI_LIST = ["Q100"]
LISFLOOD_RUNNER_URL = "http://lisflood-runner:5000"
DATA_DIR = Path("/data")


def q100(da_sqkm: float) -> float:
    """Estimate Q100 (m3/s) from drainage area (km2) using USGS regression for Vermont."""
    da_sqmi = da_sqkm / 2.58999
    q_cfs = 224 * (da_sqmi**0.807)
    return q_cfs / 35.3147


RI_FUNC_MAP = {"Q100": q100}

### DATABASE ###


class DataBaseConnection:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

        # Build tables
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS model_runs (
                    model_id INT,
                    run_id TEXT,
                    qin REAL,
                    ds_model_id TEXT,
                    processed_at TEXT,
                    successful INTEGER
                )"""
            )
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS models (
                    model_id INT PRIMARY KEY,
                    processed_at TEXT,
                    domain_ready INTEGER
                )"""
            )
            conn.commit()

    def add_model(self, model_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR IGNORE INTO models (model_id, processed_at, domain_ready)
                   VALUES (?, datetime('now'), 0)""",
                (model_id,),
            )
            conn.commit()

    def add_run(self, model_id: str, run_id: str, qin: float, ds_model_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO model_runs (model_id, run_id, qin, ds_model_id, processed_at, successful)
                   VALUES (?, ?, ?, ?, datetime('now'), 0)""",
                (model_id, run_id, qin, ds_model_id),
            )
            conn.commit()

    def get_next_run_ready(self) -> tuple[str, str, float, str] | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT a.model_id, a.run_id, a.qin, a.ds_model_id FROM model_runs a
                   LEFT JOIN model_runs ds ON a.ds_model_id = ds.model_id
                   WHERE a.successful = 0
                   AND (a.ds_model_id IS NULL OR ds.successful = 1)
                   ORDER BY a.processed_at ASC
                   LIMIT 1"""
            )
            row = cursor.fetchone()
            if row:
                return row  # type: ignore
            return None

    def get_model_domains_to_process(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT model_id FROM models
                   WHERE domain_ready = 0"""
            )
            rows = cursor.fetchall()
            return [row[0] for row in rows]  # type: ignore

    def mark_domain_ready(self, model_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE models
                   SET domain_ready = 1, processed_at = datetime('now')
                   WHERE model_id = ?""",
                (model_id,),
            )
            conn.commit()

    def mark_run_successful(self, model_id: str, run_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE model_runs
                   SET successful = 1, processed_at = datetime('now')
                   WHERE model_id = ? AND run_id = ?""",
                (model_id, run_id),
            )
            conn.commit()


def populate_database():
    """Traverse network, populating database with model runs to be executed."""
    walker = NetworkWalker(HYDROFABRIC_PATH)
    db = DataBaseConnection(DB_PATH)

    # Add root
    da_sqkm = walker.network[ROOT].da
    db.add_model(ROOT)
    for ri in RI_LIST:
        qin = RI_FUNC_MAP[ri](da_sqkm)
        db.add_run(
            model_id=ROOT,
            run_id=ri,
            qin=qin,
            ds_model_id=None,
        )

    # Add upstream
    for i in walker.walk_network_us(ROOT, 1e10):
        da_sqkm = walker.network[i].da
        db.add_model(i)
        for ri in RI_LIST:
            qin = RI_FUNC_MAP[ri](da_sqkm)
            db.add_run(
                model_id=i,
                run_id=ri,
                qin=qin,
                ds_model_id=walker.network[i].ds,
            )


def setup_domains():
    """Traverse network, generating model domains for each reach."""
    db = DataBaseConnection(DB_PATH)
    model_ids = db.get_model_domains_to_process()
    for model_id in model_ids:
        print(f"Processing domain for model {model_id}")
        model = HydraulicModel.from_hydrofabric(
            VPU, model_id, RESOLUTION, DATA_DIR / str(model_id)
        )
        domain = model.domains[next(iter(model.domains))]
        domain.load_3dep_terrain()
        domain.load_nlcd_roughness()
        model.save()
        db.mark_domain_ready(model_id)


def execute_runs():
    """Traverse network, running each model for each reach."""
    db = DataBaseConnection(DB_PATH)
    while True:
        next_run = db.get_next_run_ready()
        if next_run is None:
            print("No more runs to process.")
            break
        create_run(*next_run)
        if execute_steady_state_run(next_run[0], next_run[1]):
            db.mark_run_successful(next_run[0], next_run[1])


def create_run(model_id: int, run_id: str, qin: float, ds_model_id: int | None):
    """Create and execute a single model run for a given reach."""
    print(f"Processing model {model_id} run {run_id} with Q_in={qin} m3/s")
    model = HydraulicModel.from_file(DATA_DIR / str(model_id) / "model.json")
    # Get domain
    domain = next(iter(model.domains))

    ### DEBUGGING AD HOC FORCE TERRAIN AND ROUGHNESS ###
    # if model.domains[domain].terrain is None:
    #     terrain_meta = DatasetMetadata("url", USGS_3DEP_URL, transformations=[])
    #     terrain = Terrain(
    #         domain, f"rasters/{domain}_usgs_dem_10.ascii", "meters", terrain_meta
    #     )
    #     model.domains[domain].terrain = terrain
    # if model.domains[domain].roughness is None:
    #     rough_meta = DatasetMetadata("url", NLCD_WMS_URL, transformations=[])
    #     roughness = Roughness(
    #         domain, f"rasters/{domain}_nlcd_roughness_10.ascii", rough_meta
    #     )
    #     model.domains[domain].roughness = roughness

    # Establish connection with downstream model if applicable
    if ds_model_id is not None:
        model.add_connection(run_id, DATA_DIR / str(ds_model_id) / "model.json", run_id)

    # Establish boundary conditions
    bcs = []
    bcs.append(BoundaryCondition("us_bc_line", "QFIX", qin))
    if ds_model_id is not None:
        bcs.append(BoundaryCondition("transfer", "TRANSFER", run_id))
    bcs.append(BoundaryCondition("transfer_offset", "HFIX", -999))

    run = HydraulicModelRun(run_id, "unsteady", domain, bcs, sim_time=10000)
    model.add_run(run)
    model.save()


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
            print(f"Model API returned {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Lisflood API: {e}")
    return False


def model_converged(run_path: str, tolerance: float) -> bool:
    # Get all files matching the pattern run_path.parent/*-####.wd
    run_dir = Path(run_path).parent
    wd_files = sorted(run_dir.glob("*-????.wd"))
    if len(wd_files) < 2:
        return False  # Not enough files to compare
    latest_file = wd_files[-1]
    previous_file = wd_files[-2]

    # Read the last time step from both files
    latest_data = read_wd_file(latest_file)
    previous_data = read_wd_file(previous_file)

    # Calculate average depth and average depth change
    avg_depth = latest_data[latest_data > 0].mean()
    depth_change = abs(latest_data - previous_data)
    avg_depth_change = depth_change[latest_data > 0].mean()
    if avg_depth == 0:
        return False  # Avoid division by zero
    relative_change = avg_depth_change / avg_depth
    print(
        f"Average Depth: {avg_depth}, Average Depth Change: {avg_depth_change}, Relative Change: {relative_change}%"
    )

    return relative_change < tolerance


def read_wd_file(file_path: Path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read the first band
    return data


def execute_steady_state_run(
    model_id: int,
    run_id: str,
    tolerance: float = 0.05,
    inc: int = 900,
    max_iter: int = 30,
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
            print("Model run failed.")
            return False

        # Check for convergence
        if model_converged(run_path, tolerance):
            print(f"Model converged after {iter + 1} iterations.")
            return True

        iter += 1
        tmp_run = model.runs[run_id]
        tmp_run.sim_time += inc
        tmp_run.initial_state = tmp_run.depth_grid_path
        model.add_run(tmp_run)


def main():
    """Traverse network, generating LISFLOOD model for each reach and running it."""
    populate_database()
    setup_domains()
    execute_runs()


if __name__ == "__main__":
    main()
