import json
from dataclasses import InitVar, asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal, cast

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from shapely import Polygon, unary_union
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from twodimfim.consts import (
    BCI_FILE,
    COMMON_CRS,
    MANNINGS_LC_LOOKUP,
    NLCD_WMS_URL,
    PAR_FILE,
    USGS_3DEP_URL,
)
from twodimfim.hydrofabric.data_models import ReachContext
from twodimfim.utils.etl import get_nlcd_mannings, get_usgs_dem
from twodimfim.utils.geospatial import (
    BBox,
    poly_to_edges,
    rasterize_line,
    snap_bbox_to_grid,
)

DEFAULT_RASTER_DIR = "rasters"
DEFAULT_VECTOR_DIR = "vectors"
DEFAULT_RUN_DIR = "runs"

BCType = LineString | Point


@dataclass
class VectorDataset:
    path: str

    @cached_property
    def gdf(self) -> gpd.GeoDataFrame:
        if self.path.endswith(".parquet"):
            return gpd.read_parquet(self.path)
        else:
            return gpd.read_file(self.path)

    @cached_property
    def shape(self) -> BaseGeometry:
        return self.gdf.iloc[0].geometry


@dataclass
class BoundaryCondition:
    geometry_vector: str
    type_: Literal["QFIX", "HFIX", "FREE"]
    value: float | str


@dataclass
class Terrain:
    path: str
    vertical_units: Literal["feet", "meters"]
    metadata: dict


@dataclass
class Roughness:
    path: str
    metadata: dict


@dataclass
class HydraulicModelContext:
    model_root: Path
    crs: CRS

    def to_dict(self) -> dict:
        return {
            "model_root": str(self.model_root),
            "crs": self.crs.to_wkt(),
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(model_root=Path(d["model_root"]), crs=CRS.from_user_input(d["crs"]))


@dataclass
class HydraulicModelRun:
    idx: str
    type_: Literal["unsteady", "quasi-steady"]
    domain: str
    boundary_conditions: list[BoundaryCondition]
    save_interval: float = 900
    mass_interval: float = 15
    sim_time: float | None = None
    steady_state_tolerance: float | None = None
    initial_tstep: float = 0.5
    initial_state: str | None = None
    run_dir: str | None = None
    par_path: str | None = None
    bci_path: str | None = None

    def __post_init__(self):
        self.boundary_conditions = [
            BoundaryCondition(**i) for i in self.boundary_conditions
        ]

@dataclass
class ModelDomain:
    idx: str
    transform: Affine
    rows: int
    cols: int
    resolution: float
    terrain: Terrain | None = None
    roughness: Roughness | None = None
    _context: HydraulicModelContext | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def from_bbox(cls, idx: str, bbox: BBox, resolution: float):
        bbox = snap_bbox_to_grid(bbox, resolution)

        affine = Affine(resolution, 0, bbox.xmin, 0, -resolution, bbox.ymax)
        cols = int((bbox.xmax - bbox.xmin) / resolution)
        rows = int((bbox.ymax - bbox.ymin) / resolution)
        return cls(idx, affine, rows, cols, resolution)

    @classmethod
    def from_dict(cls, d: dict) -> dict:
        d["terrain"] = Terrain(**d["terrain"]) if d["terrain"] is not None else None
        d["roughness"] = (
            Roughness(**d["roughness"]) if d["roughness"] is not None else None
        )
        d["transform"] = Affine(**d["transform"])
        return cls(**d)

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "transform": self.transform._asdict(),
            "rows": self.rows,
            "cols": self.cols,
            "resolution": self.resolution,
            "terrain": asdict(self.terrain) if self.terrain is not None else None,
            "roughness": asdict(self.roughness) if self.roughness is not None else None,
        }

    @property
    def crs(self) -> CRS:
        if self._context is None:
            raise RuntimeError("Context not attached")
        return self._context.crs

    @property
    def model_root(self) -> Path:
        if self._context is None:
            raise RuntimeError("Context not attached")
        return self._context.model_root

    @property
    def bbox(self) -> BBox:
        transform = self.transform
        corners = [
            (0, 0),
            (self.cols, 0),
            (0, self.rows),
            (self.cols, self.rows),
        ]
        xs, ys = zip(*(transform * corner for corner in corners))
        return BBox(min(xs), min(ys), max(xs), max(ys))

    def load_3dep_terrain(self) -> None:
        save_path = (
            self.model_root
            / DEFAULT_RASTER_DIR
            / f"{self.idx}_usgs_dem_{self.resolution}.ascii"
        )
        get_usgs_dem(
            save_path,
            self.bbox,
            self.cols,
            self.rows,
            ":".join(self.crs.to_authority()),
        )
        self.terrain = Terrain(str(save_path), "feet", {"source": USGS_3DEP_URL})

    def load_nlcd_roughness(self) -> None:
        save_path = (
            self.model_root
            / DEFAULT_RASTER_DIR
            / f"{self.idx}_nlcd_roughness_{self.resolution}.ascii"
        )
        get_nlcd_mannings(
            save_path,
            self.bbox,
            self.cols,
            self.rows,
            ":".join(self.crs.to_authority()),
        )
        self.roughness = Roughness(
            str(save_path),
            {"source": NLCD_WMS_URL, "landcover_mannings_lookup": MANNINGS_LC_LOOKUP},
        )

    def geometry_to_bc_points(
        self, geometry: LineString | Polygon | Point
    ) -> list[tuple[str, float, float]]:
        if isinstance(geometry, Point):
            return [("P", geometry.x, geometry.y)]
        elif isinstance(geometry, LineString):
            pts = rasterize_line(geometry, self.rows, self.cols, self.transform)
            return [("P", i[0], i[1]) for i in pts]
        elif isinstance(geometry, Polygon):
            return poly_to_edges(geometry, self.bbox)


@dataclass
class HydraulicModel:
    context: HydraulicModelContext
    domains: dict[str, ModelDomain] = field(default_factory=dict)
    vectors: dict[str, VectorDataset] = field(default_factory=dict)
    runs: dict[str, HydraulicModelRun] = field(default_factory=dict)
    identifiers: list[str] = field(default_factory=list)
    notes: str = field(default_factory=str)

    def __post_init__(self):
        for i in self.domains.values():
            # Model domain needs access to some properties, but we don't want to duplicate that field in storage
            # Inject context from parent at run time instead
            i._context = self.context

    @staticmethod
    def init_model_dir(dir: str | Path) -> None:
        root = Path(dir)
        root.mkdir(exist_ok=True, parents=True)

        (root / DEFAULT_RASTER_DIR).mkdir(exist_ok=True)
        (root / DEFAULT_VECTOR_DIR).mkdir(exist_ok=True)
        (root / DEFAULT_RUN_DIR).mkdir(exist_ok=True)

    @classmethod
    def from_hydrofabric(
        cls,
        vpu: int,
        reach_id: int,
        resolution: float,
        model_root: str | Path,
        walk_us_dist_pct: float = 0.25,
        inflow_width: float = 10,
        domain_buffer: float = 100,
        crs: str = COMMON_CRS,
    ):
        # Initialize model directory
        context = HydraulicModelContext(Path(model_root), CRS.from_user_input(crs))
        cls.init_model_dir(context.model_root)

        # Populate model geometry files
        vector_dir = context.model_root / DEFAULT_VECTOR_DIR
        reach_context = ReachContext(vpu, reach_id).export_default_domain(
            vector_dir, walk_us_dist_pct, inflow_width
        )
        vectors = {k: VectorDataset(v) for k, v in reach_context.items()}

        # Make model domain
        base_bbox = BBox(
            *unary_union([vectors["divide"].shape, vectors["us_bc_line"].shape]).bounds
        )
        base_bbox.buffer(domain_buffer)
        idx = f"hydrofabric_res_{str(resolution)}"
        domain = ModelDomain.from_bbox(idx, base_bbox, resolution)
        domain._context = context

        # Make model
        return cls(
            context=context,
            domains={f"hydrofabric_{round(resolution, 1)}": domain},
            vectors=vectors,
        )

    @classmethod
    def from_dict(cls, d: dict):
        context = HydraulicModelContext.from_dict(d["context"])
        domains = (
            {k: ModelDomain.from_dict(v) for k, v in d["domains"].items()}
            if d["domains"] is not None
            else None
        )
        vectors = (
            {k: VectorDataset(**v) for k, v in d["vectors"].items()}
            if d["vectors"] is not None
            else None
        )
        runs = (
            {k: HydraulicModelRun(**v) for k, v in d["runs"].items()}
            if d["runs"] is not None
            else None
        )
        runs = {}
        return cls(context, domains, vectors, runs, d["identifiers"], d["notes"])

    @classmethod
    def from_file(cls, in_path: str | Path):
        with open(in_path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    def to_dict(self) -> dict:
        return {
            "context": self.context.to_dict(),
            "domains": {k: v.to_dict() for k, v in self.domains.items()},
            "vectors": {k: asdict(v) for k, v in self.vectors.items()},
            "runs": {k: asdict(v) for k, v in self.runs.items()},
            "identifiers": self.identifiers,
            "notes": self.notes,
        }

    def to_file(self, out_path: str | Path) -> None:
        with open(out_path, mode="w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def save(self) -> None:
        out_path = self.context.model_root / "model.json"
        self.to_file(out_path)

    def build_run(self, run: str) -> None:
        # TODO: will need to expand this for other models.
        # TODO: this function is too long and doesn't belong here.
        run_inst = self.runs[run]
        domain = self.domains[run_inst.domain]
        run_dir = Path(run_inst.run_dir)
        run_dir.mkdir(exist_ok=True, parents=True)

        # Generate boundary conditions
        bc_lines = []
        for i in run_inst.boundary_conditions:
            row_pts = domain.geometry_to_bc_points(
                self.vectors[i.geometry_vector].shape
            )
            if i.type_ == "QFIX":
                tmp_val = i.value / (domain.resolution * len(row_pts))
            else:
                tmp_val = i.value
            for j in row_pts:
                bc = " ".join(
                    [
                        j[0],
                        str(round(j[1], 4)),
                        str(round(j[2], 4)),
                        i.type_,
                        str(tmp_val),
                        "\n",
                    ]
                )
                bc_lines.append(bc)
            with open(run_inst.bci_path, mode="w+") as f:
                f.writelines(bc_lines)

        # Generate parameter file
        cfg = {
            "resroot": run_inst.idx,
            "dirroot": run_inst.run_dir,
            "DEMfile": domain.terrain.path,
            "manningfile": domain.roughness.path,
            "bcifile": str(run_inst.bci_path),
            "saveint": run_inst.save_interval,
            "massint": run_inst.mass_interval,
            "sim_time": run_inst.sim_time,
            "initial_tstep": run_inst.initial_tstep,
            "acceleration": "",
            "elevoff": "",
        }
        with open(run_inst.par_path, mode="w") as f:
            for k, v in cfg.items():
                f.write(f"{k} {v}\n")

    def add_run(self, run: HydraulicModelRun) -> None:
        run.run_dir = str(self.context.model_root / DEFAULT_RUN_DIR / run.idx)
        run.par_path = str(self.context.model_root / DEFAULT_RUN_DIR / run.idx / PAR_FILE)
        run.bci_path = str(self.context.model_root / DEFAULT_RUN_DIR / run.idx / BCI_FILE)
        self.runs[run.idx] = run
        self.build_run(run.idx)

    def execute_run(self, run: str) -> None:
        pass
