import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from re import S
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from affine import Affine
from numcodecs import Blosc
from pyproj import CRS
from rasterio.errors import RasterioIOError
from rasterio.features import shapes
from shapely import MultiLineString, MultiPolygon, Polygon, from_geojson, unary_union
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge

from twodimfim import __version__ as self_version
from twodimfim.consts import (
    BCI_FILE,
    COMMON_CRS,
    DEFAULT_MODEL_PATH_NAME,
    DEFAULT_RASTER_DIR,
    DEFAULT_RUN_DIR,
    DEFAULT_VECTOR_DIR,
    ELEVOFF,
    PAR_FILE,
    USE_CUDA,
    BCType,
    ModelPathTypes,
    RunType,
    SupportedModels,
    UnitsType,
)
from twodimfim.hydrofabric.data_models import ReachContext
from twodimfim.models.lisflood import write_bci_file, write_par_file
from twodimfim.utils.etl import DatasetMetadata, get_nlcd_mannings, get_usgs_dem
from twodimfim.utils.geospatial import (
    BBox,
    Raster,
    poly_to_edges,
    rasterize_geometry,
    sample_raster,
    sample_wse_from_depth_el,
    snap_bbox_to_grid,
    water_on_invalid_boundary,
)


@dataclass
class HydraulicModelMetadata:
    title: str
    author: str = ""
    model_brand: SupportedModels = "LISFLOOD-FP"
    engineer_notes: str = ""
    tags: list[str] = field(default_factory=list)
    path_types: ModelPathTypes = "relative"
    twodimfim_version: str = self_version
    creation_date: datetime = field(default_factory=datetime.now)
    last_edited: datetime = field(default_factory=datetime.now)
    crs: CRS = field(default_factory=lambda: CRS.from_user_input(COMMON_CRS))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["creation_date"] = self.creation_date.isoformat()
        d["last_edited"] = self.last_edited.isoformat()
        d["crs"] = self.crs.to_wkt()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        d["creation_date"] = datetime.fromisoformat(d["creation_date"])
        d["last_edited"] = datetime.fromisoformat(d["last_edited"])
        d["crs"] = CRS.from_wkt(d["crs"])
        return cls(**d)


@dataclass
class HydraulicModelContext:
    model_root: Path
    crs: CRS

    @property
    def crs_authority_str(self) -> str:
        return ":".join(self.crs.to_authority())


def generate_generic_context():
    return HydraulicModelContext(Path(".").resolve(), CRS(COMMON_CRS))


@dataclass
class VectorDataset:
    idx: str
    path_stem: str
    metadata: DatasetMetadata
    _context: HydraulicModelContext = field(default_factory=generate_generic_context)

    @cached_property
    def path(self) -> Path:
        return self._context.model_root / self.path_stem

    @cached_property
    def gdf(self) -> gpd.GeoDataFrame:
        if self.path.suffix == ".parquet":
            return gpd.read_parquet(self.path)
        else:
            return gpd.read_file(self.path)

    @cached_property
    def shape(self) -> BaseGeometry:
        return self.gdf.iloc[0].geometry

    @cached_property
    def geom_type(self) -> str:
        return self.shape.geom_type

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        del d["_context"]
        return d


@dataclass
class BoundaryCondition:
    geometry_vector: str
    bc_type: BCType
    value: float | str


@dataclass
class Terrain:
    idx: str
    path_stem: str
    vertical_units: UnitsType
    metadata: DatasetMetadata
    _context: HydraulicModelContext = field(default_factory=generate_generic_context)

    @cached_property
    def path(self) -> Path:
        return self._context.model_root / self.path_stem

    @classmethod
    def from_3dep(
        cls,
        domain_idx: str,
        resolution: float,
        bbox: BBox,
        cols: int,
        rows: int,
        units: UnitsType = "meters",
        context: HydraulicModelContext = field(
            default_factory=generate_generic_context
        ),
    ):
        idx = f"{domain_idx}_usgs_dem_{resolution}"
        save_path = context.model_root / DEFAULT_RASTER_DIR / f"{idx}.ascii"
        path_stem = save_path.relative_to(context.model_root)
        meta = get_usgs_dem(
            save_path, bbox, cols, rows, context.crs_authority_str, units
        )
        return cls(idx, str(path_stem), units, meta, context)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        del d["_context"]
        return d


@dataclass
class Roughness:
    idx: str
    path_stem: str
    metadata: DatasetMetadata
    _context: HydraulicModelContext = field(default_factory=generate_generic_context)

    @cached_property
    def path(self) -> Path:
        return self._context.model_root / self.path_stem

    @classmethod
    def from_nlcd(
        cls,
        domain_idx: str,
        resolution: float,
        bbox: BBox,
        cols: int,
        rows: int,
        context: HydraulicModelContext = field(
            default_factory=generate_generic_context
        ),
    ):
        idx = f"{domain_idx}_nlcd_roughness_{resolution}"
        save_path = context.model_root / DEFAULT_RASTER_DIR / f"{idx}.ascii"
        path_stem = save_path.relative_to(context.model_root)
        meta = get_nlcd_mannings(save_path, bbox, cols, rows, context.crs_authority_str)
        return cls(idx, str(path_stem), meta, context)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        del d["_context"]
        return d


@dataclass
class HydraulicModelRun:
    idx: str
    run_type: RunType
    domain: str
    boundary_conditions: list[BoundaryCondition]
    save_interval: float = 900
    mass_interval: float = 15
    sim_time: float | None = None
    steady_state_tolerance: float | None = None
    initial_tstep: float = 0.5
    initial_state: str | None = None
    run_dir_stem: str = DEFAULT_RUN_DIR
    parfile_name: str = PAR_FILE
    bcifile_name: str = BCI_FILE
    use_cuda: bool = USE_CUDA
    elevoff: bool = ELEVOFF
    _context: HydraulicModelContext = field(default_factory=generate_generic_context)

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        d["boundary_conditions"] = [
            BoundaryCondition(**i) for i in d["boundary_conditions"]
        ]
        return cls(**d)

    @property
    def run_dir(self) -> Path:
        return self._context.model_root / self.run_dir_stem / self.idx

    @property
    def parfile_path(self) -> Path:
        return self.run_dir / self.parfile_name

    @property
    def bcifile_path(self) -> Path:
        return self.run_dir / self.bcifile_name

    @property
    def depth_file_paths(self) -> list[Path]:
        paths = self.run_dir.glob(f"{self.idx}-????.wd")
        return sorted(paths, key=lambda x: int(x.stem[-4:]))

    @property
    def depth_grid_path(self) -> Path:
        return self.run_dir / f"{self.idx}.max"

    @property
    def wse_grid_path(self) -> Path:
        return self.run_dir / f"{self.idx}.mxe"

    @property
    def init_time_path(self) -> Path | None:
        return self.run_dir / f"{self.idx}.inittm"

    @property
    def mass_file_path(self) -> Path:
        return self.run_dir / f"{self.idx}.mass"

    @property
    def zarr_path(self) -> Path:
        return self.run_dir / f"{self.idx}_depth.zarr"

    @property
    def mass_table(self) -> pd.DataFrame:
        return pd.read_csv(self.mass_file_path, sep="\\s+")

    @property
    def sum_qfix_in(self) -> float:
        return sum(bc.value for bc in self.boundary_conditions if bc.bc_type == "QFIX")

    def is_converged(
        self,
        tolerance: float,
        method: str = "mean_depth",
        resolution: float | None = None,
    ) -> bool:
        wd_files = sorted(self.depth_file_paths)
        if len(wd_files) < 2:
            return False  # Not enough files to compare
        latest_file = wd_files[-1]
        previous_file = wd_files[-2]

        # Read the last time step from both files
        latest_data = Raster(latest_file).data
        previous_data = Raster(previous_file).data

        if method == "mean_depth":
            # Calculate average depth and average depth change
            avg_depth = latest_data[latest_data > 0].mean()
            depth_change = abs(latest_data - previous_data)
            avg_depth_change = depth_change[latest_data > 0].mean()
            if avg_depth == 0:
                return False
            relative_change = avg_depth_change / avg_depth
        elif method == "max_depth_change":
            avg_depth = latest_data[latest_data > 0].mean()
            depth_change = abs(latest_data - previous_data)
            max_depth_change = depth_change.max()
            if avg_depth == 0:
                return False
            relative_change = max_depth_change / avg_depth
        elif method == "volume_change_deprecated":
            # NOTE: LISFLOOD-FP mass file is not filled out when CUDA is enabled
            df = self.mass_table
            vol_series = df["Vol"].values
            if len(vol_series) < 2:
                return False
            dv = np.diff(vol_series)
            relative_change = abs(dv[-1]) / dv.mean()
        elif method == "volume_change":
            if resolution is None:
                raise ValueError("Resolution must be provided for volume_change method")
            v1 = np.nansum(latest_data[latest_data > 0]) * (resolution**2)
            v2 = np.nansum(previous_data[previous_data > 0]) * (resolution**2)
            delta_volume = v1 - v2
            relative_change = delta_volume / (self.sum_qfix_in * self.save_interval)

        return relative_change < tolerance

    def water_on_invalid_boundary(self, valid_polygon: Polygon) -> bool:
        """Check if water is along a boundary that is not allowed."""
        return any(self.check_water_on_invalid_boundaries(valid_polygon).values())

    def check_water_on_invalid_boundaries(
        self, valid_polygon: Polygon
    ) -> dict[str, bool]:
        """Check if water is along a boundary that is not allowed."""
        wd_files = sorted(self.depth_file_paths)
        if not wd_files:
            return {
                "north": False,
                "south": False,
                "west": False,
                "east": False,
            }  # No depth files to check
        latest_file = wd_files[-1]

        return water_on_invalid_boundary(latest_file, valid_polygon)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        del d["_context"]
        return d

    def inundation_polygon(self, transform: Affine) -> Polygon:
        """Export the inundated area as a polygon shapefile."""
        wd_files = sorted(self.depth_file_paths)
        if not wd_files:
            raise RuntimeError("No depth files available to export inundation polygon.")
        latest_file = wd_files[-1]

        depth_data = Raster(latest_file).data

        # Convert mask to polygons
        geoms = shapes(
            np.ones_like(depth_data), mask=(depth_data > 0), transform=transform
        )

        return unary_union(
            [from_geojson(json.dumps(geom)) for geom, value in geoms if value == 1]
        )

    def export_zarr(self, chunk_xy: int = 512, chunk_time: int = 1) -> None:
        """Write or append to a Zarr store with the depth time series rasters."""
        rasters = []
        target_count = self.sim_time // self.save_interval
        for p in self.depth_file_paths[: target_count + 1]:
            da = rioxarray.open_rasterio(p).squeeze("band", drop=True)
            da = da.where(da != 0)
            da = da.astype("float32")
            rasters.append(da)
        ds = xr.concat(rasters, dim="time")
        ds.attrs["_FillValue"] = np.float32(np.nan)
        ds = ds.assign_coords(time=np.arange(ds.sizes["time"]))
        ds.name = "depth"
        ds = ds.chunk(
            {
                "time": chunk_time,
                "x": chunk_xy,
                "y": chunk_xy,
            }
        )
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        encoding = {
            "depth": {
                "compressor": compressor,
                "dtype": "float32",
                # "_FillValue": np.float32(np.nan),
            }
        }

        if self.zarr_path.exists() and self.initial_state is not None:
            for k in ["add_offset", "scale_factor", "_FillValue", "missing_value"]:
                ds.attrs.pop(k, None)
            ds.to_zarr(
                self.zarr_path,
                mode="a",
                append_dim="time",
            )
        else:
            if self.zarr_path.exists():
                shutil.rmtree(self.zarr_path)
            ds.to_zarr(
                self.zarr_path,
                mode="w",
                encoding=encoding,
                zarr_version=2,
            )


@dataclass
class ModelDomain:
    idx: str
    transform: Affine
    rows: int
    cols: int
    resolution: float
    terrain: Terrain | None = None
    roughness: Roughness | None = None
    _context: HydraulicModelContext = field(default_factory=generate_generic_context)

    def __post_init__(self):
        if self.terrain is not None:
            self.terrain._context = self._context
        if self.roughness is not None:
            self.roughness._context = self._context

    @classmethod
    def from_bbox(
        cls,
        idx: str,
        bbox: BBox,
        resolution: float,
        context: HydraulicModelContext = field(
            default_factory=generate_generic_context
        ),
    ):
        bbox = snap_bbox_to_grid(bbox, resolution)

        affine = Affine(resolution, 0, bbox.xmin, 0, -resolution, bbox.ymax)
        cols = int((bbox.xmax - bbox.xmin) / resolution)
        rows = int((bbox.ymax - bbox.ymin) / resolution)
        return cls(idx, affine, rows, cols, resolution, _context=context)

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        if d["terrain"] is not None:
            d["terrain"]["_context"] = d["_context"]
            d["terrain"] = Terrain(**d["terrain"])
        if d["roughness"] is not None:
            d["roughness"]["_context"] = d["_context"]
            d["roughness"] = Roughness(**d["roughness"])
        d["transform"] = Affine(**d["transform"])
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        del d["_context"]
        d["transform"] = self.transform._asdict()
        if self.terrain is not None:
            d["terrain"] = self.terrain.to_dict()
        if self.roughness is not None:
            d["roughness"] = self.roughness.to_dict()
        return d

    @property
    def crs(self) -> CRS:
        return self._context.crs

    @property
    def model_root(self) -> Path:
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

    def load_3dep_terrain(self, units: UnitsType = "meters") -> None:
        self.terrain = Terrain.from_3dep(
            self.idx,
            self.resolution,
            self.bbox,
            self.cols,
            self.rows,
            units,
            self._context,
        )

    def load_nlcd_roughness(self) -> None:
        self.roughness = Roughness.from_nlcd(
            self.idx,
            self.resolution,
            self.bbox,
            self.cols,
            self.rows,
            self._context,
        )

    def geometry_to_bc_points(self, geometry: BaseGeometry) -> list[list[str | float]]:
        if isinstance(geometry, Point):
            return [["P", geometry.x, geometry.y]]
        elif isinstance(geometry, MultiLineString):
            geometry = linemerge(geometry)
            pts = rasterize_geometry(geometry, self.rows, self.cols, self.transform)
            return [["P", i[0], i[1]] for i in pts]
        elif isinstance(geometry, LineString):
            pts = rasterize_geometry(geometry, self.rows, self.cols, self.transform)
            return [["P", i[0], i[1]] for i in pts]
        elif isinstance(geometry, Polygon):
            return poly_to_edges(geometry, self.bbox)
        elif isinstance(geometry, MultiPolygon):
            geometry = unary_union(geometry)
            return poly_to_edges(geometry, self.bbox)
        else:
            raise RuntimeError(
                f"Boundary condition was type {geometry.geom_type}, but only Point, LineString, and Polygon are accepted"
            )

    def edges_to_bc_points(self) -> list[list[str | float]]:
        return [
            ["N", self.bbox.xmin, self.bbox.xmax],
            ["S", self.bbox.xmin, self.bbox.xmax],
            ["W", self.bbox.ymin, self.bbox.ymax],
            ["E", self.bbox.ymin, self.bbox.ymax],
        ]

    def check_files(self) -> None:
        if self.terrain is None:
            raise RuntimeError(f"No terrain available for domain {self.idx}")
        if self.roughness is None:
            raise RuntimeError(f"No roughness available for domain {self.idx}")


@dataclass
class ModelConnection:
    idx: str
    model_path: Path
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "model_path": str(self.model_path),
            "run_id": self.run_id,
        }


@dataclass
class HydraulicModel:
    metadata: HydraulicModelMetadata
    domains: dict[str, ModelDomain] = field(default_factory=dict)
    vectors: dict[str, VectorDataset] = field(default_factory=dict)
    runs: dict[str, HydraulicModelRun] = field(default_factory=dict)
    connections: dict[str, ModelConnection] = field(default_factory=dict)
    _context: HydraulicModelContext = field(default_factory=generate_generic_context)

    def __post_init__(self):
        for i in [*self.domains.values(), *self.vectors.values(), *self.runs.values()]:
            setattr(i, "_context", self._context)

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
        gpkg_path: str | Path,
        reach_id: int,
        resolution: float,
        model_root: str | Path,
        walk_us_dist_pct: float = 0.25,
        inflow_width: float = 10,
        domain_buffer: float = 100,
        crs: str = COMMON_CRS,
        metadata: dict = {},
        vector_ftype: str = "parquet",
    ):
        # Initialize model directory
        context = HydraulicModelContext(Path(model_root), CRS.from_user_input(crs))
        cls.init_model_dir(context.model_root)

        # Populate model geometry files
        vector_dir = context.model_root / DEFAULT_VECTOR_DIR
        reach_context = ReachContext(gpkg_path, reach_id)
        _vectors = reach_context.export_default_domain(
            vector_dir, walk_us_dist_pct, inflow_width, domain_buffer, vector_ftype
        )
        std_meta = DatasetMetadata("file", reach_context.gpkg_path)
        vectors = {}
        for k, v in _vectors.items():
            vectors[k] = VectorDataset(
                k, str(Path(v).relative_to(context.model_root)), std_meta, context
            )

        # Make model domain
        base_bbox = BBox(
            *unary_union([vectors["divide"].shape, vectors["us_bc_line"].shape]).bounds
        )
        base_bbox.buffer(domain_buffer)
        idx = f"hydrofabric_res_{str(resolution)}"
        domains = {idx: ModelDomain.from_bbox(idx, base_bbox, resolution, context)}

        # Make metadata
        metadata["title"] = str(reach_id)
        metadata["engineer_notes"] = (
            f"Initialized from hydrofabric. path: {gpkg_path}. Flowpath ID: {reach_id}"
        )
        meta = HydraulicModelMetadata(**metadata)

        # Make model
        return cls(meta, domains, vectors, _context=context)

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        metadata = HydraulicModelMetadata.from_dict(d["metadata"])
        if "context" in d:
            context = HydraulicModelContext(
                Path(d["context"]["model_root"]), metadata.crs
            )
            for i in d["domains"]:
                d["domains"][i]["_context"] = context
            for i in d["vectors"]:
                d["vectors"][i]["_context"] = context
            for i in d["runs"]:
                d["runs"][i]["_context"] = context
        domains = {k: ModelDomain.from_dict(v) for k, v in d["domains"].items()}
        vectors = {k: VectorDataset(**v) for k, v in d["vectors"].items()}
        runs = {k: HydraulicModelRun.from_dict(v) for k, v in d["runs"].items()}
        connections = {k: ModelConnection(**v) for k, v in d["connections"].items()}
        return cls(metadata, domains, vectors, runs, connections, _context=context)

    @classmethod
    def from_file(cls, in_path: str | Path):
        with open(in_path) as f:
            d = json.load(f)
        d["context"] = {"model_root": Path(in_path).parent}
        return cls.from_dict(d)

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "domains": {k: v.to_dict() for k, v in self.domains.items()},
            "vectors": {k: v.to_dict() for k, v in self.vectors.items()},
            "runs": {k: v.to_dict() for k, v in self.runs.items()},
            "connections": {k: v.to_dict() for k, v in self.connections.items()},
        }

    def to_file(self, out_path: str | Path) -> None:
        with open(out_path, mode="w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def save(self) -> None:
        out_path = self._context.model_root / DEFAULT_MODEL_PATH_NAME
        self.metadata.last_edited = datetime.now()
        self.to_file(out_path)

    def write_run(self, run: HydraulicModelRun) -> None:
        # Check file validity and build run directory
        domain = self.domains[run.domain]
        domain.check_files()
        run.run_dir.mkdir(exist_ok=True, parents=True)
        pts = self._process_bc_lines(run.boundary_conditions, domain)
        write_bci_file(run.bcifile_path, pts)
        write_par_file(run, domain)

    def add_run(self, run: HydraulicModelRun | dict[str, Any]) -> None:
        if isinstance(run, dict):
            run = HydraulicModelRun.from_dict(run)
        self.write_run(run)
        self.runs[run.idx] = run

    def add_connection(self, idx: str, model_path: str | Path, run_id: str) -> None:
        cnx = ModelConnection(idx, Path(model_path), run_id)
        self.connections[idx] = cnx

    def _process_bc_lines(
        self, bcs: list[BoundaryCondition], domain: ModelDomain
    ) -> list[list[str | float]]:
        all_pts = []
        for i in bcs:
            all_pts.extend(self._process_bc_line(i, domain))
        return all_pts

    def _process_bc_line(
        self, bc: BoundaryCondition, domain: ModelDomain
    ) -> list[list[str | float]]:
        # Get points from geometry
        if bc.geometry_vector == "all":
            pts = domain.edges_to_bc_points()
        else:
            line = self.vectors[bc.geometry_vector]
            pts = domain.geometry_to_bc_points(line.shape)

        if len(pts) == 0:
            return []  # TODO: think of how to raise an error here

        # Build final points with BC info
        if bc.bc_type == "QFIX":
            q_tmp = float(bc.value) / (domain.resolution * len(pts))
            pts = [[*i, "QFIX", q_tmp] for i in pts]
        elif bc.bc_type == "TRANSFER":
            pts = self._get_transfer_wses(pts, str(bc.value))
        else:
            pts = [[*i, bc.bc_type, bc.value] for i in pts]
        return pts

    def _get_transfer_wses(
        self, pts: list[list[str | float]], idx: str
    ) -> list[list[str | float]]:
        connection = self.connections[idx]
        tmp_model = HydraulicModel.from_file(connection.model_path)
        tmp_run = tmp_model.runs[connection.run_id]
        coords = [(float(i[1]), float(i[2])) for i in pts]
        try:
            vals = sample_raster(tmp_run.wse_grid_path, coords)
        except RasterioIOError:
            vals = sample_wse_from_depth_el(
                tmp_model.domains[tmp_run.domain].terrain.path,
                tmp_run.depth_file_paths[-1],
                coords,
            )
        return [[*i, "HFIX", v] for i, v in zip(pts, vals) if v is not None]
