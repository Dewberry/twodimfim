import math
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Iterator

import numpy as np
import pyproj
import pyproj.datadir
import rasterio.features
import rasterio.transform
from affine import Affine
from pyproj import Transformer
from shapely import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

# Seems to fix an issue where transforms were all going infinite?
pyproj.__version__
pyproj.datadir.get_data_dir()


@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.xmin, self.ymin, self.xmax, self.ymax))

    @property
    def shape(self) -> Polygon:
        return box(self.xmin, self.ymin, self.xmax, self.ymax)

    def buffer(self, buffer_dist: float) -> None:
        self.xmin -= buffer_dist
        self.ymin -= buffer_dist
        self.xmax += buffer_dist
        self.ymax += buffer_dist

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


def snap_bbox_to_grid(bbox: BBox, resolution: float) -> BBox:
    return BBox(
        xmin=floor(bbox.xmin / resolution) * resolution,
        ymin=floor(bbox.ymin / resolution) * resolution,
        xmax=ceil(bbox.xmax / resolution) * resolution,
        ymax=ceil(bbox.ymax / resolution) * resolution,
    )


def transform_shape(geometry: BaseGeometry, from_crs: str, to_crs: str) -> BaseGeometry:
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return shapely_transform(transformer.transform, geometry)


def perpendicular_line(
    base_line: LineString, at_point: Point, length: float
) -> LineString:
    # Get local direction of the base line near that point
    d = 0.001 * base_line.length  # small distance to estimate tangent
    p1 = base_line.interpolate(base_line.project(at_point) - d)
    p2 = base_line.interpolate(base_line.project(at_point) + d)

    # Compute the angle of the line at that point
    angle = math.atan2(p2.y - p1.y, p2.x - p1.x)

    # Perpendicular angle
    perp_angle = angle + math.pi / 2

    # Half-length offsets
    dx = (length / 2) * math.cos(perp_angle)
    dy = (length / 2) * math.sin(perp_angle)

    # Construct the perpendicular line centered at the point
    p_left = Point(at_point.x - dx, at_point.y - dy)
    p_right = Point(at_point.x + dx, at_point.y + dy)

    return LineString([p_left, p_right])


def rasterize_geometry(
    geometry: BaseGeometry, rows: int, cols: int, transform: Affine
) -> list[tuple[float, float]]:
    # Create an empty mask
    mask = np.zeros((rows, cols), dtype=np.uint8)

    # Rasterize the line geometry into the mask
    rasterio.features.rasterize(
        [(geometry, 1)],
        out=mask,
        transform=transform,
        all_touched=True,
        dtype="uint8",
    )

    # Get row/col indices where the line intersects
    sel_rows, sel_cols = np.where(mask == 1)

    # Convert row/col indices to map (x, y) coordinates
    xs, ys = rasterio.transform.xy(transform, sel_rows, sel_cols)
    return list(zip(xs, ys))


def poly_to_edges(poly: Polygon, bbox: BBox) -> list[list[str | float]]:
    """For each line (NSEW) of the bbox perimeter, clip to poly, the report end coords."""
    # Create perimeter lines
    bottom_line = LineString([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymin)])
    top_line = LineString([(bbox.xmin, bbox.ymax), (bbox.xmax, bbox.ymax)])
    left_line = LineString([(bbox.xmin, bbox.ymin), (bbox.xmin, bbox.ymax)])
    right_line = LineString([(bbox.xmax, bbox.ymin), (bbox.xmax, bbox.ymax)])

    # Clip to polygon
    top_clipped = top_line.intersection(poly)
    bottom_clipped = bottom_line.intersection(poly)
    left_clipped = left_line.intersection(poly)
    right_clipped = right_line.intersection(poly)

    # Export
    out_list = []
    if not top_clipped.is_empty:
        out_list.append(["N", top_clipped.bounds[0], top_clipped.bounds[2]])
    if not bottom_clipped.is_empty:
        out_list.append(["S", bottom_clipped.bounds[0], bottom_clipped.bounds[2]])
    if not left_clipped.is_empty:
        out_list.append(["W", left_clipped.bounds[1], left_clipped.bounds[3]])
    if not right_clipped.is_empty:
        out_list.append(["E", right_clipped.bounds[1], right_clipped.bounds[3]])

    return out_list


def sample_raster(
    raster_path: str | Path, points: list[tuple[float, float]]
) -> list[float]:
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        transform = src.transform

    inv = ~transform
    x_ind, y_ind = zip(*[inv * (x, y) for x, y in points])
    x_ind = np.floor(x_ind).astype(int)
    y_ind = np.floor(y_ind).astype(int)

    ### DEBUG ###
    # In production, this case should not happen.
    # If this happens in productions, it indicates that the current reach abuts more than it's next d/s reach
    x_mask = (np.array(x_ind) < 0) | (np.array(x_ind) >= arr.shape[1])
    y_mask = (np.array(y_ind) < 0) | (np.array(y_ind) >= arr.shape[0])
    total_mask = x_mask | y_mask
    x_ind = x_ind[~total_mask]
    y_ind = y_ind[~total_mask]

    out = np.full(len(points), np.nan)
    out[~total_mask] = arr[y_ind, x_ind]

    return [i if not np.isnan(i) else None for i in out]


def sample_wse_from_depth_el(
    dem_path: str | Path,
    depth_path: str | Path,
    points: list[tuple[float, float]],
) -> list[float]:
    els = sample_raster(dem_path, points)
    ds = sample_raster(depth_path, points)
    vals = []
    for e, d in zip(els, ds):
        if e is None or d is None:
            vals.append(None)
        else:
            vals.append(e + d if d > 0.01 else None)
    return vals


def raster_2_array(file_path: Path | str) -> np.ndarray:
    """Load a raster file and return its data as a NumPy array."""
    with rasterio.open(file_path) as src:
        data = src.read(1)
    return data


def water_on_invalid_boundary(raster_path: str | Path, valid_polygon: Polygon) -> bool:
    """Check if water is along a boundary that is not allowed."""
    # Load raster data and rasterize the valid polygon
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        out_shape = (src.height, src.width)
    valid_mask = rasterio.features.rasterize(
        [(valid_polygon, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    ### DEBUG TEMP EXPORT POLYGON TO parquet ###
    # import geopandas as gpd

    # gpd.GeoDataFrame(geometry=[valid_polygon], crs="EPSG:5070").to_parquet(
    #     "debug_valid_polygon.parquet"
    # )
    # #############################################

    # ### DEBUG TEMP EXPORT MASK TO RASTER TIF ###
    # with rasterio.open(
    #     "debug_valid_mask.tif",
    #     "w",
    #     driver="GTiff",
    #     height=valid_mask.shape[0],
    #     width=valid_mask.shape[1],
    #     count=1,
    #     dtype=valid_mask.dtype,
    #     crs="EPSG:5070",
    #     transform=transform,
    # ) as dst:
    #     dst.write(valid_mask, 1)
    #############################################

    # Identify water cells (assuming water depth > 0.01 indicates water presence)
    water_mask = data > 0.01

    # Create a boundary mask
    boundary_mask = np.zeros_like(data, dtype=bool)
    boundary_mask[0, :] = True  # Top row
    boundary_mask[-1, :] = True  # Bottom row
    boundary_mask[:, 0] = True  # Left column
    boundary_mask[:, -1] = True  # Right column

    # Check for water on invalid boundaries
    invalid_boundary_mask = boundary_mask & water_mask & (valid_mask == 0)

    # Split out into N, S, E, W boundaries
    return {
        "north": np.any(invalid_boundary_mask[0, :]),
        "south": np.any(invalid_boundary_mask[-1, :]),
        "west": np.any(invalid_boundary_mask[:, 0]),
        "east": np.any(invalid_boundary_mask[:, -1]),
    }
