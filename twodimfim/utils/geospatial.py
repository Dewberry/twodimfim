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


def rasterize_line(
    line: LineString, rows: int, cols: int, transform: Affine
) -> list[tuple[float, float]]:
    # Create an empty mask
    mask = np.zeros((rows, cols), dtype=np.uint8)

    # Rasterize the line geometry into the mask
    rasterio.features.rasterize(
        [(line, 1)],
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

    return list(arr[y_ind, x_ind])
