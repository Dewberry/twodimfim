from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import requests
from owslib.wms import WebMapService
from rasterio import mask
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from shapely import Polygon, box

from twodimfim.consts import (
    COMMON_CRS,
    FT_TO_METERS,
    MANNINGS_LC_LOOKUP,
    NLCD_WMS_URL,
    USGS_3DEP_URL,
)
from twodimfim.utils.geospatial import BBox, transform_shape

### DATA MODELS ###

SourceType = Literal["file", "url"]

# TODO: Implement this
# @dataclass
# class DatasetMetadata:
#     """Metadata tracking dataset provenance and processing."""

#     source_type: SourceType
#     source_location: str
#     created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
#     transformations: list[str] = field(default_factory=list)
#     extra: dict[str, Any] = field(default_factory=dict)


def get_nlcd_mannings(
    out_path: str | Path, bbox: BBox, cols: int, rows: int, crs: str = COMMON_CRS
):
    # Get download URL
    url = (
        WebMapService(NLCD_WMS_URL)
        .getmap(
            layers=["NLCD_2021_Land_Cover_L48"],
            srs=crs,
            bbox=bbox,
            size=(cols, rows),
            format="image/geotiff",
        )
        .geturl()
    )

    # Download data
    r = requests.get(url)
    with MemoryFile(r.content) as memfile:
        with memfile.open() as src:
            out_meta = src.meta
            nlcd, out_transform = mask.mask(
                src, [bbox.shape], all_touched=True, crop=True
            )

    # Convert LC type to mannings
    mannings_array = np.full_like(nlcd, np.nan, dtype=float)
    for code, n_value in MANNINGS_LC_LOOKUP.items():
        mannings_array = np.where(nlcd == code, n_value, mannings_array)
    out_meta["dtype"] = "float32"
    out_meta["driver"] = "AAIGrid"

    # Write data
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(mannings_array)


def get_usgs_dem(
    out_path: str | Path, bbox: BBox, cols: int, rows: int, crs: str = COMMON_CRS
):
    # Open remote dataset
    with rasterio.open(USGS_3DEP_URL) as src:
        src_bbox = transform_shape(bbox.shape, crs, src.crs)
        values, out_transform = mask.mask(src, [src_bbox], all_touched=True, crop=True)

        # Define target metadata
        transform = from_bounds(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, cols, rows)
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": crs,
                "transform": transform,
                "width": cols,
                "height": rows,
                "driver": "AAIGrid",
            }
        )
        values *= FT_TO_METERS

        # Reproject and write data
        with rasterio.open(out_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.nearest,
            )
