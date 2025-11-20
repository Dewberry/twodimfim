import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import requests
from owslib.util import Authentication
from owslib.wms import WebMapService
from rasterio import mask
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

from twodimfim.consts import (
    COMMON_CRS,
    FT_TO_METERS,
    MANNINGS_LC_LOOKUP,
    NLCD_WMS_URL,
    USGS_3DEP_URL,
    SourceType,
)
from twodimfim.models.data_models import UnitsType
from twodimfim.utils.geospatial import BBox, transform_shape

### DATA MODELS ###


@dataclass
class DatasetMetadata:
    """Metadata tracking dataset provenance."""

    source_type: SourceType
    source_location: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    transformations: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def get_nlcd_mannings(
    out_path: str | Path, bbox: BBox, cols: int, rows: int, crs: str = COMMON_CRS
) -> DatasetMetadata:
    # Get download URL
    auth = Authentication(verify=False)
    url = (
        WebMapService(NLCD_WMS_URL, auth=auth)
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
    r = requests.get(url, verify=False)
    with MemoryFile(r.content) as memfile:
        with memfile.open() as src:
            out_meta = src.meta
            t = [f"reproject: {':'.join(src.crs.to_authority())} -> {crs}"]
            nlcd, out_transform = mask.mask(
                src, [bbox.shape], all_touched=True, crop=True
            )

    # Convert LC type to mannings
    mannings_array = np.full_like(nlcd, np.nan, dtype=float)
    for code, n_value in MANNINGS_LC_LOOKUP.items():
        mannings_array = np.where(nlcd == code, n_value, mannings_array)
    t.append(f"mannings_lookup: {json.dumps(MANNINGS_LC_LOOKUP)}")
    out_meta["dtype"] = "float32"
    out_meta["driver"] = "AAIGrid"

    # Write data
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(mannings_array)

    # Generate and return metadata
    return DatasetMetadata("url", NLCD_WMS_URL, transformations=t)


def get_usgs_dem(
    out_path: str | Path,
    bbox: BBox,
    cols: int,
    rows: int,
    crs: str = COMMON_CRS,
    units: UnitsType = "meters",
) -> DatasetMetadata:
    # Open remote dataset
    with rasterio.open(USGS_3DEP_URL) as src:
        src_crs = src.crs
        src_bbox = transform_shape(bbox.shape, crs, src_crs)
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
        t = [f"reproject: {':'.join(src_crs.to_authority())} -> {crs}"]

        if units == "meters":
            values *= FT_TO_METERS
            t.append(f"convert feet to meters: {FT_TO_METERS}")

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

    # Generate and return metadata
    return DatasetMetadata("url", USGS_3DEP_URL, transformations=t)
