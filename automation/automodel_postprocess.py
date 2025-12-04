### POSTPROCESSING ###

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from rasterio.features import rasterize
from rasterio.shutil import copy as rio_copy
from shapely.geometry import MultiPolygon, Polygon

# --------- CONFIG ---------
DATA_DIR = Path("/data")
OUTPUT_DIR = DATA_DIR / "postprocessed"
OUTPUT_DIR.mkdir(exist_ok=True)
VRT_PATH = OUTPUT_DIR / "Q100_depths.vrt"
GPKG_PATH = OUTPUT_DIR / "combined.gpkg"
# ---------------------------


def rasterize_mask(gdf, template_ds):
    """Rasterize a single-feature GeoDataFrame to match a raster profile."""
    shapes = [(geom, 1) for geom in gdf.geometry]
    return rasterize(
        shapes,
        out_shape=(template_ds.height, template_ds.width),
        transform=template_ds.transform,
        fill=0,
        dtype="uint8",
    )


def process_depth_raster(model_dir: Path):
    depth_path = model_dir / "runs" / "Q100" / "Q100.max"
    try:
        us_gdf = gpd.read_parquet(model_dir / "vectors" / "us_ms_divide.parquet")
    except FileNotFoundError:
        us_gdf = None

    # ds_gdf = gpd.read_parquet(model_dir / "vectors" / "all_ds_divides.parquet")

    print(f"Processing {depth_path}")

    with rasterio.open(depth_path) as src:
        depth = src.read(1)
        profile = src.profile.copy()
        nodata = profile.get("nodata", -9999)

        # Rasterize masks
        # ds_mask = rasterize_mask(ds_gdf, src)
        ds_mask = np.zeros_like(depth, dtype=np.uint8)
        if us_gdf is not None:
            us_mask = rasterize_mask(us_gdf, src)
        else:
            us_mask = np.zeros_like(depth, dtype=np.uint8)

        combined_mask = (us_mask == 1) | (ds_mask == 1) | (depth == 0)

        depth_masked = depth.copy()
        depth_masked[combined_mask] = nodata

        out_path = OUTPUT_DIR / f"{model_dir.name}.tif"
        tmp_path = out_path.with_suffix(".tmp.tif")

        profile.update(
            driver="GTiff",
            compress="LZW",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            crs="EPSG:5070",  # <-- ADD HERE
        )

        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(depth_masked, 1)

        rio_copy(
            tmp_path,
            out_path,
            copy_src_overviews=True,
            driver="COG",
            compress="LZW",
            overview_resampling="average",
        )

        os.remove(tmp_path)
        return out_path


def force_multipolygon(gdf):
    """Ensure all geometries are MultiPolygons (avoid QGIS crashes)."""
    gdf["geometry"] = gdf.geometry.apply(
        lambda g: MultiPolygon([g]) if isinstance(g, Polygon) else g
    )
    return gdf


def process_geometries(model_dir: Path):
    """Return GeoDataFrames for reach_polygons and inflow_bc for a single model."""
    model_id = model_dir.name
    vectors_dir = model_dir / "vectors"

    # 1. Reach bbox clipped by US + DS divides
    bbox = gpd.read_parquet(vectors_dir / "bbox.parquet")
    bbox["model_id"] = model_id

    try:
        us_poly = gpd.read_parquet(vectors_dir / "us_ms_divide.parquet")
    except FileNotFoundError:
        us_poly = None

    # ds_poly = gpd.read_parquet(vectors_dir / "all_ds_divides.parquet")

    # Clip bbox by US/DS divides
    clipped = bbox.copy()
    if us_poly is not None:
        clipped = gpd.overlay(clipped, us_poly, how="difference")
    # clipped = gpd.overlay(clipped, ds_poly, how="difference")
    clipped["model_id"] = model_id
    clipped = force_multipolygon(clipped)

    # 2. Upstream inflow boundary conditions
    inflow_path = vectors_dir / "us_bc_line.parquet"
    if not inflow_path.exists():
        inflow_path = vectors_dir / "transfer.parquet"

    inflow = gpd.read_parquet(inflow_path)
    inflow["model_id"] = model_id

    return clipped, inflow


def post_process():
    # 1. Find all Q100.max rasters in model folders
    depth_rasters = list(DATA_DIR.glob("69*/runs/Q100/Q100.max"))
    output_files = []

    # Collect all GeoDataFrames
    all_reach_polygons = []
    all_inflow_bc = []

    for depth_path in depth_rasters:
        model_dir = depth_path.parent.parent.parent

        # Process rasters
        out = process_depth_raster(model_dir)
        if out:
            output_files.append(str(out))

        # Process geometries
        clipped, inflow = process_geometries(model_dir)
        all_reach_polygons.append(clipped)
        all_inflow_bc.append(inflow)

    # 2. Build VRT
    if output_files:
        print("Building VRT...")
        gdal.BuildVRT(str(VRT_PATH), output_files)
        print(f"VRT created at {VRT_PATH}")
    else:
        print("No output rasters created.")

    # 3. Combine and write GeoPackage
    if all_reach_polygons or all_inflow_bc:
        # Remove old GPKG
        if GPKG_PATH.exists():
            GPKG_PATH.unlink()

        reach_gdf = gpd.GeoDataFrame(pd.concat(all_reach_polygons, ignore_index=True))
        inflow_gdf = gpd.GeoDataFrame(pd.concat(all_inflow_bc, ignore_index=True))

        reach_gdf.to_file(GPKG_PATH, layer="reach_polygons", driver="GPKG")
        inflow_gdf.to_file(GPKG_PATH, layer="inflow_bc", driver="GPKG")
        print(f"GeoPackage created at {GPKG_PATH}")
    else:
        print("No geometries to write to GeoPackage.")


if __name__ == "__main__":
    post_process()
