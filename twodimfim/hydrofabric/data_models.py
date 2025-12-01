from functools import cached_property
from pathlib import Path
from typing import cast

import geopandas as gpd
from shapely import LineString, Point, Polygon, box, clip_by_rect, unary_union
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge

from twodimfim.consts import (
    COMMON_CRS,
    DIVIDE_ID_COL,
    DIVIDE_ID_PREFIX,
    DIVIDES_LAYER,
    HYDROFABRIC_BASE_URI,
    STREAM_ID_COL,
    STREAM_ID_PREFIX,
    STREAM_LAYER,
)
from twodimfim.errors import MissingReachError
from twodimfim.utils.geospatial import (
    BBox,
    perpendicular_line,
)
from twodimfim.utils.network import NetworkWalker


class ReachContext:
    def __init__(self, vpu: str, reach_id: int, us_ds_walk_dist_km: float = 10) -> None:
        self.vpu = vpu
        self.reach_id = reach_id
        self.us_ds_walk_dist_km = us_ds_walk_dist_km
        self.stream_id = STREAM_ID_PREFIX + str(reach_id)
        self.divide_id = DIVIDE_ID_PREFIX + str(reach_id)
        self.crs = COMMON_CRS
        self.gpkg_path = HYDROFABRIC_BASE_URI.format(vpu=str(vpu).rjust(2, "0"))
        self.get_divide(self.divide_id)

    def get_divide(self, divide_id: str) -> Polygon:
        try:
            query = (
                f"SELECT * FROM {DIVIDES_LAYER} WHERE {DIVIDE_ID_COL} = '{divide_id}'"
            )
            geom = (
                gpd.read_file(self.gpkg_path, sql=query)
                .to_crs(self.crs)
                .geometry.iloc[0]
            )
            return cast(Polygon, geom)
        except IndexError:
            raise MissingReachError(f"Could not find a reach with ID {divide_id}")

    def get_centerline(self, stream_id: str) -> LineString:
        query = f"SELECT * FROM {STREAM_LAYER} WHERE {STREAM_ID_COL} = '{stream_id}'"
        geom = (
            gpd.read_file(self.gpkg_path, sql=query).to_crs(self.crs).geometry.iloc[0]
        )
        geom = linemerge(geom)
        return cast(LineString, geom)

    @cached_property
    def walker(self) -> NetworkWalker:
        return NetworkWalker(self.gpkg_path)

    @cached_property
    def is_headwater(self) -> bool:
        """Determine if the reach is a headwater (no upstream reaches)."""
        us = self.walker.network[self.reach_id].us_ms
        if us == -9999:
            return True
        return False

    @cached_property
    def divide(self) -> Polygon:
        return self.get_divide(self.divide_id)

    @cached_property
    def centerline(self) -> LineString:
        return self.get_centerline(self.stream_id)

    @cached_property
    def us_ms_divide(self) -> Polygon:
        """Upstream divide polygon along mainstem."""
        us_ms_id = DIVIDE_ID_PREFIX + str(self.walker.network[self.reach_id].us_ms)
        if us_ms_id == "cat--9999":
            return None
        else:
            return self.get_divide(us_ms_id)

    @cached_property
    def us_ms_centerline(self) -> LineString:
        """Upstream divide polygon along mainstem."""
        us_ms_id = STREAM_ID_PREFIX + str(self.walker.network[self.reach_id].us_ms)
        if us_ms_id == "fp--9999":
            return None
        else:
            return self.get_centerline(us_ms_id)

    @cached_property
    def all_us_divides(self) -> Polygon:
        """All upstream divides."""
        us_divides = self.walker.walk_network_us(self.reach_id, self.us_ds_walk_dist_km)
        if len(us_divides) == 0:
            return None
        us_divides_str = "','".join([DIVIDE_ID_PREFIX + str(i) for i in us_divides])
        query = f"SELECT * FROM {DIVIDES_LAYER} WHERE {DIVIDE_ID_COL} in ('{us_divides_str}')"
        geom = (
            gpd.read_file(self.gpkg_path, sql=query)
            .to_crs(self.crs)
            .dissolve()
            .geometry.iloc[0]
        )
        return cast(Polygon, geom)

    @cached_property
    def all_ds_divides(self) -> Polygon:
        """All downstream divides."""
        ds_divides = self.walker.walk_network_ds(self.reach_id, self.us_ds_walk_dist_km)
        ds_divides_str = "','".join([DIVIDE_ID_PREFIX + str(i) for i in ds_divides])
        query = f"SELECT * FROM {DIVIDES_LAYER} WHERE {DIVIDE_ID_COL} in ('{ds_divides_str}')"
        geom = (
            gpd.read_file(self.gpkg_path, sql=query)
            .to_crs(self.crs)
            .dissolve()
            .geometry.iloc[0]
        )
        return cast(Polygon, geom)

    def export_to_dir(
        self, export_dir: str | Path, ftype: str = "parquet"
    ) -> dict[str, str]:
        export_dir = Path(export_dir)
        out_dict = {}

        for i in [
            "divide",
            "centerline",
            "us_ms_divide",
            "us_ms_centerline",
            "all_us_divides",
            "all_ds_divides",
        ]:
            geom = getattr(self, i)
            if geom is None:
                continue
            out_path = export_dir / f"{i}.{ftype}"
            self.export_shape(geom, out_path)
            out_dict[i] = str(out_path)

        return out_dict

    def export_shape(self, geometry: BaseGeometry, out_path: str | Path) -> None:
        gdf = gpd.GeoDataFrame({"ind": [1]}, geometry=[geometry], crs=self.crs)
        if str(out_path).endswith("parquet"):
            gdf.to_parquet(out_path)
        else:
            gdf.to_file(out_path)

    def make_us_bc_line(
        self,
        walk_us_dist_pct: float = 0.25,
        inflow_width: float = 10,
    ):
        if self.is_headwater:
            us_bc_pt = Point(self.centerline.coords[0])
            return perpendicular_line(self.centerline, us_bc_pt, inflow_width)
        else:
            # Walk upstream a bit for u/s boundary condition
            walk_us_dist = self.us_ms_centerline.length * walk_us_dist_pct
            us_bc_pt = self.us_ms_centerline.interpolate(1 - walk_us_dist)
            return perpendicular_line(self.us_ms_centerline, us_bc_pt, inflow_width)

    def make_transfer_line(self, bbox: BBox) -> LineString:
        geom = clip_by_rect(
            self.all_ds_divides.exterior, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        )
        return cast(LineString, geom)

    def export_default_domain(
        self,
        export_dir: str | Path,
        walk_us_dist_pct: float = 0.25,
        inflow_width: float = 10,
        buffer: float = 100,
        ftype: str = "parquet",
    ) -> dict[str, str]:
        # Export standard elements
        out_dict = self.export_to_dir(export_dir, ftype)

        # Export additional elements
        us_bc = self.make_us_bc_line(walk_us_dist_pct, inflow_width)
        out_path = Path(export_dir) / f"us_bc_line.{ftype}"
        self.export_shape(us_bc, out_path)
        out_dict["us_bc_line"] = str(out_path)

        bbox = BBox(*unary_union([self.divide, us_bc]).bounds)
        bbox.buffer(buffer)
        bbox_shape = box(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
        out_path = Path(export_dir) / f"bbox.{ftype}"
        self.export_shape(bbox_shape, out_path)
        out_dict["bbox"] = str(out_path)

        transfer_line = self.make_transfer_line(bbox)
        out_path = Path(export_dir) / f"transfer.{ftype}"
        self.export_shape(transfer_line, out_path)
        out_dict["transfer"] = str(out_path)

        return out_dict
