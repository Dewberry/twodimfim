from functools import cached_property
from pathlib import Path
from typing import Literal, cast

import geopandas as gpd
from shapely import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
    unary_union,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge

from twodimfim.consts import (
    COMMON_CRS,
    NEW_HF_NETWORK_FORMAT,
    OLD_HF_NETWORK_FORMAT,
)
from twodimfim.errors import DownstreamModelMisalignmentError, MissingReachError
from twodimfim.hydrofabric.utils import get_hf_type
from twodimfim.utils.geospatial import (
    BBox,
    perpendicular_line,
)
from twodimfim.utils.network import NetworkWalker


class HF_ACCESS:
    """Convenient querying of hydrofabric data."""

    def __init__(self, gpkg_path: str | Path):
        self.gpkg_path = gpkg_path
        self.type: Literal["old", "new"] = get_hf_type(gpkg_path)
        if self.type == "old":
            self.fmt = OLD_HF_NETWORK_FORMAT
        else:
            self.fmt = NEW_HF_NETWORK_FORMAT

    @property
    def divides_layer(self) -> str:
        return self.fmt["divides_layer"]

    @property
    def divide_id_col(self) -> str:
        return self.fmt["divide_id_col"]

    @property
    def stream_layer(self) -> str:
        return self.fmt["stream_layer"]

    @property
    def stream_id_col(self) -> str:
        return self.fmt["stream_id_col"]

    @property
    def divide_id_prefix(self) -> str:
        return self.fmt["divide_id_prefix"]

    @property
    def stream_id_prefix(self) -> str:
        return self.fmt["stream_id_prefix"]

    def get_divide(self, divide_id: int, to_crs: str | None = None) -> Polygon:
        if self.divide_id_prefix == "":
            divide_id = f"{divide_id}"
        else:
            divide_id = f"'{self.divide_id_prefix}{divide_id}'"
        try:
            query = f"SELECT * FROM {self.divides_layer} WHERE {self.divide_id_col} = {divide_id}"
            gdf = gpd.read_file(self.gpkg_path, sql=query)
            if to_crs is not None:
                gdf = gdf.to_crs(to_crs)
            geom = gdf.geometry.iloc[0]
            return cast(Polygon, geom)
        except IndexError:
            raise MissingReachError(f"Could not find a reach with ID {divide_id}")

    def get_centerline(self, stream_id: int, to_crs: str | None = None) -> LineString:
        if self.stream_id_prefix == "":
            stream_id_str = f"{stream_id}"
        else:
            stream_id_str = f"'{self.stream_id_prefix}{stream_id}'"
        query = f"SELECT * FROM {self.stream_layer} WHERE {self.stream_id_col} = {stream_id_str}"
        gdf = gpd.read_file(self.gpkg_path, sql=query)
        if to_crs is not None:
            gdf = gdf.to_crs(to_crs)
        geom = gdf.geometry.iloc[0]
        geom = linemerge(geom)
        return cast(LineString, geom)

    def get_flowpaths(
        self, flowpath_ids: list[str], to_crs: str | None = None
    ) -> gpd.GeoDataFrame:
        if self.stream_id_prefix == "":
            flowpaths_str = "(" + ", ".join([str(i) for i in flowpath_ids]) + ")"
        else:
            flowpaths_str = (
                "('"
                + "', '".join([self.stream_id_prefix + str(i) for i in flowpath_ids])
                + "')"
            )

        query = f"SELECT * FROM {self.stream_layer} WHERE {self.stream_id_col} in {flowpaths_str}"
        gdf = gpd.read_file(self.gpkg_path, sql=query)
        if to_crs is not None:
            gdf = gdf.to_crs(to_crs)
        return gdf

    def get_divides(
        self, divide_ids: list[str], to_crs: str | None = None
    ) -> gpd.GeoDataFrame:
        if self.divide_id_prefix == "":
            divides_str = "(" + ", ".join([str(i) for i in divide_ids]) + ")"
        else:
            divides_str = (
                "('"
                + "', '".join([self.divide_id_prefix + str(i) for i in divide_ids])
                + "')"
            )

        query = f"SELECT * FROM {self.divides_layer} WHERE {self.divide_id_col} in {divides_str}"
        gdf = gpd.read_file(self.gpkg_path, sql=query)
        if to_crs is not None:
            gdf = gdf.to_crs(to_crs)
        return gdf


class ReachContext:
    def __init__(
        self, gpkg_path: str | Path, reach_id: int, us_ds_walk_dist_km: float = 10
    ) -> None:
        self.gpkg_path = str(gpkg_path)
        self.reach_id = reach_id
        self.us_ds_walk_dist_km = us_ds_walk_dist_km
        self.stream_id = reach_id
        self.divide_id = reach_id
        self.crs = COMMON_CRS
        self.hf_access = HF_ACCESS(self.gpkg_path)
        self.hf_access.get_divide(self.divide_id)

    @cached_property
    def walker(self) -> NetworkWalker:
        return NetworkWalker.from_gpkg(self.gpkg_path)

    @cached_property
    def is_headwater(self) -> bool:
        """Determine if the reach is a headwater (no upstream reaches)."""
        us = self.walker.network[self.reach_id].us_ms
        if us == -9999:
            return True
        return False

    @cached_property
    def divide(self) -> Polygon:
        return self.hf_access.get_divide(self.divide_id, to_crs=self.crs)

    @cached_property
    def centerline(self) -> LineString:
        return self.hf_access.get_centerline(self.stream_id, to_crs=self.crs)

    @cached_property
    def us_ms_divide(self) -> Polygon:
        """Upstream divide polygon along mainstem."""
        us_ms_id = self.walker.network[self.reach_id].us_ms
        if us_ms_id == -9999:
            return None
        else:
            return self.hf_access.get_divide(us_ms_id, to_crs=self.crs)

    @cached_property
    def us_ms_centerline(self) -> LineString:
        """Upstream divide polygon along mainstem."""
        us_ms_id = self.walker.network[self.reach_id].us_ms
        if us_ms_id == -9999:
            return None
        else:
            return self.hf_access.get_centerline(us_ms_id, to_crs=self.crs)

    @cached_property
    def all_us_divides(self) -> Polygon:
        """All upstream divides."""
        us_divides = self.walker.walk_network_us(self.reach_id, self.us_ds_walk_dist_km)
        if len(us_divides) == 0:
            return None

        geom = (
            self.hf_access.get_divides(us_divides, to_crs=self.crs)
            .dissolve()
            .geometry.iloc[0]
        )
        return cast(Polygon, geom)

    @cached_property
    def all_ds_divides(self) -> Polygon:
        """All downstream divides."""
        ds_divides = self.walker.walk_network_ds(self.reach_id, self.us_ds_walk_dist_km)
        if len(ds_divides) == 0:
            return None
        geom = (
            self.hf_access.get_divides(ds_divides, to_crs=self.crs)
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

    def make_transfer_line(self) -> LineString:
        geom = LineString(self.all_ds_divides.exterior.coords)
        if geom.is_empty:
            raise DownstreamModelMisalignmentError(
                "The downstream model's divide does not intersect with the upstream model's bbox."
            )
        return cast(LineString, geom)

    def make_transfer_line_offset(self, resolution: float) -> LineString:
        debuff = self.all_ds_divides.buffer(-resolution)
        if isinstance(debuff, MultiPolygon):
            debuff = max(debuff.geoms, key=lambda g: g.area)
        debuff = debuff.exterior

        geom = LineString(debuff.coords)
        if geom.is_empty:
            raise DownstreamModelMisalignmentError(
                "The downstream model's divide does not intersect with the upstream model's bbox."
            )
        return cast(LineString, geom)

    def export_default_domain(
        self,
        export_dir: str | Path,
        walk_us_dist_pct: float = 0.25,
        inflow_width: float = 10,
        ftype: str = "parquet",
        resolution: float = 10,
    ) -> dict[str, str]:
        # Export standard elements
        out_dict = self.export_to_dir(export_dir, ftype)

        # Export additional elements
        us_bc = self.make_us_bc_line(walk_us_dist_pct, inflow_width)
        out_path = Path(export_dir) / f"us_bc_line.{ftype}"
        self.export_shape(us_bc, out_path)
        out_dict["us_bc_line"] = str(out_path)

        return out_dict
