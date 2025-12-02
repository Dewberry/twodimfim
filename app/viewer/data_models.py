import json
from functools import cached_property
from pathlib import Path
from urllib.parse import quote

import geopandas as gpd
import requests
from pyproj import Transformer

from app.consts import TITILER_URL_INTERNAL, TITILER_PORT


class OverlayLayer:
    """Class representing an overlay layer on the map."""

    idx: str
    url: str
    type_: str

    def __init__(self, idx: str, url: str, type_: str = "terrain"):
        """Initialize OverlayLayer."""
        self.idx = idx
        self.url = url
        self.type_ = type_

    @property
    def colormap(self):
        """Return colormap name based on type."""
        if self.type_ == "terrain":
            return "schwarzwald"
        elif self.type_ == "roughness":
            return "binary"
        elif self.type_ == "depth":
            return "cool"
        elif self.type_ == "wse":
            return "gnuplot"

    @property
    def opacity(self):
        """Return opacity based on type."""
        if self.type_ in ["terrain", "roughness"]:
            return 1
        elif self.type_ in ["depth", "wse"]:
            return 0.85

    @property
    def statistics(self):
        """Fetch min/max statistics from TiTiler."""
        meta_url = f"{TITILER_URL_INTERNAL}/cog/statistics?url={quote(self.url, safe='')}"
        r = requests.get(meta_url).json()

        # Set reasonable defaults if min/max are not available
        if self.type_ == "roughness":
            vmin = 0
            vmax = 1
        elif self.type_ == "terrain":
            vmin = 0
            vmax = 1000
        elif self.type_ == "depth":
            vmin = 0
            vmax = 50
        elif self.type_ == "wse":
            vmin = 0
            vmax = 1000
        else:
            vmin = 0
            vmax = 1

        # Try to get actual min/max from metadata
        try:
            vmin = r["b1"]["min"]
            vmax = r["b1"]["max"]
        except KeyError:
            pass
        return {"min": vmin, "max": vmax}

    @property
    def bbox_4326(self):
        """Fetch bounding box from TiTiler."""
        meta_url = f"{TITILER_URL_INTERNAL}/cog/info?url={quote(self.url, safe='')}"
        r = requests.get(meta_url).json()

        transformer = Transformer.from_crs(
            r.get("crs", "EPSG:5070"), "EPSG:4326", always_xy=True
        )
        minx, miny, maxx, maxy = r.get("bounds", [-1, -1, 1, 1])
        llx, lly = transformer.transform(minx, miny)
        urx, ury = transformer.transform(maxx, maxy)
        return llx, lly, urx, ury

    def to_overlay_dict(self):
        """Generate overlay dictionary for map configuration."""
        stats = self.statistics
        base = f"{TITILER_URL_EXTERNAL}/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
        url = f"{base}?url={quote(self.url, safe='')}&colormap_name={self.colormap}&rescale={stats['min']},{stats['max']}"
        return {
            "name": self.idx,
            "url": url,
            "attribution": "NA",
            "opacity": self.opacity,
            "visible": True,
        }


class VectorLayer:
    """Class representing a vector layer on the map."""

    idx: str
    url: str
    type_: str

    def __init__(self, idx: str, url: str, type_: str = "depth"):
        """Initialize VectorLayer."""
        self.idx = idx
        self.url = url
        self.type_ = type_

    @property
    def bbox_4326(self):
        """Fetch bounding box from GeoDataFrame."""
        bounds = self.gdf.total_bounds  # minx, miny, maxx, maxy
        return bounds[0], bounds[1], bounds[2], bounds[3]

    @cached_property
    def gdf(self):
        """Load GeoDataFrame from file."""
        if self.url.endswith(".parquet"):
            return gpd.read_parquet(self.url).to_crs(epsg=4326)
        else:
            return gpd.read_file(self.url).to_crs(epsg=4326)

    def to_geojson_dict(self):
        """Generate GeoJSON dictionary for map configuration."""
        gjson = self.gdf.to_json()
        return {
            "name": self.idx,
            "data": gjson,
            "style": {
                "color": "blue",
                "weight": 2,
                "opacity": 0.6,
                "fillOpacity": 0.2,
            },
            "visible": True,
        }
