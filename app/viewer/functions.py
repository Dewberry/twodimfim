import json
from pathlib import Path

import rasterio
import streamlit as st
from matplotlib.pyplot import get
from rasterio.enums import Resampling

from app.consts import BASEMAPS
from app.viewer.data_models import OverlayLayer, VectorLayer


def get_map_center():
    """Return the center coordinates for the map."""
    valid_layers = [i for i in st.session_state["layers"].values() if i is not None]
    if len(valid_layers) == 0:
        center = [39.8, -98.5]
    else:
        bounds = [9999, 9999, -9999, -9999]
        for i in valid_layers:
            llx, lly, urx, ury = i.bbox_4326

            bounds[0] = min(bounds[0], llx)
            bounds[1] = min(bounds[1], lly)
            bounds[2] = max(bounds[2], urx)
            bounds[3] = max(bounds[3], ury)

        center = [
            bounds[1] + (bounds[3] - bounds[1]) / 2,
            bounds[0] + (bounds[2] - bounds[0]) / 2,
        ]
    return center


def generate_configuration():
    """Create a config json to be used by the map viewer."""
    cfg = {
        "center": get_map_center(),
        "zoom": 12,
        "basemaps": BASEMAPS,
        "overlays": [
            i.to_overlay_dict()
            for i in st.session_state["layers"].values()
            if i is not None and type(i).__name__ == "OverlayLayer"
        ],
        "vectors": [
            i.to_geojson_dict()
            for i in st.session_state["layers"].values()
            if i is not None and type(i).__name__ == "VectorLayer"
        ],
    }
    return json.dumps(cfg)


def ensure_cog(url: str) -> str:
    """Ensure that a raster file is a COG. If not, convert and return new path."""
    in_url = url
    out_url = in_url + ".tif"
    if in_url.split(".")[-1] == "tif":
        return in_url
    if Path(out_url).exists():
        return out_url

    with rasterio.open(in_url) as r:
        data = r.read(1)
        profile = r.profile.copy()
        nd = r.nodata

    if profile.get("crs") is None:
        from rasterio.crs import CRS

        profile["crs"] = CRS.from_string("EPSG:5070")

    if in_url.endswith(".max"):
        data[data == 0] = nd

    profile.update(
        driver="COG",
        compress="DEFLATE",
        nodata=nd,
        add_mask=True,
        resampling=Resampling.bilinear,
    )

    with rasterio.open(out_url, "w", **profile) as w:
        w.write(data, 1)

    return out_url


def add_layer(idx, url, type_):
    """Add an overlay layer to the map."""
    if st.session_state[idx]:
        url = ensure_cog(url)
        st.session_state["layers"][idx] = OverlayLayer(idx, url, type_)
    else:
        st.session_state["layers"][idx] = None


def add_vector_layer(idx, url, type_):
    """Add a vector layer to the map."""
    if st.session_state[idx]:
        st.session_state["layers"][idx] = VectorLayer(idx, url, type_)
    else:
        st.session_state["layers"][idx] = None


def sync_layers():
    """Remove models from layers when they are not in multiselect."""
    l = list(st.session_state["layers"])
    for i in l:
        remove = True
        for j in st.session_state["map_model_select"]:
            if i.startswith(j):
                remove = False
        if remove:
            del st.session_state["layers"][i]
