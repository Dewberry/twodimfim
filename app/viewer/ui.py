from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from app.consts import DATA_DIR
from app.viewer.functions import (
    add_layer,
    add_vector_layer,
    generate_configuration,
    sync_layers,
)
from twodimfim.models.data_models import HydraulicModel


def model_view_section(model_path: str | Path):
    """Container for viewer dialogs for a given model."""
    m = HydraulicModel.from_file(model_path)
    with st.expander(m.metadata.title):
        d, v, r = st.tabs(["Domains", "Vectors", "Runs"])
        with d:
            domain_section(m)
        with v:
            vector_section(m)
        with r:
            run_section(m)


def domain_section(m: HydraulicModel):
    """Section for viewing domain layers."""
    for k, v in m.domains.items():
        with st.container():
            st.subheader(k)
            t_layer_name = f"{m.metadata.title}-{k}-Terrain"
            if v.terrain is not None:
                t_path = str(v.terrain.path.resolve())
                t_disable = False
            else:
                t_path = None
                t_disable = True
            r_layer_name = f"{m.metadata.title}-{k}-Roughness"
            if v.roughness is not None:
                r_path = str(v.roughness.path.resolve())
                r_disable = False
            else:
                r_path = None
                r_disable = True
            st.checkbox(
                "Terrain",
                on_change=add_layer,
                args=[t_layer_name, t_path, "terrain"],
                key=t_layer_name,
                disabled=t_disable,
            )
            st.checkbox(
                "Roughness",
                on_change=add_layer,
                args=[r_layer_name, r_path, "roughness"],
                key=r_layer_name,
                disabled=r_disable,
            )


def vector_section(m: HydraulicModel):
    """Section for viewing vector layers."""
    for k, v in m.vectors.items():
        lname = f"{m.metadata.title}-{k}"
        st.checkbox(
            lname,
            on_change=add_vector_layer,
            args=[lname, str(v.path.resolve()), "depth"],
            key=lname,
        )


def run_section(m: HydraulicModel):
    """Section for viewing run layers."""
    for k, v in m.runs.items():
        with st.container():
            st.subheader(k)
            d_layer_name = f"{m.metadata.title}-{k}-Max-Depth"
            if v.depth_grid_path.exists():
                d_path = str(v.depth_grid_path.resolve())
                d_disable = False
            else:
                d_path = None
                d_disable = True
            w_layer_name = f"{m.metadata.title}-{k}-Max-WSE"
            if v.wse_grid_path is not None:
                w_path = str(v.wse_grid_path.resolve())
                w_disable = False
            else:
                w_path = None
                w_disable = True
            st.checkbox(
                "Max Depth",
                on_change=add_layer,
                args=[d_layer_name, d_path, "depth"],
                key=d_layer_name,
                disabled=d_disable,
            )
            st.checkbox(
                "Max WSE",
                on_change=add_layer,
                args=[w_layer_name, w_path, "wse"],
                key=w_layer_name,
                disabled=w_disable,
            )


def map_tab():
    """Top-level function for the map viewer tab."""
    # Initialize state
    if "layers" not in st.session_state:
        st.session_state["layers"] = {}

    # Make map
    cfg = generate_configuration()
    with open("app/viewer/map.html", "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__CONFIG_JSON__", cfg)
    components.html(html, height=600)

    # Model selection
    models = [
        str(i.name) for i in Path(DATA_DIR).iterdir() if (i / "model.json").exists()
    ]
    sel_models = st.multiselect(
        "Select models to view", models, key="map_model_select", on_change=sync_layers
    )
    for i in sel_models:
        model_view_section(Path(DATA_DIR) / i / "model.json")
