import os
import time
from pathlib import Path

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from twodimfim.models.data_models import HydraulicModel, HydraulicModelRun

DATA_DIR = "/data"
MODEL_HOST = os.getenv("MODEL_HOST", "lisflood-model")
MODEL_PORT = os.getenv("MODEL_PORT", "5000")
BASE_URL = f"http://{MODEL_HOST}:{MODEL_PORT}"


def make_new_model(vpu, reach_id, resolution, inflow_width):
    model_root = Path(DATA_DIR) / str(reach_id)
    st.session_state["model"] = HydraulicModel.from_hydrofabric(
        vpu, reach_id, resolution, model_root, inflow_width=inflow_width
    )


@st.dialog("Create a new model")
def new_model():
    vpu = st.number_input(label="Vector Processing Unit (VPU)", value=1)
    reach_id = st.number_input(label="Reach ID", step=1)
    resolution = st.number_input(label="Model Resolution (m)", value=10)
    inflow_width = st.number_input(label="Inflow Width (m)", value=100)
    if st.button("Create Model"):
        make_new_model(vpu, reach_id, resolution, inflow_width)
        st.rerun()


@st.dialog("Load a model")
def load_model():
    all_models = [
        str(i.name)
        for i in Path(DATA_DIR).iterdir()
        if i.is_dir() and (i / "model.json").exists()
    ]
    path = st.selectbox("Saved models", options=all_models)
    if st.button("Open model"):
        st.session_state["model"] = HydraulicModel.from_file(
            Path(DATA_DIR) / path / "model.json"
        )
        st.rerun()


@st.dialog("Create a new run")
def new_run():
    idx = st.text_input("Run ID")
    types_ = ["Unsteady", "Quasi-steady"]
    type_ = st.selectbox("Run Type", options=types_)
    domains = st.session_state["model"].domains.keys()
    domain = st.selectbox("Model Domain", options=domains)
    bc_default = pd.DataFrame(
        {
            "Geometry": [],
            "Type": [],
            "Value": [],
        }
    )
    bc_default["Geometry"] = bc_default["Geometry"].astype(str)
    bc_default["Type"] = bc_default["Type"].astype(str)
    ccfg = {
        "Geometry": st.column_config.TextColumn(),
        "Type": st.column_config.SelectboxColumn(options=["QFIX", "HFIX", "FREE"]),
        "Value": st.column_config.NumberColumn(),
    }
    st.text("Boundary Conditions")
    bcs = st.data_editor(bc_default, num_rows="dynamic", column_config=ccfg)

    with st.expander("Additional parameters"):
        save_interval = st.number_input("Save Interval", value=900)
        mass_interval = st.number_input("Mass Interval", value=15)
        sim_time = st.number_input("Simulation Time", value=3600)
        steady_state_tolerance = st.number_input("Steady State Tolerance", value=0.95)
        initial_tstep = st.number_input("Initial Timestep", value=0.5)

    if st.button("Save Run"):
        bcs_refactor = [
            {"geometry_vector": i.Geometry, "type_": i.Type, "value": i.Value}
            for i in bcs.itertuples()
        ]
        tmp_run = HydraulicModelRun(
            idx=idx,
            type_=type_.lower(),
            domain=domain,
            boundary_conditions=bcs_refactor,
            save_interval=save_interval,
            mass_interval=mass_interval,
            sim_time=sim_time,
            steady_state_tolerance=steady_state_tolerance,
            initial_tstep=initial_tstep
        )
        st.session_state["model"].add_run(tmp_run)
        st.rerun()


def save_model():
    if st.session_state["model"] is not None:
        st.session_state["model"].save()


def model_control():
    with st.container(border=True, width=400) as c:
        c1, c2, c3 = st.columns(3, gap="small")
        c1.button("New Model", width=125, on_click=new_model)
        c2.button("Load Model", width=125, on_click=load_model)
        c3.button("Save Model", width=125, on_click=save_model)


def domain_editor():
    domains = st.session_state["model"].domains.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        domain_ = st.selectbox("Select a domain", domains, index=0)
        st.button("New domain")
        st.button("Delete domain")
    domain = st.session_state["model"].domains[domain_]
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            text = "\n\n".join(
                [
                    "**{}**: {}".format(i, getattr(domain.bbox, i))
                    for i in ["xmin", "xmax", "ymin", "ymax"]
                ]
            )
            st.markdown(text)
        with c2:
            text = f"**Columns**: {domain.cols}\n\n**Rows**: {domain.rows}\n\n**Resolution**: {domain.resolution}\n\n"
            st.markdown(text)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        if st.button("Download USGS Terrain"):
            st.toast("Downloading USGS terrain", icon=":material/cloud_download:")
            domain.load_3dep_terrain()
            st.toast("Downloaded USGS terrain", icon=":material/download_done:")
        if st.button("Download NLCD Roughness"):
            st.toast("Downloading NLCD Roughness", icon=":material/cloud_download:")
            domain.load_nlcd_roughness()
            st.toast("Downloaded NLCD Roughness", icon=":material/download_done:")


def run_editor():
    runs = st.session_state["model"].runs.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        run_ = st.selectbox("Select a run", runs, index=0, key="redit")
        st.button("New run", on_click=new_run)
        st.button("Delete run")


def run_executor():
    runs = st.session_state["model"].runs.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        run_ = st.selectbox("Select a run", runs, index=0, key="rexec")
        if st.button("Execute run"):
            run_path = st.session_state["model"].runs[run_].par_path
            with st.spinner(f"Running Lisflood model at {run_path}"):
                run_model(run_path)


def run_model(run_path):
    time.sleep(2)
    try:
        response = requests.post(f"{BASE_URL}/run_model", json={"model_dir": run_path})
        if response.status_code == 200:
            result = response.json()
        else:
            st.error(f"Model API returned {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Lisflood API: {e}")


def model_editor():
    t1, t2, t3 = st.tabs(["Domain Editor", "Run Editor", "Run Executor"])
    with t1:
        domain_editor()
    with t2:
        run_editor()
    with t3:
        run_executor()


def editor_tab():
    model_control()
    if st.session_state["model"] is not None:
        model_editor()


def map_tab():
    if st.session_state["model"] is not None:
        layers = []
        for k in [
            "all_us_divides",
            "all_ds_divides",
            "divide",
            "us_bc_line",
            "centerline",
        ]:
            gdf = st.session_state["model"].vectors[k].gdf
            gdf["feature"] = k
            gdf = gdf.to_crs("EPSG:4326")

            if k == "all_us_divides" or k == "all_ds_divides":
                lc = [0, 0, 0]
                fc = [0, 0, 0, 0.25]
                lw = 12
            elif k == "divide":
                lc = [255, 0, 0]
                fc = [0, 0, 0, 0.0]
                lw = 30
            elif k == "centerline":
                lc = [0, 0, 255]
                fc = [0, 0, 0, 0.0]
                lw = 30
            elif k == "us_bc_line":
                lc = [255, 153, 0]
                fc = [0, 0, 0, 0.0]
                lw = 30
            else:
                continue

            geojson = gdf.__geo_interface__
            layer = pdk.Layer(
                "GeoJsonLayer",
                id=k,
                data=geojson,
                stroked=True,
                filled=True,
                get_line_color=lc,
                get_fill_color=fc,
                get_line_width=lw,
                pickable=True,
            )

            layers.append(layer)

        # Build the deck
        bounds = (
            st.session_state["model"]
            .vectors["centerline"]
            .gdf.to_crs("EPSG:4326")
            .total_bounds
        )
        lon_center = (bounds[0] + bounds[2]) / 2
        lat_center = (bounds[1] + bounds[3]) / 2
        view_state = pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=12)
        tooltip = {"html": "<b>{feature}</b>", "style": {"color": "white"}}
        r = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/light-v9",
        )
        st.pydeck_chart(r)

    else:
        st.pydeck_chart()


def init_session_state():
    """Initialize session state."""
    st.session_state["model"] = None


def main():
    st.set_page_config(layout="wide")
    if "model" not in st.session_state:
        st.session_state["model"] = None

    st.markdown(
        """
    <style>
        .stAppDeployButton {display:none;}
        .stAppHeader {display:none;}
        .block-container {
               padding-top: 2rem;
               padding-bottom: 0rem;
            }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 style='text-align: center;'>FIM 2D Model Development Tool</h1>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        editor_tab()
    with c2:
        map_tab()


if __name__ == "__main__":
    main()
