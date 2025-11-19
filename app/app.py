import os
import shutil
import subprocess
from gc import disable
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import rasterio
import requests
import streamlit as st
from PIL import Image
from pyproj import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely import unary_union

from twodimfim.models.data_models import (
    BoundaryCondition,
    HydraulicModel,
    HydraulicModelMetadata,
    HydraulicModelRun,
    ModelConnection,
    ModelDomain,
    VectorDataset,
)
from twodimfim.utils.etl import DatasetMetadata
from twodimfim.utils.geospatial import BBox

DATA_DIR = os.getenv("MODEL_DIR", "/data")
REMOTE_DATA_DIR = os.getenv("REMOTE_DATA_DIR", "/remote/data")
MODEL_HOST = os.getenv("MODEL_HOST", "lisflood-model")
MODEL_PORT = os.getenv("MODEL_PORT", "5000")
BASE_URL = f"http://{MODEL_HOST}:{MODEL_PORT}"

MARKDOWN_DIVIDER = """
            <div style="width:100%; margin-top:10px; margin-bottom:30px; padding:0;">
            <hr style="margin:0; padding:0; height:1px; border:none; background-color:#919191;">
            </div>
            """

### WIDGETS ###


def bc_maker(defaults: list[dict] | None = None, editable: bool = True):
    if defaults is None:
        bc = pd.DataFrame(
            {
                "Geometry": [],
                "Type": [],
                "Value": [],
            }
        )
    else:
        geoms_ = []
        types = []
        values = []
        for i in defaults:
            geoms_.append(i["geometry_vector"])
            types.append(i["bc_type"])
            values.append(i["value"])
        bc = pd.DataFrame(
            {
                "Geometry": geoms_,
                "Type": types,
                "Value": values,
            }
        )
    bc["Geometry"] = bc["Geometry"].astype(str)
    bc["Type"] = bc["Type"].astype(str)
    bc["Value"] = bc["Value"].astype(str)
    geoms = st.session_state["model"].vectors.keys()
    ccfg = {
        "Geometry": st.column_config.SelectboxColumn(options=geoms),
        "Type": st.column_config.SelectboxColumn(
            options=["QFIX", "HFIX", "FREE", "TRANSFER"]
        ),
        "Value": st.column_config.TextColumn(),
    }
    st.text("Boundary Conditions")
    if editable:
        nr = "dynamic"
    else:
        nr = "fixed"
    return st.data_editor(bc, num_rows=nr, column_config=ccfg)


### MAP PANEL ###


def make_raster_layer(raster_path):
    raster_path = Path(raster_path)

    # Open the raster
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype("float32")
        transform = src.transform
        width, height = src.width, src.height
    src_crs = st.session_state["model"]._context.crs

    # Reproject to EPSG:4326
    dst_crs = "EPSG:4326"
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, *src.bounds
    )
    dst_data = np.zeros((dst_height, dst_width), dtype="float32")
    reproject(
        source=data,
        destination=dst_data,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    # Normalize values to 0-1
    min_val = np.nanmin(dst_data)
    max_val = np.nanmax(dst_data)
    if max_val > min_val:
        norm = (dst_data - min_val) / (max_val - min_val)
    else:
        norm = np.zeros_like(dst_data)

    # Define cyan -> dark blue gradient
    nan_mask = norm == 0
    r_channel = np.zeros_like(norm, dtype=np.uint8)
    g_channel = ((1 - norm) * 255).astype(np.uint8)  # interpolate green
    b_channel = np.ones_like(norm, dtype=np.uint8) * 255  # interpolate blue
    alpha_channel = np.full_like(r_channel, 178, dtype=np.uint8)  # 70% opacity
    alpha_channel[nan_mask] = 0

    # Stack into RGBA and save PNG
    img = np.stack([r_channel, g_channel, b_channel, alpha_channel], axis=2)
    pil_img = Image.fromarray(img, mode="RGBA")
    png_path = Path(raster_path).with_suffix(".png")
    pil_img.save(png_path)

    # Return BitmapLayer with WGS84 bounds
    minx, miny = dst_transform * (0, dst_height)
    maxx, maxy = dst_transform * (dst_width, 0)
    return pdk.Layer(
        "BitmapLayer",
        image=str(png_path),
        bounds=[minx, miny, maxx, maxy],
        opacity=0.7,
    )


def map_tab():
    if st.session_state["model"] is not None:
        layers = []
        if len(st.session_state["model"].runs) > 0:
            run_ = next(iter(st.session_state["model"].runs))
            run = st.session_state["model"].runs[run_]
            raster_path = Path(run.run_dir) / f"{run_}.max"
            if raster_path.exists():
                layers.append(make_raster_layer(str(raster_path)))
        for k in [
            "all_us_divides",
            "all_ds_divides",
            "divide",
            "us_bc_line",
            "centerline",
        ]:
            if not k in st.session_state["model"].vectors:
                continue
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
                auto_highlight=True,
                highlight_color=[0, 0, 0, 100],
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


### MODEL EDITOR PANEL ###


def make_new_model(vpu, reach_id, resolution, inflow_width):
    model_root = Path(DATA_DIR) / str(reach_id)
    st.session_state["model"] = HydraulicModel.from_hydrofabric(
        vpu, reach_id, resolution, model_root, inflow_width=inflow_width
    )
    st.session_state["model"].save()


@st.dialog("Create a new model")
def new_model():
    vpu = st.text_input(label="Vector Processing Unit (VPU)", value="1")
    reach_id = st.number_input(label="Reach ID", step=1)
    resolution = st.number_input(label="Model Resolution (m)", value=10)
    inflow_width = st.number_input(label="Inflow Width (m)", value=100)
    if st.button("Create Model"):
        make_new_model(vpu, reach_id, resolution, inflow_width)
        st.rerun()


@st.dialog("Delete model")
def delete_model():
    st.markdown(
        f"Are you sure that you want to delete model {st.session_state['model'].metadata.title}?"
    )
    if st.button("Delete model"):
        shutil.rmtree(st.session_state["model"]._context.model_root)
        st.session_state["model"] = None
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
    bcs = bc_maker()

    with st.expander("Additional parameters"):
        save_interval = st.number_input("Save Interval", value=900)
        mass_interval = st.number_input("Mass Interval", value=15)
        sim_time = st.number_input("Simulation Time", value=3600)
        steady_state_tolerance = st.number_input("Steady State Tolerance", value=0.95)
        initial_tstep = st.number_input("Initial Timestep", value=0.5)

    if st.button("Save Run"):
        bcs_refactor = [
            BoundaryCondition(geometry_vector=i.Geometry, bc_type=i.Type, value=i.Value)
            for i in bcs.itertuples()
        ]
        tmp_run = HydraulicModelRun(
            idx=idx,
            run_type=type_.lower(),
            domain=domain,
            boundary_conditions=bcs_refactor,
            save_interval=save_interval,
            mass_interval=mass_interval,
            sim_time=sim_time,
            steady_state_tolerance=steady_state_tolerance,
            initial_tstep=initial_tstep,
            _context=st.session_state["model"]._context,
        )
        st.session_state["model"].add_run(tmp_run)
        st.rerun()


@st.dialog("Create a new domain")
def new_domain(cur_domain: str = ""):
    st.markdown("Create a domain from the bounding box of selected geometries.")
    idx = st.text_input("Domain ID", value=cur_domain)
    if cur_domain in st.session_state["model"].domains:
        default_res = st.session_state["model"].domains[idx].resolution
        but_text = "Update domain"
    else:
        default_res = 10
        but_text = "Create domain"
    resolution = st.number_input("Resolution (m)", value=default_res)
    vectors = st.session_state["model"].vectors
    geoms = st.multiselect("Geometries", options=vectors.keys())
    buffer = st.number_input("Buffer distance (m)", value=100)
    if st.button(but_text):
        base_bbox = BBox(*unary_union([vectors[i].shape for i in geoms]).bounds)
        base_bbox.buffer(buffer)
        st.session_state["model"].domains[idx] = ModelDomain.from_bbox(
            idx, base_bbox, resolution, st.session_state["model"]._context
        )
        st.rerun()


@st.dialog("Create a vector layer")
def add_vector():
    vector_dir = next(iter(st.session_state["model"].vectors.values())).path.parent
    existing_paths = set([i.path for i in st.session_state["model"].vectors.values()])
    vector_files = list(set(vector_dir.iterdir()).difference(existing_paths))
    opts = [i.stem for i in vector_files]
    geom = st.selectbox("Geometry", options=opts)
    ds_src = st.text_input("Describe data original source")
    mods = st.text_input("Describe any manipulation done to the source dataset")
    if st.button("Add geometry", disabled=ds_src == ""):
        i = opts.index(geom)
        idx = geom
        meta = DatasetMetadata(
            "file", ds_src, transformations=[f"user_modifications: {mods}"]
        )
        stem = str(
            vector_files[i].relative_to(st.session_state["model"]._context.model_root)
        )
        geom = VectorDataset(idx, stem, meta, st.session_state["model"]._context)
        st.session_state["model"].vectors[idx] = geom


@st.dialog("Create a connection")
def new_connection():
    idx = st.text_input("Connection ID")
    all_models = [
        str(i.name)
        for i in Path(DATA_DIR).iterdir()
        if i.is_dir() and (i / "model.json").exists()
    ]
    path = st.selectbox("Saved models", options=all_models)
    if path is not None:
        model_path = Path(DATA_DIR) / path / "model.json"
        tmp_model = HydraulicModel.from_file(model_path)
        run_opts = tmp_model.runs.keys()
        disable = False
    else:
        run_opts = []
        disable = True
    run_id = st.selectbox("Model runs", options=run_opts, disabled=disable)
    if st.button("Create connection", disabled=run_id is None):
        st.session_state["model"].add_connection(idx, model_path, run_id)
        st.rerun()


def save_model():
    if st.session_state["model"] is not None:
        st.session_state["model"].save()


def rsync(src: str, dst: str):
    cmd = ["rsync", "-a", "--mkpath", src, dst]
    print(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 23:
        st.toast(f"remote path {src} does not exist")
    elif result.returncode != 0:
        print(result.returncode)
        print(result.stdout)
        print(result.stderr)
    return result.returncode


def push_to_remote():
    root = st.session_state["model"]._context.model_root.resolve()
    src = str(root) + "/"
    dst = str(Path(REMOTE_DATA_DIR) / root.relative_to(Path(DATA_DIR))) + "/"
    rcode = rsync(src, dst)
    if rcode == 0:
        st.toast(f"Succesfully pushed data")


def pull_from_Remote():
    root = st.session_state["model"]._context.model_root.resolve()
    dst = str(root) + "/"
    src = str(Path(REMOTE_DATA_DIR) / root.relative_to(Path(DATA_DIR))) + "/"
    rcode = rsync(src, dst)
    if rcode == 0:
        st.toast(f"Succesfully pulled data")
    new_mod = root / "model.json"
    if new_mod.exists():
        st.session_state["model"] = HydraulicModel.from_file(new_mod)
    else:
        st.session_state["model"] = None


def model_control():
    with st.container(width=550, vertical_alignment="bottom") as c:
        c1, c2, c3, c4 = st.columns(4, gap="small")
        c1.button("New Model", width=125, on_click=new_model, type="primary")
        c2.button("Load Model", width=125, on_click=load_model, type="primary")
        c3.button(
            "Save Model",
            width=125,
            on_click=save_model,
            type="primary",
            disabled=st.session_state["model"] is None,
        )
        c4.button(
            "Delete Model",
            width=125,
            on_click=delete_model,
            type="primary",
            disabled=st.session_state["model"] is None,
        )


def model_summary_panel():
    meta: HydraulicModelMetadata = st.session_state["model"].metadata
    c1, c2 = st.columns(2)
    with c1:
        title = st.text_input("Title", meta.title)
        author = st.text_input("Author", meta.author)
        brands = ["LISFLOOD-FP", "TRITON", "SFINCS"]
        brand = st.selectbox(
            "Brand", options=brands, index=brands.index(meta.model_brand)
        )
        crs = st.text_input("CRS", ":".join(meta.crs.to_authority()))
    with c2:
        notes = st.text_area("Engineer's notes", meta.engineer_notes, height=178)
        if len(meta.tags) > 0:
            tag_opts = meta.tags
        else:
            tag_opts = []
        tags = st.multiselect(
            "Tags", options=tag_opts, accept_new_options=True, default=meta.tags
        )
        if st.button("Save Summary", use_container_width=True, type="primary"):
            meta.title = title
            meta.author = author
            meta.model_brand = brand
            meta.crs = CRS.from_user_input(crs)
            meta.engineer_notes = notes
            meta.tags = tags
            st.session_state["model"].save()


def geometry_editor():
    if st.button("Add vector", type="primary", use_container_width=True):
        add_vector()
    to_delete = []
    for i in st.session_state["model"].vectors:
        with st.container(border=True):
            c1, c2 = st.columns(2, vertical_alignment="center")
            with c1:
                with st.container(horizontal_alignment="left"):
                    st.markdown(i)
            with c2:
                with st.container(horizontal_alignment="right"):
                    if st.button("", icon=":material/close:", key=f"close_{i}"):
                        to_delete.append(i)

    for i in to_delete:
        del st.session_state["model"].vectors[i]
    if len(to_delete) > 0:
        st.rerun()


def domain_editor():
    domains = st.session_state["model"].domains.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        domain_ = st.selectbox("Select a domain", domains, index=0)
        st.button("New domain", on_click=new_domain)
        if st.button("Delete domain"):
            del st.session_state["model"].domains[domain_]
            st.rerun()
    if domain_ is not None:
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
                if st.button("Update domain"):
                    new_domain(domain_)
        with st.container(border=True):
            has_terrain = domain.terrain is not None
            c1, c2 = st.columns(2, vertical_alignment="bottom")
            with c1:
                with st.container(horizontal_alignment="left"):
                    st.markdown("### Terrain")
                    if has_terrain:
                        st.badge("Terrain created", color="green")
                    else:
                        st.badge("Terrain not available", color="grey")
            with c2:
                with st.container(
                    horizontal_alignment="right", vertical_alignment="bottom"
                ):
                    if st.button("Download USGS Terrain", disabled=has_terrain):
                        st.toast(
                            "Downloading USGS terrain", icon=":material/cloud_download:"
                        )
                        domain.load_3dep_terrain()
                        st.toast(
                            "Downloaded USGS terrain", icon=":material/download_done:"
                        )
                        st.rerun()

        with st.container(border=True):
            has_roughness = domain.roughness is not None
            c1, c2 = st.columns(2, vertical_alignment="bottom")
            with c1:
                with st.container(horizontal_alignment="left"):
                    st.markdown("### Roughness")
                    if has_roughness:
                        st.badge("Roughness created", color="green")
                    else:
                        st.badge("Roughness not available", color="grey")
            with c2:
                with st.container(
                    horizontal_alignment="right", vertical_alignment="bottom"
                ):
                    if st.button("Download NLCD Roughness", disabled=has_roughness):
                        st.toast(
                            "Downloading NLCD Roughness",
                            icon=":material/cloud_download:",
                        )
                        domain.load_nlcd_roughness()
                        st.toast(
                            "Downloaded NLCD Roughness", icon=":material/download_done:"
                        )
                        st.rerun()


def connection_editor():
    cnx = st.session_state["model"].connections.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        cnx_ = st.selectbox("Select a connection", cnx, index=0, key="cedit")
        st.button("New connection", on_click=new_connection)
        if st.button("Delete connection"):
            del st.session_state["model"].connections[cnx_]
            cnx_ = None
    if cnx_ is not None:
        st.text(f"{cnx_}")


def run_editor():
    runs = st.session_state["model"].runs.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        run_ = st.selectbox("Select a run", runs, index=0, key="redit")
        st.button("New run", on_click=new_run)
        if st.button("Delete run"):
            del st.session_state["model"].runs[run_]
            run_ = None
    if run_ is not None:
        r: HydraulicModelRun = st.session_state["model"].runs[run_]
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                idx = st.text_input("ID", value=r.idx)
                rtypes = ["Unsteady", "Quasi-steady"]
                i = rtypes.index(r.run_type.capitalize())
                rtype = st.selectbox("Run type", rtypes, index=i).lower()
                domains = list(st.session_state["model"].domains.keys())
                if r.domain in domains:
                    domain = st.selectbox(
                        "Domain", domains, index=domains.index(r.domain)
                    )
                else:
                    domain = st.selectbox("Domain", domains)
                save_interval = st.number_input(
                    "Save interval (s)", value=r.save_interval
                )
                mass_interval = st.number_input(
                    "Mass interval (s)", value=r.save_interval
                )
                steady_state_tolerance = (
                    st.number_input(
                        "Steady state tolerance (%)", value=r.save_interval * 100
                    )
                    / 100
                )
                initial_tstep = st.number_input(
                    "Initial timestep (s)", value=r.save_interval
                )
            with c2:
                bcs = bc_maker(r.boundary_conditions)


def run_executor():
    runs = st.session_state["model"].runs.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        run_ = st.selectbox("Select a run", runs, index=0, key="rexec")
        exec_run = st.button("Execute run")
    if exec_run:
        run_path = str(st.session_state["model"].runs[run_].parfile_path)
        with st.spinner(f"Running Lisflood model at {run_path}"):
            run_model(run_path)
            st.rerun()


def run_model(run_path):
    try:
        response = requests.post(f"{BASE_URL}/run_model", json={"model_dir": run_path})
        if response.status_code == 200:
            result = response.json()
        else:
            st.error(f"Model API returned {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Lisflood API: {e}")


def model_editor():
    t1, t2, t3, t4, t5, t6 = st.tabs(
        [
            "Model Summary",
            "Geometry Editor",
            "Domain Editor",
            "Connection Editor",
            "Run Editor",
            "Run Executor",
        ]
    )
    with t1:
        model_summary_panel()
    with t2:
        geometry_editor()
    with t3:
        domain_editor()
    with t4:
        connection_editor()
    with t5:
        run_editor()
    with t6:
        run_executor()


def editor_tab():
    with st.container(border=True, gap=None):
        c1, c2 = st.columns([1, 2], vertical_alignment="top")
        with c1:
            st.markdown("# Model Editor")
        with c2:
            with st.container(vertical_alignment="top", horizontal_alignment="right"):
                with st.popover("Sync with cloud"):
                    with st.container(horizontal=True):
                        st.button(
                            "Pull from cloud",
                            type="primary",
                            disabled=st.session_state["model"] is None,
                            on_click=pull_from_Remote,
                        )
                        st.button(
                            "Push to cloud",
                            type="primary",
                            disabled=st.session_state["model"] is None,
                            on_click=push_to_remote,
                        )
        c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
        with c1:
            if st.session_state["model"] is None:
                model_name = "None selected"
            else:
                model_name = st.session_state["model"].metadata.title
            st.markdown(f"**Currently editing:** {model_name}")
        with c2:
            model_control()
        st.markdown(MARKDOWN_DIVIDER, unsafe_allow_html=True)
        if st.session_state["model"] is not None:
            model_editor()
        else:
            st.space("small")


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
               padding-top: 0.5rem;
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
