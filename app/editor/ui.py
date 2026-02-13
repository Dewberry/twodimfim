import shutil
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from pyproj import CRS
from shapely import line_interpolate_point
from shapely.ops import unary_union

from app.consts import BASE_URL, DATA_DIR, DEFAULT_VECTOR_FILE_TYPE, MARKDOWN_DIVIDER
from app.editor.functions import (
    make_new_empty_model,
    make_new_hydrofabric_model,
    pull_from_remote,
    push_to_remote,
    run_model,
    save_model,
)
from app.utils import list_models
from app.viewer.functions import add_vector_layer, sync_layers
from twodimfim.consts import HYDROFABRIC_DIR
from twodimfim.models.data_models import (
    BoundaryCondition,
    DatasetMetadata,
    HydraulicModel,
    HydraulicModelMetadata,
    HydraulicModelRun,
    ModelDomain,
    VectorDataset,
)
from twodimfim.utils.geospatial import BBox, export_wse_contour

### WIDGETS ###


def bc_maker(defaults: list[dict] | None = None, editable: bool = True):
    """Specific data editor for boundary conditions."""
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
            geoms_.append(i.geometry_vector)
            types.append(i.bc_type)
            values.append(i.value)
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
        "Geometry": st.column_config.SelectboxColumn(
            options=[k for k in geoms] + ["all"]
        ),
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


### POP-UP DIALOGS ###


@st.dialog("Create a new model")
def new_model():
    """Create a new HydraulicModel and store it in session state."""
    t1, t2 = st.tabs(["From Hydrofabric", "Custom Model"])
    with t1:
        hf_available = [i for i in HYDROFABRIC_DIR.iterdir() if i.is_file()]
        hf = st.selectbox(label="Hydrofabric Source", options=hf_available)
        reach_id = st.number_input(label="Reach ID", step=1)
        resolution = st.number_input(label="Model Resolution (m)", value=10)
        ds_connection_run = st.text_input(
            label="(optional) Downstream connection run", value=""
        )
        if ds_connection_run == "":
            ds_connection_run = None
        if st.button("Create Model", key="hydrofabric_model"):
            make_new_hydrofabric_model(
                hf.resolve(), reach_id, resolution, ds_connection_run
            )
            st.rerun()
    with t2:
        model_id = st.text_input(label="Model ID", value="")
        crs = st.text_input(label="CRS", value="EPSG:5070")
        if st.button("Create Model", key="custom_model"):
            make_new_empty_model(model_id, crs)
            st.rerun()


@st.dialog("Delete model")
def delete_model():
    """Delete the current model from disk and session state."""
    st.markdown(
        f"Are you sure that you want to delete model {st.session_state['model'].metadata.title}?"
    )
    if st.button("Delete model"):
        shutil.rmtree(st.session_state["model"]._context.model_root)
        st.session_state["model"] = None
        st.rerun()


@st.dialog("Load a model")
def load_model():
    """Load a HydraulicModel from disk into session state."""
    path = st.selectbox("Saved models", options=list_models())
    if st.button("Open model"):
        st.session_state["model"] = HydraulicModel.from_file(
            Path(DATA_DIR) / path / "model.json"
        )
        st.rerun()


@st.dialog("Create a new run")
def new_run():
    """Create a new HydraulicModelRun and add it to the current model."""
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
    """Create a new ModelDomain and add it to the current model."""
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
    """Add a new VectorDataset to the current model."""
    vector_dir = st.session_state["model"]._context.model_root / "vectors"
    existing_paths = set([i.path for i in st.session_state["model"].vectors.values()])
    opts = [i.name for i in vector_dir_list(vector_dir) if i not in existing_paths]
    geom = st.selectbox("Geometry", options=opts)
    ds_src = st.text_input("Describe data original source")
    mods = st.text_input("Describe any manipulation done to the source dataset")
    if st.button("Add geometry", disabled=ds_src == ""):
        idx = geom.split(".")[0]
        meta = DatasetMetadata(
            "file", ds_src, transformations=[f"user_modifications: {mods}"]
        )
        stem = str(vector_dir / geom)
        geom = VectorDataset(idx, stem, meta, st.session_state["model"]._context)
        st.session_state["model"].vectors[idx] = geom
        st.session_state["model"].save()
        st.rerun()


def vector_dir_list(path: Path) -> list[Path]:
    """List all vector files in a directory."""
    vector_files = []
    for ext in [".shp", ".geojson", ".gpkg", ".parquet"]:
        vector_files.extend(list(path.glob(f"*{ext}")))
    return vector_files


@st.dialog("Create a connection")
def new_connection():
    """Create a connection to another HydraulicModelRun."""
    idx = st.text_input("Connection ID")
    path = st.selectbox("Saved models", options=list_models())
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
        st.session_state["model"].add_connection(idx, str(model_path), run_id)
        st.rerun()


### DEFAULT DIALOGS ###


def model_control():
    """Panel for creating, loading, saving, and deleting models."""
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
    """Panel for viewing and editing model metadata."""
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
    """Dialog for creating and editing model geometries."""
    geoms = st.session_state["model"].vectors.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        geom_ = st.selectbox("Select a geometry", geoms, index=0)
        st.button("New geometry", on_click=add_vector)
        if st.button("Delete geometry"):
            del st.session_state["model"].vectors[geom_]
            st.rerun()


def domain_editor():
    """Dialog for creating and editing model domains."""
    # Domain selection
    domains = st.session_state["model"].domains.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        domain_ = st.selectbox("Select a domain", domains, index=0)
        st.button("New domain", on_click=new_domain)
        if st.button("Delete domain"):
            del st.session_state["model"].domains[domain_]
            st.rerun()

    # Edit dialog
    if domain_ is not None:
        # Information and update
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

        # Terrain and roughness
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
    """Dialog for creating and editing model connections."""
    # Connection selection
    cnx = st.session_state["model"].connections.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        cnx_ = st.selectbox("Select a connection", cnx, index=0, key="cedit")
        st.button("New connection", on_click=new_connection)
        if st.button("Delete connection"):
            del st.session_state["model"].connections[cnx_]
            cnx_ = None

    # Edit dialog
    if cnx_ is not None:
        cols = st.columns(2, vertical_alignment="top")
        with cols[0]:
            c = st.session_state["model"].connections[cnx_]
            text = f"**ID**: {c.idx}\n\n**Model Path**: {c.model_path}\n\n**Run ID**: {c.run_id}\n\n"
            st.markdown("#### Connection Information")
            st.markdown(text)
        with cols[1]:
            st.markdown("#### Import Geometry")
            if st.button("Import Inundation Polygon", width="stretch"):
                st.session_state["model"].import_connection_inun_poly(
                    cnx_, f_type=DEFAULT_VECTOR_FILE_TYPE
                )
                st.toast(
                    "Imported Connection Inundation Polygon",
                    icon=":material/download_done:",
                )
            if st.button("Import STL", width="stretch"):
                st.session_state["model"].import_connection_stl(
                    cnx_, f_type=DEFAULT_VECTOR_FILE_TYPE
                )
                st.toast("Imported Connection STL", icon=":material/download_done:")


def run_editor():
    """Dialog for creating and editing model runs."""
    # Run selection
    runs = st.session_state["model"].runs.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        run_ = st.selectbox("Select a run", runs, index=0, key="redit")
        st.button("New run", on_click=new_run)
        if st.button("Delete run"):
            del st.session_state["model"].runs[run_]
            run_ = None

    # Edit dialog
    if run_ is not None:
        r: HydraulicModelRun = st.session_state["model"].runs[run_]
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
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
                    "Mass interval (s)", value=r.mass_interval
                )
                sim_time = st.number_input("Simulation time (s)", value=r.sim_time)
                if r.steady_state_tolerance is not None:
                    def_ss = r.steady_state_tolerance * 100
                else:
                    def_ss = 0.0
                steady_state_tolerance = (
                    st.number_input(
                        "Steady state tolerance (%)",
                        value=def_ss,
                    )
                    / 100
                )
                initial_tstep = st.number_input(
                    "Initial timestep (s)", value=r.initial_tstep
                )
            with c2:
                bcs = bc_maker(r.boundary_conditions)
                hot_start_opts = [None] + [str(i.name) for i in r.depth_file_paths]
                if r.initial_state is not None and r.initial_state in hot_start_opts:
                    hs_idx = hot_start_opts.index(Path(r.initial_state).name)
                else:
                    hs_idx = 0
                hot_start = st.selectbox(
                    "Hot start",
                    options=hot_start_opts,
                    index=hs_idx,
                )
                elevoff = not st.checkbox("Export Elevations", value=not r.elevoff)
                if st.button("Save run", use_container_width=True, type="primary"):
                    r.run_type = rtype
                    r.domain = domain
                    r.save_interval = save_interval
                    r.mass_interval = mass_interval
                    r.sim_time = sim_time
                    r.steady_state_tolerance = steady_state_tolerance
                    r.initial_tstep = initial_tstep
                    r.elevoff = elevoff
                    bcs_refactor = [
                        BoundaryCondition(
                            geometry_vector=i.Geometry, bc_type=i.Type, value=i.Value
                        )
                        for i in bcs.itertuples()
                    ]
                    r.boundary_conditions = bcs_refactor
                    if hot_start is not None:
                        r.initial_state = str(r.run_dir / hot_start)
                    else:
                        r.initial_state = None
                    st.session_state["model"].add_run(r)
                    st.session_state["model"].save()


def run_executor():
    """Dialog for executing model runs via Lisflood API."""
    runs = st.session_state["model"].runs.keys()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        run_ = st.selectbox("Select a run", runs, index=0, key="rexec")
        exec_run = st.button("Execute run")
    if exec_run:
        run = st.session_state["model"].runs[run_]
        run_path = str(run.parfile_path)
        with st.spinner(f"Running Lisflood model at {run_path}"):
            success = run_model(run_path)
            if success:
                run.export_zarr()
                st.rerun()


def model_editor():
    """Container for model editing dialogs."""
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
    """Top-level editor tab."""
    with st.container(border=True, gap=None):
        # Model controls and main text
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
                            on_click=pull_from_remote,
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

        # Editing dialogs
        st.markdown(MARKDOWN_DIVIDER, unsafe_allow_html=True)
        if st.session_state["model"] is not None:
            model_editor()
        else:
            st.space("small")
