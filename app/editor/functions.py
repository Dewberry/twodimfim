import subprocess
from pathlib import Path
from urllib import request

import requests
import streamlit as st

from app.consts import BASE_URL, DATA_DIR, REMOTE_DATA_DIR
from twodimfim.models.data_models import HydraulicModel


def make_new_model(vpu: str, reach_id: int, resolution: float, inflow_width: float):
    """Create a new HydraulicModel and store it in session state."""
    model_root = Path(DATA_DIR) / str(reach_id)
    st.session_state["model"] = HydraulicModel.from_hydrofabric(
        vpu,
        reach_id,
        resolution,
        model_root,
        inflow_width=inflow_width,
        vector_ftype="shp",
    )
    st.session_state["model"].save()


def save_model():
    """Save the current model to disk."""
    if st.session_state["model"] is not None:
        st.session_state["model"].save()


def rsync(src: str, dst: str):
    """Sync files between one directory and another using rsync."""
    cmd = ["rsync", "-a", "--mkpath", src, dst]
    st.toast(f"Copying {src} to {dst}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 23:
        st.toast(f"remote path {src} does not exist")
    elif result.returncode != 0:
        print(result.returncode)
        print(result.stdout)
        print(result.stderr)
    return result.returncode


def push_to_remote():
    """Push model data to remote directory."""
    root = st.session_state["model"]._context.model_root.resolve()
    src = str(root) + "/"
    dst = str(Path(REMOTE_DATA_DIR) / root.relative_to(Path(DATA_DIR))) + "/"
    rcode = rsync(src, dst)
    if rcode == 0:
        st.toast(f"Succesfully pushed data")


def pull_from_remote():
    """Pull model data from remote directory."""
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


def run_model(run_path):
    """Trigger model run via Lisflood API."""
    try:
        response = requests.post(f"{BASE_URL}/run_model", json={"model_dir": run_path})
        if response.status_code == 200:
            result = response.json()
        else:
            st.error(f"Model API returned {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Lisflood API: {e}")
