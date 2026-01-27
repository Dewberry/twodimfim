import sqlite3
from pathlib import Path
from typing import Literal

from twodimfim.consts import OLD_HF_NETWORK_FORMAT


def list_gpkg_layers(gpkg_path: str | Path) -> list[str]:
    with sqlite3.connect(gpkg_path) as conn:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cur.fetchall()]


def get_hf_type(gpkg_path: str | Path) -> Literal["old", "new"]:
    layers = list_gpkg_layers(gpkg_path)

    return "old" if OLD_HF_NETWORK_FORMAT["tree_layer"] in layers else "new"
