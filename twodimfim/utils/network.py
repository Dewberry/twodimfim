import pickle
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from turtle import st
from typing import List

import geopandas as gpd
import pandas as pd

from twodimfim.consts import NEW_HF_NETWORK_FORMAT, OLD_HF_NETWORK_FORMAT
from twodimfim.hydrofabric.utils import get_hf_type


@dataclass
class RiverReach:
    us: List[int] = field(default_factory=list)
    us_da: List[float] = field(default_factory=list)
    us_ms: int = -9999
    ds: int = -9999
    length_km: int = -9999
    da: float = -9999.0


class NetworkWalker:
    def __init__(
        self,
        gpkg_path: str | Path,
        tree_layer: str,
        id_col: str,
        from_id_col: str,
        to_id_col: str,
        da_col: str,
        length_col: str,
        id_prefix: str,
        from_id_prefix: str,
        to_id_prefix: str,
    ):
        self.root_node = 0
        network_pickle_path = Path(gpkg_path).with_suffix(".net.pkl")
        if network_pickle_path.exists():
            with open(network_pickle_path, "rb") as file:
                self.network = pickle.load(file)
            return

        cols = list(set([id_col, from_id_col, to_id_col, da_col, length_col]))
        df = gpd.read_file(
            gpkg_path,
            layer=tree_layer,
            columns=cols,
        )

        df = pd.merge(
            df,
            df[[id_col, from_id_col]],
            left_on=to_id_col,
            right_on=from_id_col,
            how="left",
            suffixes=("", "_ds"),
        )
        df[to_id_col] = df[id_col + "_ds"]
        df = df[cols]

        if id_prefix != "":
            df[id_col] = df[id_col].astype(str).str.replace(id_prefix, "").astype(int)
        if from_id_prefix != "":
            df[from_id_col] = (
                df[from_id_col].astype(str).str.replace(from_id_prefix, "").astype(int)
            )
        if to_id_prefix != "":
            df[to_id_col] = (
                df[to_id_col].astype(str).str.replace(to_id_prefix, "").astype(int)
            )

        df = df.drop_duplicates()

        self.network = defaultdict(RiverReach)
        for r in df.itertuples(index=False):
            stream_toid = getattr(r, to_id_col)
            stream_id = getattr(r, id_col)
            self.network[stream_toid].us.append(stream_id)
            self.network[stream_toid].us_da.append(getattr(r, da_col))
            self.network[stream_id].ds = stream_toid
            self.network[stream_id].length_km = getattr(r, length_col)
            self.network[stream_id].da = getattr(r, da_col)
        self.network = dict(
            self.network
        )  # re-cast to plain dict so that non-existent nodes don't have blank entries

        idx = df.groupby(to_id_col)[da_col].idxmax()
        max_us_map = dict(zip(df.loc[idx, to_id_col], df.loc[idx, id_col]))
        for k, v in self.network.items():
            v.us_ms = max_us_map.get(k, -9999)

        with open(network_pickle_path, "wb") as file:
            pickle.dump(self.network, file)

    @classmethod
    def from_gpkg(cls, gpkg_path: str | Path) -> "NetworkWalker":
        """Auto-detect network format from hydrofabric and create NetworkWalker."""
        if get_hf_type(gpkg_path) == "old":
            fmt = OLD_HF_NETWORK_FORMAT
        else:
            fmt = NEW_HF_NETWORK_FORMAT

        return cls(
            gpkg_path=gpkg_path,
            tree_layer=fmt["tree_layer"],
            id_col=fmt["id_col"],
            from_id_col=fmt["from_id_col"],
            to_id_col=fmt["to_id_col"],
            da_col=fmt["da_col"],
            length_col=fmt["length_col"],
            id_prefix=fmt["id_prefix"],
            from_id_prefix=fmt["from_id_prefix"],
            to_id_prefix=fmt["to_id_prefix"],
        )

    @property
    def roots(self):
        return [i for i in self.network if self.network[i].ds not in self.network]

    def walk_network_us(self, start, max_walk_km):
        q = [{"path": [start], "length": 0}]
        walked = set()
        while len(q) > 0:
            path = q.pop(0)
            last_node = self.network.get(path["path"][-1], None)

            if last_node is None or len(last_node.us) == 0:
                walked |= set(path["path"])
                continue

            for i in last_node.us:
                tmp_path = deepcopy(path)
                tmp_path["length"] += last_node.length_km
                if tmp_path["length"] < max_walk_km:
                    tmp_path["path"].append(i)
                    q.append(tmp_path)
                else:
                    walked |= set(tmp_path["path"])

        return list(walked.difference([start]))

    def walk_network_us_ms(self, start, max_walk_km):
        walk_length = 0
        path = [start]
        while walk_length < max_walk_km:
            next_ = self.network.get(path[-1], None)
            if next_ is None:
                break
            walk_length += next_.length_km
            if next_.us_ms == -9999:
                break
            path.append(next_.us_ms)
        return path[1:]

    def walk_network_ds(self, start, max_walk_km):
        walk_length = 0
        path = [start]
        while walk_length < max_walk_km:
            next_ = self.network.get(path[-1], None)
            if next_ is None:
                break
            walk_length += next_.length_km
            path.append(next_.ds)
            if next_.ds == self.root_node:
                break
        return path[1:]
