from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import geopandas as gpd

from twodimfim.consts import (
    DA_COL,
    DS_ID_PREFIX,
    LENGTH_COL,
    STREAM_ID_COL,
    STREAM_ID_PREFIX,
    STREAM_TOID_COL,
    TREE_LAYER,
)


@dataclass
class RiverReach:
    us: List[int] = field(default_factory=list)
    us_da: List[float] = field(default_factory=list)
    us_ms: int = -9999
    ds: int = -9999
    length_km: int = -9999


class NetworkWalker:
    def __init__(self, gpkg_path: str | Path):
        df = gpd.read_file(
            gpkg_path,
            layer=TREE_LAYER,
            columns=[STREAM_TOID_COL, STREAM_ID_COL, DA_COL, LENGTH_COL],
        )
        df[STREAM_ID_COL] = (
            df[STREAM_ID_COL].str.replace(STREAM_ID_PREFIX, "").astype(int)
        )
        df[STREAM_TOID_COL] = (
            df[STREAM_TOID_COL].str.replace(DS_ID_PREFIX, "").astype(int)
        )
        df = df.drop_duplicates()
        self.root_node = 0

        self.network = defaultdict(RiverReach)
        for r in df.itertuples(index=False):
            stream_toid = getattr(r, STREAM_TOID_COL)
            stream_id = getattr(r, STREAM_ID_COL)
            self.network[stream_toid].us.append(stream_id)
            self.network[stream_toid].us_da.append(getattr(r, DA_COL))
            self.network[stream_id].ds = stream_toid
            self.network[stream_id].length_km = getattr(r, LENGTH_COL)
        self.network = dict(
            self.network
        )  # re-cast to plain dict so that non-existent nodes don't have blank entries

        idx = df.groupby(STREAM_TOID_COL)[DA_COL].idxmax()
        max_us_map = dict(zip(df.loc[idx, STREAM_TOID_COL], df.loc[idx, STREAM_ID_COL]))
        for k, v in self.network.items():
            v.us_ms = max_us_map.get(k, -9999)

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
