from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twodimfim.models.data_models import (
        HydraulicModelRun,
        ModelDomain,
        VectorDataset,
    )


def write_bci_file(bci_path: str | Path, pts: list[list[str | float]]) -> None:
    # Clean points
    pts_clean = []
    for i in pts:
        i[0] = str(i[0])
        i[1] = str(round(float(i[1]), 4))
        i[2] = str(round(float(i[2]), 4))
        i[3] = str(i[3])
        i[4] = str(i[4])
        if len(i) < 6:
            i.append("\n")
        else:
            i[5] = "\n"
        pts_clean.append(i[:6])

    # Format lines and write
    bc_lines = [" ".join(i) for i in pts_clean]
    with open(bci_path, mode="w+") as f:
        f.writelines(bc_lines)


def write_par_file(run: "HydraulicModelRun", domain: "ModelDomain") -> None:
    cfg = {
        "resroot": run.idx,
        "dirroot": run.run_dir,
        "DEMfile": getattr(domain.terrain, "path", None),
        "manningfile": getattr(domain.roughness, "path", None),
        "bcifile": str(run.bcifile_path),
        "saveint": run.save_interval,
        "massint": run.mass_interval,
        "sim_time": run.sim_time,
        "initial_tstep": run.initial_tstep,
        "acceleration": "",
    }
    if run.use_cuda:
        cfg["cuda"] = ""
    if run.elevoff:
        cfg["elevoff"] = ""
    if run.initial_state:
        cfg["startfile"] = str(run.initial_state)
    with open(run.parfile_path, mode="w") as f:
        for k, v in cfg.items():
            f.write(f"{k} {v}\n")
