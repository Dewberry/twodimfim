from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twodimfim.models.data_models import (
        HydraulicModelRun,
        ModelDomain,
        VectorDataset,
    )


def write_bci_file(
    run: HydraulicModelRun, domain: ModelDomain, vectors: dict[str, VectorDataset]
) -> None:
    bc_lines = []
    for i in run.boundary_conditions:
        row_pts = domain.geometry_to_bc_points(vectors[i.geometry_vector].shape)
        if i.bc_type == "QFIX":
            tmp_val = i.value / (domain.resolution * len(row_pts))
        else:
            tmp_val = i.value
        for j in row_pts:
            bc = " ".join(
                [
                    j[0],
                    str(round(j[1], 4)),
                    str(round(j[2], 4)),
                    i.bc_type,
                    str(tmp_val),
                    "\n",
                ]
            )
            bc_lines.append(bc)
        with open(run.bcifile_path, mode="w+") as f:
            f.writelines(bc_lines)


def write_par_file(run: HydraulicModelRun, domain: ModelDomain) -> None:
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
        "elevoff": "",
    }
    with open(run.parfile_path, mode="w") as f:
        for k, v in cfg.items():
            f.write(f"{k} {v}\n")
