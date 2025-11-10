"""What process I expect the streamlit app to automate."""

# Build the model domain
from dataclasses import asdict
from pathlib import Path

from twodimfim.models.data_models import HydraulicModel, HydraulicModelRun

DATA_DIR = "data"

# Make the model or load from file
resolution = 10
vpu = 1
reach_id = 18122
inflow_width = 100
model_root = Path(DATA_DIR) / str(reach_id)
model_path = model_root / "model.json"
if model_path.exists():
    model = HydraulicModel.from_file(model_path)
else:
    model = HydraulicModel.from_hydrofabric(
        vpu, reach_id, resolution, model_root, inflow_width=inflow_width
    )

    # Download topo and nlcd
    current_domain = next(iter(model.domains.keys()))
    model.domains[current_domain].load_3dep_terrain()
    model.domains[current_domain].load_nlcd_roughness()

    model.to_file(model_path)


# Create several plans.  User inputs discharges
run_dict = [
    {
        "idx": "test",
        "type_": "quasi-steady",
        "domain": "hydrofabric_10",
        "boundary_conditions": [
            {"geometry_vector": "us_bc_line", "type_": "QFIX", "value": 800},
            {"geometry_vector": "all_ds_divides", "type_": "FREE", "value": 0.01},
        ],
    }
]
for run in run_dict:
    model.add_run(HydraulicModelRun(**run))


# Run the model
runs = ["test"]

for i in runs:
    model.execute_run(i)
