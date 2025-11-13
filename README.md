# twodimfim
2D hydrodynamic flood modeling to support OWP flood inundation mapping (FIM) efforts


# Model Schmas

# `HydraulicModel` Schema

| Field | Type | Default |
|-------|------|----------|
| `metadata` | `HydraulicModelMetadata` | — |
| `domains` | `dict[str, ModelDomain]` | dict |
| `vectors` | `dict[str, VectorDataset]` | dict |
| `runs` | `dict[str, HydraulicModelRun]` | dict |
| `_context` | `HydraulicModelContext` | generate_generic_context |

# `HydraulicModelMetadata` Schema

| Field | Type | Default |
|-------|------|----------|
| `title` | `str` | — |
| `author` | `str` | '' |
| `model_brand` | `Literal[LISFLOOD-FP, SFINCS, TRITON]` | 'LISFLOOD-FP' |
| `engineer_notes` | `str` | '' |
| `tags` | `list[str]` | list |
| `path_types` | `Literal[relative, absolute]` | 'relative' |
| `twodimfim_version` | `str` | '0.1.0' |
| `creation_date` | `datetime` | now |
| `last_edited` | `datetime` | now |
| `crs` | `CRS` | <lambda> |

# `ModelDomain` Schema

| Field | Type | Default |
|-------|------|----------|
| `idx` | `str` | — |
| `transform` | `Affine` | — |
| `rows` | `int` | — |
| `cols` | `int` | — |
| `resolution` | `float` | — |
| `terrain` | `UnionType[Terrain, NoneType]` | None |
| `roughness` | `UnionType[Roughness, NoneType]` | None |
| `_context` | `HydraulicModelContext` | generate_generic_context |

# `Terrain` Schema

| Field | Type | Default |
|-------|------|----------|
| `idx` | `str` | — |
| `path_stem` | `str` | — |
| `vertical_units` | `Literal[feet, meters]` | — |
| `metadata` | `DatasetMetadata` | — |
| `_context` | `HydraulicModelContext` | generate_generic_context |

# `Roughness` Schema

| Field | Type | Default |
|-------|------|----------|
| `idx` | `str` | — |
| `path_stem` | `str` | — |
| `metadata` | `DatasetMetadata` | — |
| `_context` | `HydraulicModelContext` | generate_generic_context |

# `VectorDataset` Schema

| Field | Type | Default |
|-------|------|----------|
| `idx` | `str` | — |
| `path_stem` | `str` | — |
| `metadata` | `DatasetMetadata` | — |
| `_context` | `HydraulicModelContext` | generate_generic_context |

# `HydraulicModelRun` Schema

| Field | Type | Default |
|-------|------|----------|
| `idx` | `str` | — |
| `run_type` | `Literal[unsteady, quasi-steady]` | — |
| `domain` | `str` | — |
| `boundary_conditions` | `list[BoundaryCondition]` | — |
| `save_interval` | `float` | 900 |
| `mass_interval` | `float` | 15 |
| `sim_time` | `UnionType[float, NoneType]` | None |
| `steady_state_tolerance` | `UnionType[float, NoneType]` | None |
| `initial_tstep` | `float` | 0.5 |
| `initial_state` | `UnionType[str, NoneType]` | None |
| `run_dir_stem` | `str` | 'runs' |
| `parfile_name` | `str` | 'par.par' |
| `bcifile_name` | `str` | 'bc.bci' |
| `_context` | `HydraulicModelContext` | generate_generic_context |
