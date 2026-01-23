import os
from pathlib import Path
from typing import Literal

### CUSTOM TYPES ###

BCType = Literal["QFIX", "HFIX", "FREE", "TRANSFER"]
UnitsType = Literal["feet", "meters"]
RunType = Literal["unsteady", "quasi-steady"]
SupportedModels = Literal["LISFLOOD-FP", "SFINCS", "TRITON"]
ModelPathTypes = Literal["relative", "absolute"]
SourceType = Literal["file", "url"]

### CONVERSIONS ###

FT_TO_METERS = 0.3048

### PATHS ###

DEFAULT_MODEL_PATH_NAME = "model.json"
DEFAULT_RASTER_DIR = "rasters"
DEFAULT_VECTOR_DIR = "vectors"
DEFAULT_RUN_DIR = "runs"

### HYDROFABRIC ###

HYDROFABRIC_DIR = Path(os.getenv("HYDROFABRIC_DIR", "/hydrofabric"))
OLD_HF_NETWORK_FORMAT = {
    "tree_layer": "network",
    "id_col": "flowpath_id",
    "from_id_col": "flowpath_id",
    "to_id_col": "flowpath_toid",
    "da_col": "tot_drainage_areasqkm",
    "length_col": "flowpath_lengthkm",
    "id_prefix": "fp-",
    "from_id_prefix": "fp-",
    "to_id_prefix": "nex-",
    "divides_layer": "divides",
    "divide_id_col": "divide_id",
    "divide_id_prefix": "cat-",
    "stream_layer": "flowpaths",
    "stream_id_col": "flowpath_id",
    "stream_id_prefix": "fp-",
}
NEW_HF_NETWORK_FORMAT = {
    "tree_layer": "flowpaths",
    "id_col": "fp_id",
    "from_id_col": "up_nex_id",
    "to_id_col": "dn_nex_id",
    "da_col": "total_da_sqkm",
    "length_col": "length_km",
    "id_prefix": "",
    "from_id_prefix": "",
    "to_id_prefix": "",
    "divides_layer": "divides",
    "divide_id_col": "div_id",
    "divide_id_prefix": "",
    "stream_layer": "flowpaths",
    "stream_id_col": "fp_id",
    "stream_id_prefix": "",
}

### EXTERNAL URLS ###

NLCD_WMS_URL = "https://www.mrlc.gov/geoserver/mrlc_download/wms"
USGS_3DEP_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt"

### LISFLOOD ###

DEM_FILE = "dem.ascii"
MANNINGS_FILE = "mann.ascii"
BCI_FILE = "bc.bci"
PAR_FILE = "par.par"

USE_CUDA = os.getenv("USE_CUDA", "True").lower() == "true"
ELEVOFF = os.getenv("ELEVOFF", "True").lower() == "true"

### SETTINGS ###

COMMON_CRS = "EPSG:5070"
MANNINGS_LC_LOOKUP = {
    11: 0.04,
    21: 0.04,
    22: 0.1,
    23: 0.08,
    24: 0.15,
    31: 0.025,
    41: 0.16,
    42: 0.16,
    43: 0.16,
    52: 0.1,
    71: 0.035,
    81: 0.03,
    82: 0.035,
    90: 0.12,
    95: 0.07,
}
