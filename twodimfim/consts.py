### HYDROFABRIC ###
HYDROFABRIC_BASE_URI = "/hydrofabric/{vpu}_commmunity_nextgen.gpkg"
DIVIDES_LAYER = "divides"
STREAM_LAYER = "flowpaths"
TREE_LAYER = "network"
STREAM_ID_COL = "flowpath_id"
STREAM_TOID_COL = "flowpath_toid"
DIVIDE_ID_COL = "divide_id"
DA_COL = "tot_drainage_areasqkm"
LENGTH_COL = "flowpath_lengthkm"
STREAM_ID_PREFIX = "fp-"
DIVIDE_ID_PREFIX = "cat-"
DS_ID_PREFIX = "nex-"

### EXTERNAL URLS ###
NLCD_WMS_URL = "https://www.mrlc.gov/geoserver/mrlc_download/wms"
USGS_3DEP_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt"

### LISFLOOD ###
DEM_FILE = "dem.ascii"
MANNINGS_FILE = "mann.ascii"
BCI_FILE = "bc.bci"
PAR_FILE = "par.par"

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

### GENERAL ###
FT_TO_METERS = 0.3048
