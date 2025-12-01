import os

DATA_DIR = os.getenv("MODEL_DIR", "/data")
REMOTE_DATA_DIR = os.getenv("REMOTE_DATA_DIR", "/remote/data")
MODEL_HOST = os.getenv("MODEL_HOST", "lisflood-model")
MODEL_PORT = os.getenv("MODEL_PORT", "5000")
BASE_URL = f"http://{MODEL_HOST}:{MODEL_PORT}"
TITILER_HOST = os.environ.get("TITILER_HOST", "http://localhost")
TITILER_PORT = os.environ.get("TITILER_PORT", "8000")
TITILER_URL = f"http://{TITILER_HOST}:{TITILER_PORT}"

MARKDOWN_DIVIDER = """
            <div style="width:100%; margin-top:10px; margin-bottom:30px; padding:0;">
            <hr style="margin:0; padding:0; height:1px; border:none; background-color:#919191;">
            </div>
            """


### BASEMAP LAYERS ###


BASEMAPS = [
    {
        "name": "OpenStreetMap",
        "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
        "is_base": True,
    },
    {
        "name": "CartoDB Positron",
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "attribution": "© CartoDB",
        "is_base": True,
    },
    {
        "name": "CartoDB Dark Matter",
        "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "attribution": "© CartoDB",
        "is_base": True,
    },
    {
        "name": "Esri World Imagery",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles © Esri",
        "is_base": True,
    },
    {
        "name": "Esri Topographic",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles © Esri",
        "is_base": True,
    },
]
