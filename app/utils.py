from pathlib import Path

from app.consts import DATA_DIR


def list_models():
    """List all models in the data directory."""
    return sorted(
        [
            str(i.name)
            for i in Path(DATA_DIR).iterdir()
            if i.is_dir() and (i / "model.json").exists()
        ]
    )
