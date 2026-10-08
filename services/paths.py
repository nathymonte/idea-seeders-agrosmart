import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_LAKE_PATH = Path(os.getenv("DATA_LAKE_PATH", str(PROJECT_ROOT / "data_lake")))
IMAGE_CSV_PATH = Path(os.getenv("AGROSMART_IMAGE_CSV", str(PROJECT_ROOT / "output/classificacoes.csv")))


def validate_field_id(value: str) -> str:
    if not value or len(value) > 100 or not all(c.isascii() and (c.isalnum() or c in "-_") for c in value):
        raise ValueError("field_id deve conter apenas letras ASCII, numeros, '-' ou '_'.")
    return value
