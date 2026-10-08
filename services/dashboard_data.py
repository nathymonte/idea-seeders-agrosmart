import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

from services.data_lake import read_dataset
from services.paths import DATA_LAKE_PATH, IMAGE_CSV_PATH

SENSOR_COLUMNS = ["event_id", "field_id", "event_timestamp", "soil_moisture_percent",
                  "air_temperature_celsius", "air_humidity_percent", "luminosity_lux"]


def sensor_frame(start: datetime, end: datetime, fields: set[str] | None = None,
                 base_path=DATA_LAKE_PATH) -> pd.DataFrame:
    records = read_dataset("trusted", "sensor_readings", base_path, start, end, fields)
    frame = pd.DataFrame(records) if records else pd.DataFrame(columns=SENSOR_COLUMNS)
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True)
    return frame.sort_values("event_timestamp")


def image_frame(path=IMAGE_CSV_PATH) -> pd.DataFrame:
    columns = ["image_name", "image_path", "field_id", "data_analise", "predicted_label",
               "predicted_label_pt", "confidence", "actual_label", "is_correct", "metadata_source",
               "label_source", "anomalia"]
    source = Path(path)
    if not source.exists():
        return pd.DataFrame(columns=columns)
    with source.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    legacy = "field_id" not in frame
    for column in columns:
        if column not in frame:
            frame[column] = ""
    frame["field_id"] = frame["field_id"].fillna("")
    if legacy:
        frame["metadata_source"] = "legacy_simulated"
        frame["label_source"] = "filename_hint"
    frame["data_analise"] = pd.to_datetime(frame["data_analise"], utc=True, errors="coerce", format="mixed")
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    return frame


def filter_images(frame: pd.DataFrame, start: datetime, end: datetime,
                  fields: set[str] | None = None) -> pd.DataFrame:
    selected = frame[frame["data_analise"].between(start, end)]
    if fields is not None:
        selected = selected[selected["field_id"].isin(fields)]
    return selected.copy()
