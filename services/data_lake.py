from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from services.threshold_config import get_threshold_config

DATA_LAKE_PATH = Path(os.getenv("DATA_LAKE_PATH", "data_lake"))
LAYERS = ("raw", "trusted", "refined", "rejected")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: str | datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return utc_now()

    normalized = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def parse_datetime_or_ingestion_time(value: str | datetime | None = None) -> datetime:
    try:
        return parse_datetime(value)
    except ValueError:
        return utc_now()


def ensure_data_lake(base_path: Path | str = DATA_LAKE_PATH) -> Path:
    base = Path(base_path)
    for layer in LAYERS:
        (base / layer).mkdir(parents=True, exist_ok=True)
    return base


def build_partition_path(
    layer: str,
    dataset: str,
    reference_datetime: datetime | str | None = None,
    filename: str = "readings.jsonl",
    base_path: Path | str = DATA_LAKE_PATH,
) -> Path:
    if layer not in LAYERS:
        raise ValueError(f"Camada invalida: {layer}")

    dt = parse_datetime(reference_datetime)
    return _partition_path(layer, dataset, dt, filename, base_path)


def build_safe_partition_path(
    layer: str,
    dataset: str,
    reference_datetime: datetime | str | None = None,
    filename: str = "readings.jsonl",
    base_path: Path | str = DATA_LAKE_PATH,
) -> Path:
    if layer not in LAYERS:
        raise ValueError(f"Camada invalida: {layer}")

    dt = parse_datetime_or_ingestion_time(reference_datetime)
    return _partition_path(layer, dataset, dt, filename, base_path)


def _partition_path(
    layer: str,
    dataset: str,
    dt: datetime,
    filename: str,
    base_path: Path | str,
) -> Path:
    return (
        Path(base_path)
        / layer
        / dataset
        / f"year={dt.year:04d}"
        / f"month={dt.month:02d}"
        / f"day={dt.day:02d}"
        / filename
    )


def append_json_line(path: Path | str, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        file.write("\n")
    return target


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def iter_jsonl_files(layer: str, dataset: str, base_path: Path | str = DATA_LAKE_PATH) -> Iterable[Path]:
    root = Path(base_path) / layer / dataset
    if not root.exists():
        return []
    return root.rglob("*.jsonl")


def read_dataset(layer: str, dataset: str, base_path: Path | str = DATA_LAKE_PATH) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_jsonl_files(layer, dataset, base_path):
        records.extend(read_jsonl(path))
    return records


def write_json(path: Path | str, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
    return target


def save_raw_event(
    event: dict[str, Any],
    dataset: str = "sensors",
    base_path: Path | str = DATA_LAKE_PATH,
) -> Path:
    reference = event.get("event_timestamp") or event.get("timestamp") or event.get("ingested_at")
    path = build_safe_partition_path("raw", dataset, reference, "readings.jsonl", base_path)
    return append_json_line(path, event)


def save_trusted_sensor_reading(reading: dict[str, Any], base_path: Path | str = DATA_LAKE_PATH) -> Path:
    path = build_partition_path(
        "trusted",
        "sensor_readings",
        reading.get("event_timestamp"),
        "readings.jsonl",
        base_path,
    )
    return append_json_line(path, reading)


def save_rejected_sensor_event(
    event: dict[str, Any],
    errors: list[str],
    source: str,
    base_path: Path | str = DATA_LAKE_PATH,
) -> Path:
    reference = event.get("event_timestamp") or event.get("timestamp") or utc_now()
    path = build_safe_partition_path(
        "rejected",
        "sensor_readings",
        reference,
        "rejected.jsonl",
        base_path,
    )
    payload = {
        "event": event,
        "errors": errors,
        "source": source,
        "rejected_at": utc_now().isoformat().replace("+00:00", "Z"),
    }
    return append_json_line(path, payload)


def preserve_historical_csv(source_csv: Path | str, base_path: Path | str = DATA_LAKE_PATH) -> Path:
    source = Path(source_csv)
    now = utc_now()
    target = build_partition_path(
        "raw",
        "historical",
        now,
        f"{now.strftime('%H%M%S')}_{source.name}",
        base_path,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def load_seen_event_ids(base_path: Path | str = DATA_LAKE_PATH) -> set[str]:
    return {
        str(record["event_id"])
        for record in read_dataset("trusted", "sensor_readings", base_path)
        if record.get("event_id")
    }


def load_image_analysis_summary(output_csv: Path | str = "output/classificacoes.csv") -> dict[str, Any] | None:
    path = Path(output_csv)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return None

    latest = rows[-1]
    sick_count = sum(1 for row in rows if row.get("predicted_label") == "Sick")
    return {
        "total_images": len(rows),
        "sick_images": sick_count,
        "latest_prediction": {
            "image_name": latest.get("image_name"),
            "predicted_class": latest.get("predicted_label"),
            "confidence": _as_float(latest.get("confidence")),
            "predicted_at": latest.get("data_analise"),
        },
    }


def recompute_field_status(
    base_path: Path | str = DATA_LAKE_PATH,
    output_csv: Path | str = "output/classificacoes.csv",
    threshold_config: dict[str, Any] | None = None,
) -> list[Path]:
    config = threshold_config or get_threshold_config()
    records = read_dataset("trusted", "sensor_readings", base_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("field_id", "TALHAO-01"))].append(record)

    image_summary = load_image_analysis_summary(output_csv)
    written: list[Path] = []
    for field_id, field_records in grouped.items():
        field_records.sort(key=lambda item: item["event_timestamp"])
        summary = _build_sensor_summary(field_records)
        payload = {
            "field_id": field_id,
            "crop": "tomato",
            "calculation_period": {
                "start": field_records[0]["event_timestamp"],
                "end": field_records[-1]["event_timestamp"],
            },
            "sensor_summary": summary,
            "latest_image_analysis": image_summary,
            "attention_level": _attention_level(summary, image_summary, config),
            "recommendation": _recommendation(summary, image_summary, config),
            "updated_at": utc_now().isoformat().replace("+00:00", "Z"),
        }
        written.append(write_json(Path(base_path) / "refined" / "field_status" / f"{field_id}.json", payload))
    return written


def _build_sensor_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "average_soil_moisture_percent": round(mean(_numbers(records, "soil_moisture_percent")), 2),
        "average_air_temperature_celsius": round(mean(_numbers(records, "air_temperature_celsius")), 2),
        "average_air_humidity_percent": round(mean(_numbers(records, "air_humidity_percent")), 2),
        "average_luminosity_lux": round(mean(_numbers(records, "luminosity_lux")), 2),
        "last_reading_at": records[-1]["event_timestamp"],
        "readings_count": len(records),
    }


def _numbers(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if record.get(key) is not None]


def _attention_level(
    summary: dict[str, Any],
    image_summary: dict[str, Any] | None,
    threshold_config: dict[str, Any],
) -> str:
    risk_signals = _expected_range_risk_signals(summary, threshold_config)
    risk_signals.append(bool(image_summary and image_summary.get("sick_images", 0) > 0))
    total = sum(1 for signal in risk_signals if signal)
    if total >= 2:
        return "high"
    if total == 1:
        return "medium"
    return "normal"


def _recommendation(
    summary: dict[str, Any],
    image_summary: dict[str, Any] | None,
    threshold_config: dict[str, Any],
) -> str:
    actions: list[str] = []
    field_messages = {
        "soil_moisture_percent": "verificar irrigacao do talhao",
        "air_temperature_celsius": "acompanhar temperatura e exposicao solar",
        "air_humidity_percent": "monitorar umidade do ar",
    }
    for field_name, message in field_messages.items():
        if _is_outside_expected_range(summary, threshold_config, field_name):
            actions.append(message)
    if image_summary and image_summary.get("sick_images", 0) > 0:
        actions.append("inspecionar folhas com classificacao Sick")

    if not actions:
        return "Condicoes dentro da faixa esperada. Manter monitoramento."
    return "Recomendacao: " + "; ".join(actions) + "."


def _expected_range_risk_signals(summary: dict[str, Any], threshold_config: dict[str, Any]) -> list[bool]:
    return [
        _is_outside_expected_range(summary, threshold_config, field_name)
        for field_name, field_config in threshold_config["sensor_fields"].items()
        if field_config.get("expected_range") is not None
    ]


def _is_outside_expected_range(
    summary: dict[str, Any],
    threshold_config: dict[str, Any],
    field_name: str,
) -> bool:
    expected_range = threshold_config["sensor_fields"].get(field_name, {}).get("expected_range")
    if expected_range is None:
        return False

    summary_key = _summary_key_for_field(field_name)
    value = summary.get(summary_key)
    if value is None:
        return False
    return value < expected_range["minimum"] or value > expected_range["maximum"]


def _summary_key_for_field(field_name: str) -> str:
    return f"average_{field_name}"


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
