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
    path = build_partition_path("raw", dataset, reference, "readings.jsonl", base_path)
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
    path = build_partition_path(
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
) -> list[Path]:
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
            "attention_level": _attention_level(summary, image_summary),
            "recommendation": _recommendation(summary, image_summary),
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


def _attention_level(summary: dict[str, Any], image_summary: dict[str, Any] | None) -> str:
    risk_signals = [
        summary["average_soil_moisture_percent"] < 25,
        summary["average_air_temperature_celsius"] > 32,
        summary["average_air_humidity_percent"] < 50,
        bool(image_summary and image_summary.get("sick_images", 0) > 0),
    ]
    total = sum(1 for signal in risk_signals if signal)
    if total >= 2:
        return "high"
    if total == 1:
        return "medium"
    return "normal"


def _recommendation(summary: dict[str, Any], image_summary: dict[str, Any] | None) -> str:
    actions: list[str] = []
    if summary["average_soil_moisture_percent"] < 25:
        actions.append("verificar irrigacao do talhao")
    if summary["average_air_temperature_celsius"] > 32:
        actions.append("acompanhar temperatura e exposicao solar")
    if summary["average_air_humidity_percent"] < 50:
        actions.append("monitorar umidade do ar")
    if image_summary and image_summary.get("sick_images", 0) > 0:
        actions.append("inspecionar folhas com classificacao Sick")

    if not actions:
        return "Condicoes dentro da faixa esperada. Manter monitoramento."
    return "Recomendacao: " + "; ".join(actions) + "."


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
