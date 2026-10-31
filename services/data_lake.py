from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from services.paths import DATA_LAKE_PATH, IMAGE_CSV_PATH, validate_field_id
from services.threshold_config import get_threshold_config

LAYERS = ("raw", "trusted", "refined", "rejected")


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_datetime(value: str | datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if not value:
        return utc_now()

    normalized = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


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


def read_dataset(
    layer: str, dataset: str, base_path: Path | str = DATA_LAKE_PATH,
    start: datetime | None = None, end: datetime | None = None,
    field_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_jsonl_files(layer, dataset, base_path):
        if start or end:
            parts = dict(p.split("=", 1) for p in path.parts if "=" in p)
            partition = datetime(int(parts["year"]), int(parts["month"]), int(parts["day"]), tzinfo=UTC)
            if start and partition.date() < start.date():
                continue
            if end and partition.date() > end.date():
                continue
        for record in read_jsonl(path):
            if field_ids is not None and record.get("field_id") not in field_ids:
                continue
            if start or end:
                timestamp = parse_datetime(record.get("event_timestamp"))
                if start and timestamp < start or end and timestamp > end:
                    continue
            records.append(record)
    return records


def write_json(path: Path | str, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
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


def load_image_analysis_summary(
    output_csv: Path | str = IMAGE_CSV_PATH, field_id: str | None = None,
    start: datetime | None = None, end: datetime | None = None,
) -> dict[str, Any] | None:
    path = Path(output_csv)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return None

    if field_id is not None:
        rows = [row for row in rows if row.get("field_id") == field_id]
    dated = []
    for row in rows:
        value = row.get("data_analise")
        if not value:
            continue
        try:
            timestamp = parse_datetime(value)
        except ValueError:
            continue
        if start and timestamp < start or end and timestamp > end:
            continue
        dated.append((timestamp, row))
    if not dated:
        return None
    dated.sort(key=lambda item: item[0])
    rows = [row for _, row in dated]
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
    output_csv: Path | str = IMAGE_CSV_PATH,
    threshold_config: dict[str, Any] | None = None,
    field_ids: set[str] | None = None,
    window_hours: float = 24,
    reference_time: datetime | None = None,
    field_reference_times: dict[str, datetime] | None = None,
) -> list[Path]:
    if window_hours <= 0:
        raise ValueError("window_hours deve ser positivo")
    config = threshold_config or get_threshold_config()
    reference = reference_time or utc_now()
    start = reference - timedelta(hours=window_hours)
    if field_reference_times:
        records = []
        for field, observed_at in field_reference_times.items():
            previous_path = Path(base_path) / "refined/field_status" / f"{validate_field_id(field)}.json"
            if previous_path.exists():
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                previous_time = previous.get("sensor_summary", {}).get("last_reading_at")
                if previous_time:
                    observed_at = max(observed_at, parse_datetime(previous_time))
            records.extend(read_dataset("trusted", "sensor_readings", base_path,
                                        observed_at - timedelta(hours=window_hours), observed_at, {field}))
    else:
        records = read_dataset("trusted", "sensor_readings", base_path, field_ids=field_ids,
                               start=start if reference_time else None, end=reference if reference_time else None)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("field_id", "TALHAO-01"))].append(record)

    written: list[Path] = []
    for field_id, field_records in grouped.items():
        validate_field_id(field_id)
        field_records.sort(key=lambda item: parse_datetime(item["event_timestamp"]))
        latest_time = parse_datetime(field_records[-1]["event_timestamp"])
        # Historical imports use the latest observation as their reference, not today's date.
        period_end = reference if reference_time else latest_time
        period_start = period_end - timedelta(hours=window_hours)
        field_records = [r for r in field_records if period_start <= parse_datetime(r["event_timestamp"]) <= period_end]
        if not field_records:
            continue
        summary = _build_sensor_summary(field_records)
        image_summary = load_image_analysis_summary(output_csv, field_id, period_start, period_end)
        decision_summary = {f"average_{key}": value for key, value in field_records[-1].items()
                            if key in config["sensor_fields"]}
        reasons = decision_reasons(decision_summary, image_summary, config)
        payload = {
            "field_id": field_id,
            "crop": "tomato",
            "calculation_period": {
                "start": field_records[0]["event_timestamp"],
                "end": field_records[-1]["event_timestamp"],
            },
            "sensor_summary": summary,
            "latest_reading": field_records[-1],
            "latest_image_analysis": image_summary,
            "attention_level": _attention_level(decision_summary, image_summary, config),
            "recommendation": _recommendation(decision_summary, image_summary, config),
            "decision_reasons": reasons,
            "decision_basis": "latest_reading",
            "window_hours": window_hours,
            "updated_at": utc_now().isoformat().replace("+00:00", "Z"),
        }
        written.append(write_json(Path(base_path) / "refined" / "field_status" / f"{field_id}.json", payload))
    return written


def decision_reasons(summary: dict, image_summary: dict | None, config: dict) -> list[str]:
    reasons = []
    labels = {"soil_moisture_percent": "Umidade do solo (%)", "air_temperature_celsius": "Temperatura (C)",
              "air_humidity_percent": "Umidade do ar (%)", "luminosity_lux": "Luminosidade (lux)"}
    for field, options in config["sensor_fields"].items():
        if _is_outside_expected_range(summary, config, field):
            limits = options["expected_range"]
            value = summary['average_' + field]
            reasons.append(f"{labels.get(field, field)}: {value} fora de {limits['minimum']} a {limits['maximum']}")
    if image_summary and image_summary.get("sick_images", 0):
        reasons.append(f"{image_summary['sick_images']} imagem(ns) Sick no talhao e periodo")
    return reasons


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
