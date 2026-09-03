from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid5, NAMESPACE_URL

from services.threshold_config import get_threshold_config

REQUIRED_FIELDS = (
    "field_id",
    "event_timestamp",
    "soil_moisture_percent",
    "air_temperature_celsius",
    "air_humidity_percent",
    "luminosity_lux",
)


def validate_sensor_event(
    event: dict[str, Any],
    source: str = "kafka",
    seen_event_ids: set[str] | None = None,
    threshold_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    config = threshold_config or get_threshold_config()
    normalized = _normalize_event(event, source, config)
    errors = _required_field_errors(normalized)
    errors.extend(_timestamp_errors(event))
    errors.extend(_range_errors(normalized, config))

    seen = seen_event_ids or set()
    event_id = normalized.get("event_id")
    if event_id and event_id in seen:
        errors.append(f"event_id duplicado: {event_id}")

    if errors:
        return None, errors
    return normalized, []


def _normalize_event(event: dict[str, Any], source: str, threshold_config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(event)

    if "event_timestamp" not in normalized and "timestamp" in normalized:
        normalized["event_timestamp"] = normalized["timestamp"]

    timestamp = _normalize_timestamp(normalized.get("event_timestamp"))
    if timestamp:
        normalized["event_timestamp"] = timestamp

    normalized.setdefault("sensor_id", "SENSOR-001")
    normalized.setdefault("field_id", "TALHAO-01")
    normalized["source"] = source
    normalized["schema_version"] = "1.0"
    normalized["ingested_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for field in threshold_config["sensor_fields"]:
        if field in normalized:
            normalized[field] = _to_float(normalized[field])

    if not normalized.get("event_id"):
        stable_key = "|".join(
            str(normalized.get(field, ""))
            for field in ("sensor_id", "field_id", "event_timestamp", "source")
        )
        normalized["event_id"] = f"evt-{uuid5(NAMESPACE_URL, stable_key)}"

    return normalized


def _required_field_errors(event: dict[str, Any]) -> list[str]:
    return [f"campo obrigatorio ausente: {field}" for field in REQUIRED_FIELDS if event.get(field) in (None, "")]


def _timestamp_errors(event: dict[str, Any]) -> list[str]:
    value = event.get("event_timestamp") or event.get("timestamp")
    if not value:
        return []
    if _normalize_timestamp(value) is None:
        return [f"timestamp invalido: {value}"]
    return []


def _range_errors(event: dict[str, Any], threshold_config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field, field_config in threshold_config["sensor_fields"].items():
        value = event.get(field)
        if value is None:
            continue
        limits = field_config["allowed_range"]
        if not isinstance(value, (int, float)):
            errors.append(f"campo numerico invalido: {field}")
            continue
        if value < limits["minimum"] or value > limits["maximum"]:
            errors.append(
                f"{field} fora da faixa permitida "
                f"({limits['minimum']} a {limits['maximum']}): {value}"
            )
    return errors


def _normalize_timestamp(value: Any) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_float(value: Any) -> float | Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value
