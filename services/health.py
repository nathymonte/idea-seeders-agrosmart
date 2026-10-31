"""Indicadores de saude da plataforma, usados pela aba "Saude do sistema" e pelo healthcheck."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from services.data_lake import iter_jsonl_files
from services.observability import health_state, read_status
from services.paths import DATA_LAKE_PATH

LAYERS = {"raw": ("raw", "sensors"), "trusted": ("trusted", "sensor_readings"),
          "rejected": ("rejected", "sensor_readings")}
# Componentes executados sob demanda (lote), sem processo continuo para enviar heartbeat
ON_DEMAND = {"image-processing"}
COMPONENT_LABELS = {"producer": "Simulador IoT (producer)", "consumer": "Consumer Kafka",
                    "dashboard": "Dashboard", "image-processing": "Pipeline de imagens"}


def count_records(layer: str, dataset: str, base_path: Path | str = DATA_LAKE_PATH) -> tuple[int, float | None]:
    """Quantidade de registros e horario (epoch) da ultima escrita em uma camada."""
    total, latest = 0, None
    for path in iter_jsonl_files(layer, dataset, base_path):
        with path.open("rb") as file:
            total += sum(1 for line in file if line.strip())
        modified = path.stat().st_mtime
        latest = modified if latest is None else max(latest, modified)
    return total, latest


def refined_status(base_path: Path | str = DATA_LAKE_PATH) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((Path(base_path) / "refined/field_status").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows.append({"field_id": path.stem, "attention_level": "arquivo invalido"})
            continue
        rows.append({"field_id": data.get("field_id", path.stem),
                     "attention_level": data.get("attention_level"),
                     "last_reading_at": data.get("sensor_summary", {}).get("last_reading_at"),
                     "updated_at": data.get("updated_at"),
                     "readings_24h": data.get("sensor_summary", {}).get("readings_count")})
    return rows


def system_overview(base_path: Path | str = DATA_LAKE_PATH, now: datetime | None = None,
                    status_path: Path | str | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    stale = float(os.getenv("AGROSMART_HEALTH_STALE_SECONDS", "30"))
    offline = float(os.getenv("AGROSMART_HEALTH_OFFLINE_SECONDS", "120"))
    components = {}
    for name, label in COMPONENT_LABELS.items():
        status = read_status(name, status_path)
        state = health_state(status, now, stale, offline)
        if name in ON_DEMAND and state == "sem dados":
            state = "sob demanda"
        components[name] = {"label": label, "state": state, "status": status or {}, "on_demand": name in ON_DEMAND}
    layers = {}
    for label, (layer, dataset) in LAYERS.items():
        total, latest = count_records(layer, dataset, base_path)
        layers[label] = {"records": total,
                         "last_write_age_s": None if latest is None else max(0.0, now.timestamp() - latest)}
    received = layers["trusted"]["records"] + layers["rejected"]["records"]
    layers["quality_rate"] = round(layers["trusted"]["records"] / received, 4) if received else None
    return {"checked_at": now, "components": components, "data_lake": layers,
            "fields": refined_status(base_path)}


def overall_state(overview: dict[str, Any]) -> str:
    """Resumo: critico se o consumer estiver offline; atencao se algo estiver atrasado."""
    states = {name: item["state"] for name, item in overview["components"].items()}
    if states.get("consumer") == "offline":
        return "critico"
    if "atrasado" in states.values() or states.get("consumer") in {"sem dados"}:
        return "atencao"
    return "saudavel"
