"""Generate isolated demonstration data using the real sensor pipeline."""
import argparse
import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from services.data_lake import read_dataset, write_json
from services.paths import PROJECT_ROOT
from src.prepare_data import prepare_rows, CSV_FIELDS
from src.streaming.consumer import consume_messages
from src.streaming.producer import build_event


def build_demo(destination: Path, reference_time: datetime | None = None) -> dict:
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use uma pasta vazia para a demonstracao; dados existentes nao serao substituidos.")
    destination.mkdir(parents=True, exist_ok=True)
    now = reference_time or datetime.now(timezone.utc)
    lake = destination / "data_lake"
    output = destination / "output"
    output.mkdir()
    rng = random.Random(42)
    messages = []
    for index in range(24):
        timestamp = now - timedelta(minutes=(23 - index) * 10)
        for field_id, scenario in [("TALHAO-01", "normal"), ("TALHAO-02", "dry"), ("TALHAO-03", "high")]:
            event = build_event(field_id, "normal" if index < 18 else scenario, rng, timestamp)
            event["event_id"] = f"demo-{field_id}-{index:03}"
            event["luminosity_lux"] *= .7 + index / 50
            messages.append(SimpleNamespace(value=event))
    invalid = build_event(timestamp=now)
    invalid["event_id"] = "demo-invalid"
    invalid["soil_moisture_percent"] = "valor-invalido"
    messages.append(SimpleNamespace(value=invalid))
    source_predictions = json.loads((PROJECT_ROOT / "output/results.json").read_text(encoding="utf-8"))
    metadata = {}
    for index, item in enumerate(source_predictions["results"]):
        # Predictions are reused; only their demo association and time are simulated.
        metadata[item.get("image_path", item["image_name"])] = {
            "field_id": "TALHAO-03" if item["predicted_label"] == "Sick" else "TALHAO-01",
            "captured_at": (now - timedelta(minutes=index)).isoformat(),
            "metadata_source": "simulation",
        }
    image_csv = output / "classificacoes.csv"
    with image_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(prepare_rows(source_predictions, metadata))
    consumed = consume_messages(messages, base_path=lake, output_csv=image_csv, refine_every=10)
    records = read_dataset("trusted", "sensor_readings", lake)
    with (output / "leituras_processadas.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in records for key in row}))
        writer.writeheader()
        writer.writerows(records)
    summary = {"consumed": consumed, "accepted": len(records),
               "rejected": len(read_dataset("rejected", "sensor_readings", lake)),
               "reference_time": now.isoformat(), "simulation": True}
    write_json(destination / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstracao isolada, sem broker Kafka.")
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / ".demo")
    args = parser.parse_args()
    print(json.dumps(build_demo(args.destination), indent=2))
