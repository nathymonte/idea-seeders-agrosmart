from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.data_lake import (
    ensure_data_lake,
    load_seen_event_ids,
    preserve_historical_csv,
    recompute_field_status,
    save_rejected_sensor_event,
    save_trusted_sensor_reading,
)
from services.sensor_validator import validate_sensor_event


def ingest_csv(csv_path: Path) -> tuple[int, int]:
    ensure_data_lake()
    preserve_historical_csv(csv_path)
    seen_event_ids = load_seen_event_ids()
    accepted = 0
    rejected = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            normalized, errors = validate_sensor_event(row, source="historical_csv", seen_event_ids=seen_event_ids)
            if normalized:
                save_trusted_sensor_reading(normalized)
                seen_event_ids.add(normalized["event_id"])
                accepted += 1
            else:
                save_rejected_sensor_event(row, errors, source="historical_csv")
                rejected += 1

    recompute_field_status()
    return accepted, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingere o CSV historico no Data Lake AgroSmart.")
    parser.add_argument("--csv", default="samples/sensor_history.csv", help="Caminho do arquivo CSV historico.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    accepted_count, rejected_count = ingest_csv(Path(args.csv))
    print(f"Ingestao concluida. Trusted: {accepted_count}. Rejected: {rejected_count}.")
