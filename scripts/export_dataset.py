import argparse
import csv
from pathlib import Path

from services.data_lake import read_dataset, parse_datetime
from services.paths import DATA_LAKE_PATH


def export_dataset(destination: Path, base_path=DATA_LAKE_PATH, start=None, end=None, fields=None) -> int:
    records = read_dataset("trusted", "sensor_readings", base_path, start, end, fields)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in records for key in row}) or ["event_id", "field_id", "event_timestamp"]
    with destination.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta leituras trusted por talhao e periodo.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--fields", nargs="+")
    args = parser.parse_args()
    count = export_dataset(args.out, start=parse_datetime(args.start) if args.start else None,
                           end=parse_datetime(args.end) if args.end else None,
                           fields=set(args.fields) if args.fields else None)
    print(f"Exportadas {count} leituras para {args.out}")
