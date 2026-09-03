import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.historical_ingestion import ingest_csv
from services.data_lake import (
    read_dataset,
    recompute_field_status,
    save_trusted_sensor_reading,
)
from services.sensor_validator import validate_sensor_event
from services.threshold_config import clear_threshold_config_cache, get_threshold_config
from src.streaming.consumer import consume_messages


TEST_CONFIG = {
    "sensor_fields": {
        "soil_moisture_percent": {
            "allowed_range": {"minimum": 0, "maximum": 100},
            "expected_range": {"minimum": 40, "maximum": 70},
        },
        "air_temperature_celsius": {
            "allowed_range": {"minimum": -20, "maximum": 60},
            "expected_range": {"minimum": 15, "maximum": 30},
        },
        "air_humidity_percent": {
            "allowed_range": {"minimum": 0, "maximum": 100},
            "expected_range": {"minimum": 50, "maximum": 85},
        },
        "luminosity_lux": {
            "allowed_range": {"minimum": 0, "maximum": 120000},
        },
    }
}


def sensor_event(
    event_id: str,
    field_id: str = "TALHAO-01",
    timestamp: str = "2026-08-24T08:00:00Z",
    soil: float = 50.0,
    temp: float = 24.0,
    humidity: float = 70.0,
) -> dict:
    return {
        "event_id": event_id,
        "sensor_id": "SENSOR-001",
        "field_id": field_id,
        "event_timestamp": timestamp,
        "soil_moisture_percent": soil,
        "air_temperature_celsius": temp,
        "air_humidity_percent": humidity,
        "luminosity_lux": 12500,
    }


def trusted_event(**kwargs) -> dict:
    event = sensor_event(**kwargs)
    event["source"] = "kafka"
    event["schema_version"] = "1.0"
    event["ingested_at"] = "2026-08-24T08:00:02Z"
    return event


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class SensorPipelineTest(unittest.TestCase):
    def test_valid_event_is_normalized(self):
        normalized, errors = validate_sensor_event(sensor_event("evt-001"), threshold_config=TEST_CONFIG)

        self.assertEqual(errors, [])
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["source"], "kafka")
        self.assertEqual(normalized["schema_version"], "1.0")

    def test_invalid_range_is_rejected(self):
        event = sensor_event("evt-invalid", soil=125)

        normalized, errors = validate_sensor_event(event, threshold_config=TEST_CONFIG)

        self.assertIsNone(normalized)
        self.assertTrue(any("soil_moisture_percent fora da faixa" in error for error in errors))

    def test_duplicate_event_is_rejected(self):
        normalized, errors = validate_sensor_event(
            sensor_event("evt-001"),
            seen_event_ids={"evt-001"},
            threshold_config=TEST_CONFIG,
        )

        self.assertIsNone(normalized)
        self.assertIn("event_id duplicado: evt-001", errors)

    def test_trusted_record_generates_refined_status(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_trusted_sensor_reading(trusted_event(event_id="evt-001", soil=20.0, temp=34.0), base_path=tmp_dir)
            records = read_dataset("trusted", "sensor_readings", base_path=tmp_dir)
            written = recompute_field_status(
                base_path=tmp_dir,
                output_csv=Path(tmp_dir) / "missing.csv",
                threshold_config=TEST_CONFIG,
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(len(written), 1)
            self.assertTrue((Path(tmp_dir) / "refined" / "field_status" / "TALHAO-01.json").exists())

    def test_invalid_timestamp_is_preserved_in_raw_and_rejected(self):
        invalid_event = sensor_event("evt-bad-time", timestamp="data-invalida")
        valid_event = sensor_event("evt-after-bad-time", timestamp="2026-08-24T09:00:00Z")

        with tempfile.TemporaryDirectory() as tmp_dir:
            with redirect_stdout(io.StringIO()):
                consumed = consume_messages(
                    [SimpleNamespace(value=invalid_event), SimpleNamespace(value=valid_event)],
                    max_messages=2,
                    refine_every=10,
                    base_path=tmp_dir,
                    output_csv=Path(tmp_dir) / "missing.csv",
                    threshold_config=TEST_CONFIG,
                )
            raw_records = read_dataset("raw", "sensors", tmp_dir)
            rejected_records = read_dataset("rejected", "sensor_readings", tmp_dir)
            trusted_records = read_dataset("trusted", "sensor_readings", tmp_dir)

            self.assertEqual(consumed, 2)
            self.assertEqual(len(raw_records), 2)
            self.assertEqual(len(rejected_records), 1)
            self.assertEqual(len(trusted_records), 1)
            raw_invalid = next(record for record in raw_records if record["event_id"] == "evt-bad-time")
            rejected_invalid = next(record for record in rejected_records if record["event"]["event_id"] == "evt-bad-time")
            self.assertEqual(raw_invalid["event_timestamp"], "data-invalida")
            self.assertEqual(rejected_invalid["event"]["event_timestamp"], "data-invalida")
            self.assertTrue(any("timestamp invalido" in error for error in rejected_invalid["errors"]))

    def test_historical_ingestion_writes_all_layers_once(self):
        rows = [
            sensor_event("hist-001"),
            sensor_event("hist-002", timestamp="2026-08-24T09:00:00Z"),
            sensor_event("hist-invalid", soil=125),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "history.csv"
            write_csv(csv_path, rows)

            accepted, rejected = ingest_csv(
                csv_path,
                base_path=Path(tmp_dir) / "lake",
                output_csv=Path(tmp_dir) / "missing.csv",
                threshold_config=TEST_CONFIG,
            )

            lake = Path(tmp_dir) / "lake"
            self.assertEqual((accepted, rejected), (2, 1))
            self.assertTrue(list((lake / "raw" / "historical").rglob("*history.csv")))
            self.assertEqual(len(read_dataset("trusted", "sensor_readings", lake)), 2)
            self.assertEqual(len(read_dataset("rejected", "sensor_readings", lake)), 1)
            self.assertTrue((lake / "refined" / "field_status" / "TALHAO-01.json").exists())

    def test_reprocessing_same_csv_rejects_duplicates_without_trusted_duplication(self):
        rows = [sensor_event("hist-001"), sensor_event("hist-002", timestamp="2026-08-24T09:00:00Z")]

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "history.csv"
            lake = Path(tmp_dir) / "lake"
            write_csv(csv_path, rows)

            self.assertEqual(
                ingest_csv(csv_path, base_path=lake, output_csv=Path(tmp_dir) / "missing.csv", threshold_config=TEST_CONFIG),
                (2, 0),
            )
            self.assertEqual(
                ingest_csv(csv_path, base_path=lake, output_csv=Path(tmp_dir) / "missing.csv", threshold_config=TEST_CONFIG),
                (0, 2),
            )

            trusted = read_dataset("trusted", "sensor_readings", lake)
            rejected = read_dataset("rejected", "sensor_readings", lake)
            self.assertEqual(len(trusted), 2)
            self.assertTrue(all("event_id duplicado" in item["errors"][0] for item in rejected))

    def test_multiple_fields_generate_separate_refined_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_trusted_sensor_reading(trusted_event(event_id="evt-001", field_id="TALHAO-01"), base_path=tmp_dir)
            save_trusted_sensor_reading(trusted_event(event_id="evt-002", field_id="TALHAO-02"), base_path=tmp_dir)

            recompute_field_status(
                base_path=tmp_dir,
                output_csv=Path(tmp_dir) / "missing.csv",
                threshold_config=TEST_CONFIG,
            )

            self.assertTrue((Path(tmp_dir) / "refined" / "field_status" / "TALHAO-01.json").exists())
            self.assertTrue((Path(tmp_dir) / "refined" / "field_status" / "TALHAO-02.json").exists())

    def test_attention_levels_follow_yaml_expected_ranges(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lake = Path(tmp_dir) / "lake"
            scenarios = [
                trusted_event(event_id="normal", field_id="NORMAL", soil=50, temp=24, humidity=70),
                trusted_event(event_id="medium", field_id="MEDIUM", soil=35, temp=24, humidity=70),
                trusted_event(event_id="high", field_id="HIGH", soil=35, temp=35, humidity=70),
            ]
            for event in scenarios:
                save_trusted_sensor_reading(event, base_path=lake)

            recompute_field_status(base_path=lake, output_csv=Path(tmp_dir) / "missing.csv", threshold_config=TEST_CONFIG)

            statuses = {
                path.stem: json.loads(path.read_text(encoding="utf-8"))["attention_level"]
                for path in (lake / "refined" / "field_status").glob("*.json")
            }
            self.assertEqual(statuses["NORMAL"], "normal")
            self.assertEqual(statuses["MEDIUM"], "medium")
            self.assertEqual(statuses["HIGH"], "high")

    def test_threshold_config_is_loaded_from_yaml(self):
        yaml_text = """
sensor_fields:
  soil_moisture_percent:
    allowed_range:
      minimum: 10
      maximum: 20
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "thresholds.yaml"
            config_path.write_text(yaml_text, encoding="utf-8")
            clear_threshold_config_cache()

            config = get_threshold_config(config_path)

            self.assertEqual(config["sensor_fields"]["soil_moisture_percent"]["allowed_range"]["minimum"], 10)

    def test_micro_batch_refines_on_threshold_and_pending_batch_only_for_accepted_events(self):
        messages = [
            SimpleNamespace(value=sensor_event("evt-001")),
            SimpleNamespace(value=sensor_event("evt-rejected", soil=125)),
            SimpleNamespace(value=sensor_event("evt-002", timestamp="2026-08-24T09:00:00Z")),
            SimpleNamespace(value=sensor_event("evt-003", timestamp="2026-08-24T10:00:00Z")),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.streaming.consumer.recompute_field_status") as recompute:
                with redirect_stdout(io.StringIO()):
                    consumed = consume_messages(
                        messages,
                        max_messages=4,
                        refine_every=2,
                        base_path=tmp_dir,
                        output_csv=Path(tmp_dir) / "missing.csv",
                        threshold_config=TEST_CONFIG,
                    )

            self.assertEqual(consumed, 4)
            self.assertEqual(recompute.call_count, 2)
            self.assertEqual(len(read_dataset("trusted", "sensor_readings", tmp_dir)), 3)
            self.assertEqual(len(read_dataset("rejected", "sensor_readings", tmp_dir)), 1)


if __name__ == "__main__":
    unittest.main()
