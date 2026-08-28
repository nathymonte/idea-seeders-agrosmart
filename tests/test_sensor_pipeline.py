import tempfile
import unittest
from pathlib import Path

from services.data_lake import read_dataset, recompute_field_status, save_trusted_sensor_reading
from services.sensor_validator import validate_sensor_event


class SensorPipelineTest(unittest.TestCase):
    def test_valid_event_is_normalized(self):
        event = {
            "event_id": "evt-001",
            "sensor_id": "SENSOR-001",
            "field_id": "TALHAO-01",
            "event_timestamp": "2026-08-24T08:00:00Z",
            "soil_moisture_percent": 36.5,
            "air_temperature_celsius": 23.8,
            "air_humidity_percent": 71.2,
            "luminosity_lux": 12500,
        }

        normalized, errors = validate_sensor_event(event)

        self.assertEqual(errors, [])
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["source"], "kafka")
        self.assertEqual(normalized["schema_version"], "1.0")

    def test_invalid_range_is_rejected(self):
        event = {
            "field_id": "TALHAO-01",
            "event_timestamp": "2026-08-24T08:00:00Z",
            "soil_moisture_percent": 125,
            "air_temperature_celsius": 23.8,
            "air_humidity_percent": 71.2,
            "luminosity_lux": 12500,
        }

        normalized, errors = validate_sensor_event(event)

        self.assertIsNone(normalized)
        self.assertTrue(any("soil_moisture_percent fora da faixa" in error for error in errors))

    def test_duplicate_event_is_rejected(self):
        event = {
            "event_id": "evt-001",
            "field_id": "TALHAO-01",
            "event_timestamp": "2026-08-24T08:00:00Z",
            "soil_moisture_percent": 36.5,
            "air_temperature_celsius": 23.8,
            "air_humidity_percent": 71.2,
            "luminosity_lux": 12500,
        }

        normalized, errors = validate_sensor_event(event, seen_event_ids={"evt-001"})

        self.assertIsNone(normalized)
        self.assertIn("event_id duplicado: evt-001", errors)

    def test_trusted_record_generates_refined_status(self):
        event = {
            "event_id": "evt-001",
            "sensor_id": "SENSOR-001",
            "field_id": "TALHAO-01",
            "event_timestamp": "2026-08-24T08:00:00Z",
            "soil_moisture_percent": 20.0,
            "air_temperature_celsius": 34.0,
            "air_humidity_percent": 45.0,
            "luminosity_lux": 12500,
            "source": "kafka",
            "schema_version": "1.0",
            "ingested_at": "2026-08-24T08:00:02Z",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_trusted_sensor_reading(event, base_path=tmp_dir)
            records = read_dataset("trusted", "sensor_readings", base_path=tmp_dir)
            written = recompute_field_status(base_path=tmp_dir, output_csv=Path(tmp_dir) / "missing.csv")

            self.assertEqual(len(records), 1)
            self.assertEqual(len(written), 1)
            self.assertTrue((Path(tmp_dir) / "refined" / "field_status" / "TALHAO-01.json").exists())


if __name__ == "__main__":
    unittest.main()
