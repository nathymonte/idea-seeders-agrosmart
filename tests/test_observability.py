import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_sensor_pipeline import TEST_CONFIG, sensor_event

from services.health import overall_state, system_overview
from services.observability import get_logger, health_state, log_activity, read_status, tail_logs, write_heartbeat
from src.streaming import consumer as consumer_module
from src.streaming.consumer import ConsumerMetrics, consume_messages
from src.streaming.producer import corrupt_event, pick_scenario


class ObservabilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.dict(os.environ, {"AGROSMART_LOG_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        level = consumer_module.logger.level
        consumer_module.logger.setLevel("INFO")
        self.addCleanup(consumer_module.logger.setLevel, level)

    def consumer_log_entries(self):
        files = [Path(h.baseFilename) for h in consumer_module.logger.handlers if hasattr(h, "baseFilename")]
        entries = []
        for handler in consumer_module.logger.handlers:
            handler.flush()
        for path in files:
            entries.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        return entries

    def test_heartbeat_roundtrip_and_states(self):
        write_heartbeat("consumer", state="running", received=3)
        status = read_status("consumer")
        self.assertEqual(status["received"], 3)
        now = datetime.now(UTC)
        self.assertEqual(health_state(status, now), "online")
        self.assertEqual(health_state(status, now + timedelta(seconds=60)), "atrasado")
        self.assertEqual(health_state(status, now + timedelta(minutes=10)), "offline")
        self.assertEqual(health_state(None), "sem dados")
        self.assertEqual(health_state({**status, "state": "finished"}, now + timedelta(days=1)), "finalizado")

    def test_json_logs_carry_process_activity(self):
        logger = get_logger("test-component", self.tmp.name)
        logger.setLevel("INFO")
        with redirect_stdout(io.StringIO()):
            log_activity(logger, "Validada", "evt-1", field_id="TALHAO-01")
        for handler in logger.handlers:
            handler.flush()
        line = (Path(self.tmp.name) / "test-component.log").read_text(encoding="utf-8").splitlines()[-1]
        entry = json.loads(line)
        self.assertEqual((entry["activity"], entry["case_id"], entry["field_id"]), ("Validada", "evt-1", "TALHAO-01"))
        self.assertEqual(tail_logs(10, self.tmp.name, ("test-component",))[-1]["activity"], "Validada")

    def test_consumer_metrics_count_outcomes(self):
        metrics = ConsumerMetrics()
        duplicate = sensor_event("dup")
        invalid = sensor_event("bad")
        invalid["soil_moisture_percent"] = 150
        messages = [SimpleNamespace(value=sensor_event("ok")), SimpleNamespace(value=duplicate),
                    SimpleNamespace(value=duplicate), SimpleNamespace(value=invalid),
                    SimpleNamespace(value=b"{not json")]
        lake = Path(self.tmp.name) / "lake"
        with redirect_stdout(io.StringIO()):
            consume_messages(messages, base_path=lake, threshold_config=TEST_CONFIG,
                             output_csv=lake / "absent.csv", metrics=metrics, max_messages=5)
        snapshot = metrics.snapshot("finished")
        self.assertEqual((snapshot["received"], snapshot["accepted"], snapshot["duplicates"], snapshot["rejected"]),
                         (5, 2, 1, 2))
        self.assertGreaterEqual(snapshot["batches_refined"], 1)
        self.assertEqual(read_status("consumer")["state"], "finished")
        activities = [e.get("activity") for e in self.consumer_log_entries()]
        for expected in ("Recebida pelo consumer", "Gravada na trusted", "Descartada (duplicada)",
                         "Rejeitada (validacao)", "Rejeitada (payload invalido)", "Decisao gerada"):
            self.assertIn(expected, activities)

    def test_system_overview_reports_layers_and_overall_state(self):
        lake = Path(self.tmp.name) / "lake"
        with redirect_stdout(io.StringIO()):
            consume_messages([SimpleNamespace(value=sensor_event("a"))], base_path=lake,
                             threshold_config=TEST_CONFIG, output_csv=lake / "absent.csv", max_messages=1)
        overview = system_overview(lake)
        self.assertEqual(overview["data_lake"]["trusted"]["records"], 1)
        self.assertEqual(overview["data_lake"]["quality_rate"], 1.0)
        self.assertEqual(overview["components"]["consumer"]["state"], "finalizado")
        self.assertEqual(overview["components"]["image-processing"]["state"], "sob demanda")
        self.assertIn(overall_state(overview), {"saudavel", "atencao"})

    def test_producer_fault_injection(self):
        import random
        rng = random.Random(1)
        broken = corrupt_event({"soil_moisture_percent": 40, "air_humidity_percent": 60}, rng)
        self.assertEqual(broken["simulation_fault"], "out_of_range")
        self.assertTrue(broken["soil_moisture_percent"] > 100 or broken["air_humidity_percent"] > 100)
        self.assertEqual(pick_scenario("dry", rng), "dry")
        self.assertIn(pick_scenario("mixed", rng), {"normal", "random", "dry", "hot", "high"})


if __name__ == "__main__":
    unittest.main()


class HealthcheckAndReportTest(unittest.TestCase):
    def test_healthcheck_heartbeat_and_url_guard(self):
        from scripts.healthcheck import check_heartbeat, check_url
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGROSMART_LOG_DIR": tmp}):
            self.assertFalse(check_heartbeat("consumer", 60)[0])
            write_heartbeat("consumer", state="running")
            self.assertTrue(check_heartbeat("consumer", 60)[0])
            write_heartbeat("consumer", state="error")
            self.assertFalse(check_heartbeat("consumer", 60)[0])
        self.assertFalse(check_url("http://example.com/health")[0])
        self.assertFalse(check_url("http://localhost:1/_stcore/health")[0])

    def test_automation_report_uses_decision_engine(self):
        from src.automation import generate_report
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGROSMART_LOG_DIR": tmp}):
            lake = Path(tmp) / "lake"
            with redirect_stdout(io.StringIO()):
                consume_messages([SimpleNamespace(value=sensor_event("r1"))], base_path=lake,
                                 threshold_config=TEST_CONFIG, output_csv=lake / "absent.csv", max_messages=1)
            report = generate_report(Path(tmp) / "relatorio.md", lake, lake / "absent.csv")
            text = report.read_text(encoding="utf-8")
        self.assertIn("TALHAO-01", text)
        self.assertIn("Nivel de atencao", text)
