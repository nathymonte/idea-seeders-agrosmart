import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.build_event_log import enrich_cases, read_activity_logs

try:
    from scripts import process_mining
except ImportError:  # matplotlib e opcional (requirements-analysis.txt)
    process_mining = None

HAPPY = ["Leitura gerada", "Publicada no Kafka", "Recebida pelo consumer", "Gravada na raw", "Validada",
         "Gravada na trusted", "Consolidada na refined", "Decisao gerada"]


def write_logs(directory: Path) -> None:
    start = datetime(2026, 10, 1, tzinfo=UTC)
    cases = {"a": HAPPY, "b": [*HAPPY, "Alerta emitido"],
             "c": ["Leitura gerada", "Publicada no Kafka", "Recebida pelo consumer", "Gravada na raw",
                   "Rejeitada (validacao)"]}
    producer, consumer = [], []
    for offset, (case, activities) in enumerate(cases.items()):
        moment = start + timedelta(seconds=offset * 10)
        for activity in activities:
            # A espera do micro-lote (3 s) e o gargalo esperado; as demais etapas levam 0,1 s
            moment += timedelta(seconds=3 if activity == "Consolidada na refined" else 0.1)
            entry = {"ts": moment.isoformat().replace("+00:00", "Z"), "level": "INFO", "message": activity,
                     "activity": activity, "case_id": case, "field_id": "TALHAO-01"}
            if activity == "Leitura gerada":
                entry["scenario"] = "dry" if case == "b" else "normal"
            if activity == "Validada" and case == "b":
                entry["out_of_range_fields"] = ["soil_moisture_percent"]
            (producer if activity in {"Leitura gerada", "Publicada no Kafka"} else consumer).append(json.dumps(entry))
    (directory / "producer.log").write_text("\n".join(producer) + "\nnot-json\n", encoding="utf-8")
    (directory / "consumer.log").write_text("\n".join(consumer) + "\n", encoding="utf-8")


class ProcessMiningTest(unittest.TestCase):
    def test_event_log_from_json_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_logs(Path(tmp))
            frame = enrich_cases(read_activity_logs(Path(tmp)))
        self.assertEqual(frame["case_id"].nunique(), 3)
        self.assertEqual(len(frame), 8 + 9 + 5)
        self.assertTrue((frame.loc[frame["case_id"] == "b", "scenario"] == "dry").all())
        self.assertEqual(frame[frame["case_id"] == "a"]["activity"].tolist(), HAPPY)

    @unittest.skipIf(process_mining is None, "matplotlib nao instalado")
    def test_variants_bottlenecks_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_logs(Path(tmp))
            frame = enrich_cases(read_activity_logs(Path(tmp)))
        table = process_mining.variants(frame)
        self.assertEqual(len(table), 3)
        self.assertEqual(set(table["resultado"]), {"Aceita sem alerta", "Aceita com alerta", "Rejeitada na validacao"})
        trans = process_mining.transitions(frame)
        worst = trans.sort_values("mediana_s", ascending=False).iloc[0]
        self.assertEqual((worst["de"], worst["para"]), ("Gravada na trusted", "Consolidada na refined"))
        phases = process_mining.lead_times(frame)
        self.assertEqual(len(phases), 2)
        summary = process_mining.build_summary(frame, table, trans, phases)
        self.assertEqual(summary["leituras_fora_da_faixa"], 1)
        self.assertEqual(summary["alertas_mascarados"], 0)
        self.assertEqual(summary["caminho_feliz_pct"], 66.7)


if __name__ == "__main__":
    unittest.main()
