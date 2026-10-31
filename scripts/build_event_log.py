"""Gera o event log (mineracao de processos) a partir dos logs JSON da plataforma.

Cada linha de log com o campo ``activity`` vira um evento: case_id (event_id da leitura),
atividade, timestamp e recurso (componente que executou o passo).

Uso:
    python -m scripts.build_event_log --logs logs --out output/process_mining/event_log.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.paths import PROJECT_ROOT  # noqa: E402

ATTRIBUTES = ["field_id", "scenario", "fault", "attention_level", "decided_by", "out_of_range_fields", "errors",
              "ingestion_latency_s",
              "batch_size", "partition", "offset", "recommendation"]


def read_activity_logs(log_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(log_dir.glob("*.log*")):
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if not entry.get("activity") or not entry.get("case_id"):
                    continue
                row = {"case_id": entry["case_id"], "activity": entry["activity"], "timestamp": entry["ts"],
                       "resource": entry.get("component", path.stem.split(".")[0])}
                for key in ATTRIBUTES:
                    value = entry.get(key)
                    row[key] = "; ".join(value) if isinstance(value, list) else value
                rows.append(row)
    frame = pd.DataFrame(rows, columns=["case_id", "activity", "timestamp", "resource", *ATTRIBUTES])
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    # Ordem estavel: horario e, em empates, a ordem natural do processo no log
    frame = frame.drop_duplicates().sort_values(["case_id", "timestamp"], kind="stable").reset_index(drop=True)
    return frame


def enrich_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """Propaga atributos do caso (talhao, cenario) para todos os eventos do mesmo case_id."""
    for column in ("field_id", "scenario", "fault"):
        frame[column] = frame.groupby("case_id")[column].transform(lambda s: s.ffill().bfill())
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "output/process_mining/event_log.csv")
    parser.add_argument("--xes", action="store_true", help="Tambem exporta em XES (requer pm4py).")
    args = parser.parse_args()
    frame = read_activity_logs(args.logs)
    if frame.empty:
        raise SystemExit(f"Nenhuma atividade encontrada em {args.logs}. Execute producer e consumer primeiro.")
    frame = enrich_cases(frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Event log: {len(frame)} eventos, {frame['case_id'].nunique()} casos, "
          f"{frame['activity'].nunique()} atividades -> {args.out}")
    if args.xes:
        import pm4py

        log = pm4py.format_dataframe(frame.astype({"timestamp": "datetime64[ns, UTC]"}), case_id="case_id",
                                     activity_key="activity", timestamp_key="timestamp")
        xes_path = args.out.with_suffix(".xes")
        pm4py.write_xes(log, str(xes_path))
        print(f"XES: {xes_path}")


if __name__ == "__main__":
    main()
