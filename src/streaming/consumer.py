import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

sys.path.append(str(Path(__file__).resolve().parents[2]))

from services.data_lake import (
    ensure_data_lake,
    load_seen_event_ids,
    parse_datetime,
    recompute_field_status,
    save_raw_event,
    save_rejected_sensor_event,
    save_trusted_sensor_reading,
)
from services.observability import get_logger, log_activity, utc_iso, write_heartbeat
from services.sensor_validator import validate_sensor_event
from services.threshold_config import get_threshold_config

logger = get_logger("consumer")
HEARTBEAT_SECONDS = float(os.getenv("AGROSMART_HEARTBEAT_SECONDS", "5"))


class ConsumerMetrics:
    """Contadores expostos no heartbeat e na aba Saude do sistema."""

    def __init__(self) -> None:
        self.started_at = utc_iso()
        self.received = self.accepted = self.rejected = self.duplicates = self.batches = 0
        self.last_event_id = self.last_event_at = self.last_batch_at = None
        self.last_batch_ms = 0.0
        self.latency_total = 0.0
        self.latency_count = 0
        self.max_latency = 0.0
        self.pending = 0
        self._last_beat = 0.0

    def observe_latency(self, event: dict) -> float | None:
        try:
            generated = parse_datetime(event.get("event_timestamp") or event.get("timestamp"))
        except (TypeError, ValueError):
            return None
        latency = max(0.0, time.time() - generated.timestamp())
        self.latency_total += latency
        self.latency_count += 1
        self.max_latency = max(self.max_latency, latency)
        return round(latency, 3)

    def snapshot(self, state: str) -> dict:
        average = self.latency_total / self.latency_count if self.latency_count else 0.0
        return {
            "state": state, "started_at": self.started_at, "received": self.received,
            "accepted": self.accepted, "rejected": self.rejected, "duplicates": self.duplicates,
            "batches_refined": self.batches, "pending_in_batch": self.pending,
            "last_event_id": self.last_event_id, "last_event_at": self.last_event_at,
            "last_batch_at": self.last_batch_at, "last_batch_ms": round(self.last_batch_ms, 1),
            "avg_ingestion_latency_s": round(average, 3), "max_ingestion_latency_s": round(self.max_latency, 3),
            "rejection_rate": round((self.rejected + self.duplicates) / self.received, 4) if self.received else 0.0,
        }

    def beat(self, state: str = "running", force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_beat < HEARTBEAT_SECONDS:
            return
        self._last_beat = now
        try:
            write_heartbeat("consumer", **self.snapshot(state))
        except OSError as exc:
            logger.warning("falha ao gravar heartbeat", extra={"error": str(exc)})


def _out_of_expected_range(reading: dict, threshold_config: dict) -> list[str]:
    """Campos desta leitura fora da faixa operacional (usado na analise de alertas por leitura)."""
    fields = []
    for name, settings in threshold_config.get("sensor_fields", {}).items():
        limits = settings.get("expected_range")
        value = reading.get(name)
        if limits and isinstance(value, int | float) and not limits["minimum"] <= value <= limits["maximum"]:
            fields.append(name)
    return fields


def _rejection_activity(errors: list[str]) -> str:
    if any(error.startswith("event_id duplicado") for error in errors):
        return "Descartada (duplicada)"
    if any(error.startswith("payload") for error in errors):
        return "Rejeitada (payload invalido)"
    return "Rejeitada (validacao)"


class KafkaMessage(Protocol):
    value: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consome leituras Kafka e grava no Data Lake AgroSmart.")
    parser.add_argument("--topic", default=os.getenv("KAFKA_SENSOR_TOPIC", "sensor-readings"))
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"))
    parser.add_argument("--group-id", default="agrosmart-data-lake-consumer")
    parser.add_argument("--refine-seconds", type=float, default=float(os.getenv("AGROSMART_REFINE_SECONDS", "5")))
    parser.add_argument("--max-messages", type=int, default=0, help="0 consome continuamente.")
    parser.add_argument(
        "--refine-every",
        type=int,
        default=int(os.getenv("AGROSMART_REFINE_EVERY", "10")),
        help="Quantidade de mensagens validas antes de recalcular a camada refined.",
    )
    return parser.parse_args()


def process_sensor_event(
    event: dict,
    seen_event_ids: set[str],
    threshold_config: dict,
    base_path: Path | str = None,
    metrics: ConsumerMetrics | None = None,
    case_id: str | None = None,
) -> bool:
    save_raw_event(event, dataset="sensors", base_path=base_path or os.getenv("DATA_LAKE_PATH", "data_lake"))
    if case_id is None and isinstance(event.get("event_id"), str):
        case_id = event["event_id"]
    field_id = event.get("field_id") if isinstance(event.get("field_id"), str) else None
    log_activity(logger, "Gravada na raw", case_id, field_id=field_id)

    normalized, errors = validate_sensor_event(
        event,
        source="kafka",
        seen_event_ids=seen_event_ids,
        threshold_config=threshold_config,
    )
    if normalized:
        save_trusted_sensor_reading(normalized, base_path=base_path or os.getenv("DATA_LAKE_PATH", "data_lake"))
        seen_event_ids.add(normalized["event_id"])
        log_activity(logger, "Validada", case_id, field_id=field_id,
                     out_of_range_fields=_out_of_expected_range(normalized, threshold_config))
        log_activity(logger, "Gravada na trusted", case_id, field_id=field_id)
        if metrics:
            metrics.accepted += 1
        return True

    save_rejected_sensor_event(
        event,
        errors,
        source="kafka",
        base_path=base_path or os.getenv("DATA_LAKE_PATH", "data_lake"),
    )
    activity = _rejection_activity(errors)
    log_activity(logger, activity, case_id, field_id=field_id, errors=errors)
    if metrics:
        if activity == "Descartada (duplicada)":
            metrics.duplicates += 1
        else:
            metrics.rejected += 1
    return False


def decode_message_value(value: Any) -> tuple[dict, list[str]]:
    if isinstance(value, dict):
        return value, []

    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError:
            return {"raw_payload": value.decode("utf-8", errors="replace")}, ["payload nao esta em UTF-8 valido"]
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            return {"raw_payload": decoded}, [f"payload JSON invalido: {exc.msg}"]
        if not isinstance(payload, dict):
            return {"raw_payload": payload}, ["payload JSON deve ser um objeto"]
        return payload, []

    return {"raw_payload": value}, ["payload deve ser um objeto JSON"]


def consume_messages(
    messages: Iterable[KafkaMessage],
    max_messages: int = 0,
    refine_every: int = 10,
    base_path: Path | str = None,
    output_csv: Path | str = "output/classificacoes.csv",
    threshold_config: dict | None = None,
    refine_seconds: float = 5,
    clock: Callable[[], float] = time.monotonic,
    acknowledge: Callable[[KafkaMessage], None] | None = None,
    metrics: ConsumerMetrics | None = None,
) -> int:
    if refine_every < 1:
        raise ValueError("refine_every deve ser maior ou igual a 1")
    if refine_seconds <= 0:
        raise ValueError("refine_seconds deve ser positivo")

    data_lake_path = base_path or os.getenv("DATA_LAKE_PATH", "data_lake")
    config = threshold_config or get_threshold_config()
    ensure_data_lake(data_lake_path)
    seen_event_ids = load_seen_event_ids(data_lake_path)
    consumed = 0
    accepted_since_refine = 0
    affected_fields: set[str] = set()
    field_times = {}
    pending_messages = []
    batch_cases: dict[str, list[str]] = {}
    stats = metrics or ConsumerMetrics()
    last_refine = clock()

    def flush() -> None:
        nonlocal accepted_since_refine, last_refine
        if accepted_since_refine:
            started = time.perf_counter()
            written = recompute_field_status(data_lake_path, output_csv, config, field_ids=set(affected_fields),
                                             field_reference_times=dict(field_times))
            stats.last_batch_ms = (time.perf_counter() - started) * 1000
            stats.batches += 1
            stats.last_batch_at = utc_iso()
            _log_batch(written, batch_cases, accepted_since_refine, stats.last_batch_ms)
        if acknowledge:
            for processed_message in pending_messages:
                acknowledge(processed_message)
        pending_messages.clear()
        affected_fields.clear()
        field_times.clear()
        batch_cases.clear()
        accepted_since_refine = 0
        stats.pending = 0
        last_refine = clock()
        stats.beat()

    try:
        for message in messages:
            # None represents an idle Kafka poll, so deadlines also fire without new events.
            if message is None:
                if pending_messages and clock() - last_refine >= refine_seconds:
                    flush()
                stats.beat("idle")
                continue
            event, decode_errors = decode_message_value(message.value)
            stats.received += 1
            case_id = event.get("event_id") if isinstance(event.get("event_id"), str) else None
            # Mensagens sem identificador recebem um case_id proprio para nao se misturarem no event log
            case_id = case_id or f"sem-id-{uuid4().hex[:12]}"
            latency = stats.observe_latency(event) if not decode_errors else None
            log_activity(logger, "Recebida pelo consumer", case_id,
                         field_id=event.get("field_id") if isinstance(event.get("field_id"), str) else None,
                         ingestion_latency_s=latency)
            if decode_errors:
                save_raw_event(event, dataset="sensors", base_path=data_lake_path)
                log_activity(logger, "Gravada na raw", case_id)
                save_rejected_sensor_event(event, decode_errors, source="kafka", base_path=data_lake_path)
                was_accepted = False
                stats.rejected += 1
                log_activity(logger, "Rejeitada (payload invalido)", case_id, errors=decode_errors)
            else:
                was_accepted = process_sensor_event(event, seen_event_ids, config, data_lake_path, stats, case_id)
            stats.last_event_id = case_id
            stats.last_event_at = utc_iso()
            pending_messages.append(message)
            if was_accepted:
                accepted_since_refine += 1
                stats.pending = accepted_since_refine
                affected_fields.add(event['field_id'])
                batch_cases.setdefault(event['field_id'], []).append(event['event_id'])
                observed = parse_datetime(event.get('event_timestamp') or event.get('timestamp'))
                field_times[event['field_id']] = max(observed, field_times.get(event['field_id'], observed))
            consumed += 1
            if (accepted_since_refine >= refine_every or clock() - last_refine >= refine_seconds
                    or not accepted_since_refine):
                flush()
            if max_messages and consumed >= max_messages:
                break
    except KeyboardInterrupt:
        flush()
        stats.beat("finished", force=True)
        raise
    else:
        flush()
    stats.beat("finished" if max_messages else "running", force=True)
    return consumed


def _log_batch(written, batch_cases: dict[str, list[str]], size: int, elapsed_ms: float) -> None:
    """Registra a consolidacao do lote e a decisao gerada para cada leitura aceita."""
    decisions = {}
    for path in written or []:
        try:
            status = json.loads(Path(path).read_text(encoding="utf-8"))
            decisions[status["field_id"]] = status
        except (OSError, ValueError, KeyError, TypeError):
            continue
    logger.info("lote consolidado", extra={"batch_size": size, "refine_ms": round(elapsed_ms, 1),
                                          "fields": sorted(batch_cases)})
    for field_id, case_ids in batch_cases.items():
        status = decisions.get(field_id, {})
        level = status.get("attention_level", "desconhecido")
        # O motor decide pela ultima leitura do talhao; so ela determina o alerta do lote
        decided_by = (status.get("latest_reading") or {}).get("event_id")
        for case_id in case_ids:
            log_activity(logger, "Consolidada na refined", case_id, field_id=field_id, batch_size=size)
            log_activity(logger, "Decisao gerada", case_id, field_id=field_id, attention_level=level,
                         decided_by=decided_by)
            if level in {"medium", "high"} and case_id == decided_by:
                log_activity(logger, "Alerta emitido", case_id, field_id=field_id, attention_level=level,
                             recommendation=status.get("recommendation"))


def main() -> None:
    try:
        from kafka import KafkaConsumer, OffsetAndMetadata, TopicPartition
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependencia kafka-python nao encontrada. Instale com: pip install -r requirements.txt"
        ) from exc

    args = parse_args()
    threshold_config = get_threshold_config()
    metrics = ConsumerMetrics()
    metrics.beat("starting", force=True)
    logger.info("consumer iniciando", extra={"topic": args.topic, "bootstrap": args.bootstrap,
                                            "refine_every": args.refine_every})
    ensure_data_lake()
    # Recover refined outputs after a previous interrupted batch before replaying Kafka offsets.
    recompute_field_status(threshold_config=threshold_config)

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        group_id=args.group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )

    def poll_messages():
        while True:
            batches = consumer.poll(timeout_ms=500, max_records=1)
            if not batches:
                yield None
            for batch in batches.values():
                yield from batch

    def acknowledge(message):
        partition = TopicPartition(message.topic, message.partition)
        consumer.commit({partition: OffsetAndMetadata(message.offset + 1, "")})

    try:
        consume_messages(
            poll_messages(),
            max_messages=args.max_messages,
            refine_every=args.refine_every,
            threshold_config=threshold_config,
            refine_seconds=args.refine_seconds,
            acknowledge=acknowledge,
            metrics=metrics,
        )
    except KeyboardInterrupt:
        logger.info("consumer encerrado com lote pendente consolidado")
    except Exception:
        logger.exception("consumer interrompido por erro")
        metrics.beat("error", force=True)
        raise
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
