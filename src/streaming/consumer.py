import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Protocol, Callable

sys.path.append(str(Path(__file__).resolve().parents[2]))

from services.data_lake import (
    ensure_data_lake,
    load_seen_event_ids,
    recompute_field_status,
    save_raw_event,
    save_rejected_sensor_event,
    save_trusted_sensor_reading,
    parse_datetime,
)
from services.sensor_validator import validate_sensor_event
from services.threshold_config import get_threshold_config


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
) -> bool:
    save_raw_event(event, dataset="sensors", base_path=base_path or os.getenv("DATA_LAKE_PATH", "data_lake"))

    normalized, errors = validate_sensor_event(
        event,
        source="kafka",
        seen_event_ids=seen_event_ids,
        threshold_config=threshold_config,
    )
    if normalized:
        save_trusted_sensor_reading(normalized, base_path=base_path or os.getenv("DATA_LAKE_PATH", "data_lake"))
        seen_event_ids.add(normalized["event_id"])
        print(f"Trusted: {normalized['event_id']}")
        return True

    save_rejected_sensor_event(
        event,
        errors,
        source="kafka",
        base_path=base_path or os.getenv("DATA_LAKE_PATH", "data_lake"),
    )
    print(f"Rejected: {event.get('event_id', 'sem-event-id')} -> {', '.join(errors)}")
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
    last_refine = clock()

    def flush() -> None:
        nonlocal accepted_since_refine, last_refine
        if accepted_since_refine:
            recompute_field_status(data_lake_path, output_csv, config, field_ids=set(affected_fields),
                                   field_reference_times=dict(field_times))
        if acknowledge:
            for processed_message in pending_messages:
                acknowledge(processed_message)
        pending_messages.clear()
        affected_fields.clear()
        field_times.clear()
        accepted_since_refine = 0
        last_refine = clock()

    try:
        for message in messages:
            # None represents an idle Kafka poll, so deadlines also fire without new events.
            if message is None:
                if pending_messages and clock() - last_refine >= refine_seconds:
                    flush()
                continue
            event, decode_errors = decode_message_value(message.value)
            if decode_errors:
                save_raw_event(event, dataset="sensors", base_path=data_lake_path)
                save_rejected_sensor_event(event, decode_errors, source="kafka", base_path=data_lake_path)
                was_accepted = False
                print(f"Rejected: {event.get('event_id', 'sem-event-id')} -> {', '.join(decode_errors)}")
            else:
                was_accepted = process_sensor_event(event, seen_event_ids, config, data_lake_path)
            pending_messages.append(message)
            if was_accepted:
                accepted_since_refine += 1
                affected_fields.add(event['field_id'])
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
        raise
    else:
        flush()

    return consumed


def main() -> None:
    try:
        from kafka import KafkaConsumer, TopicPartition, OffsetAndMetadata
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependencia kafka-python nao encontrada. Instale com: pip install -r requirements.txt"
        ) from exc

    args = parse_args()
    threshold_config = get_threshold_config()
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
        )
    except KeyboardInterrupt:
        print("Consumer encerrado com lote pendente consolidado.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
