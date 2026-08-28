import argparse
import json
import os
import sys
from pathlib import Path

from kafka import KafkaConsumer

sys.path.append(str(Path(__file__).resolve().parents[2]))

from services.data_lake import (
    ensure_data_lake,
    load_seen_event_ids,
    recompute_field_status,
    save_raw_event,
    save_rejected_sensor_event,
    save_trusted_sensor_reading,
)
from services.sensor_validator import validate_sensor_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consome leituras Kafka e grava no Data Lake AgroSmart.")
    parser.add_argument("--topic", default=os.getenv("KAFKA_SENSOR_TOPIC", "sensor-readings"))
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    parser.add_argument("--group-id", default="agrosmart-data-lake-consumer")
    parser.add_argument("--max-messages", type=int, default=0, help="0 consome continuamente.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data_lake()
    seen_event_ids = load_seen_event_ids()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        group_id=args.group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )

    consumed = 0
    for message in consumer:
        event = message.value
        save_raw_event(event, dataset="sensors")

        normalized, errors = validate_sensor_event(event, source="kafka", seen_event_ids=seen_event_ids)
        if normalized:
            save_trusted_sensor_reading(normalized)
            seen_event_ids.add(normalized["event_id"])
            recompute_field_status()
            print(f"Trusted: {normalized['event_id']}")
        else:
            save_rejected_sensor_event(event, errors, source="kafka")
            print(f"Rejected: {event.get('event_id', 'sem-event-id')} -> {', '.join(errors)}")

        consumed += 1
        if args.max_messages and consumed >= args.max_messages:
            break

    consumer.close()


if __name__ == "__main__":
    main()
