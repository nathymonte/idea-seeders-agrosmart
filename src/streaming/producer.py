import json
import os
import random
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from uuid import uuid4

from kafka import KafkaProducer


def build_event() -> dict[str, object]:
    return {
        "event_id": f"evt-{uuid4()}",
        "sensor_id": "SENSOR-001",
        "field_id": "TALHAO-01",
        "event_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "soil_moisture_percent": round(random.uniform(18, 72), 2),
        "air_temperature_celsius": round(random.uniform(18, 36), 2),
        "air_humidity_percent": round(random.uniform(42, 88), 2),
        "luminosity_lux": round(random.uniform(9000, 32000), 2),
    }


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Publica leituras simuladas de sensores no Kafka.")
    parser.add_argument("--count", type=int, default=10, help="Quantidade de eventos a publicar.")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalo entre eventos em segundos.")
    parser.add_argument("--topic", default=os.getenv("KAFKA_SENSOR_TOPIC", "sensor-readings"))
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )

    for _ in range(args.count):
        event = build_event()
        producer.send(args.topic, value=event)
        print(f"Enviado para {args.topic}: {event['event_id']}")
        time.sleep(args.interval)

    producer.flush()
    producer.close()
    print("Producer finalizado.")


if __name__ == "__main__":
    main()
