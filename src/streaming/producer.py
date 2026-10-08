import json
import os
import random
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from uuid import uuid4
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.threshold_config import get_threshold_config
from services.paths import validate_field_id


def build_event(field_id="TALHAO-01", scenario="random", rng=None, timestamp=None) -> dict[str, object]:
    validate_field_id(field_id)
    if scenario not in {"random", "normal", "dry", "hot", "high"}:
        raise ValueError("Cenario desconhecido")
    generator = rng or random
    config = get_threshold_config()["sensor_fields"]
    event = {
        "event_id": f"evt-{uuid4()}",
        "sensor_id": f"SENSOR-{field_id}",
        "field_id": field_id,
        "event_timestamp": (timestamp or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z"),
        "simulation_scenario": scenario,
    }
    for field, settings in config.items():
        limits = settings.get("expected_range", settings["allowed_range"])
        low, high = limits["minimum"], limits["maximum"]
        if field == "soil_moisture_percent" and scenario in {"dry", "high"}:
            low, high = settings["allowed_range"]["minimum"], low
        if field == "air_temperature_celsius" and scenario in {"hot", "high"}:
            low, high = high, settings["allowed_range"]["maximum"]
        event[field] = round(generator.uniform(low, high) if scenario == "random" else (low + high) / 2, 2)
    return event


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Publica leituras simuladas de sensores no Kafka.")
    parser.add_argument("--count", type=int, default=10, help="Quantidade de eventos a publicar.")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalo entre eventos em segundos.")
    parser.add_argument("--topic", default=os.getenv("KAFKA_SENSOR_TOPIC", "sensor-readings"))
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"))
    parser.add_argument("--fields", nargs="+", default=["TALHAO-01"])
    parser.add_argument("--scenario", choices=["random", "normal", "dry", "hot", "high"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    try:
        from kafka import KafkaProducer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependencia kafka-python nao encontrada. Instale com: pip install -r requirements.txt"
        ) from exc

    args = parse_args()
    if args.count < 1 or args.interval < 0:
        raise SystemExit("count deve ser positivo e interval nao negativo")
    for field_id in args.fields:
        validate_field_id(field_id)
    rng = random.Random(args.seed)
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )

    for index in range(args.count):
        event = build_event(args.fields[index % len(args.fields)], args.scenario, rng)
        producer.send(args.topic, value=event, key=event["field_id"].encode()).get(timeout=30)
        print(json.dumps(event, ensure_ascii=False))
        time.sleep(args.interval)

    producer.flush()
    producer.close()
    print("Producer finalizado.")


if __name__ == "__main__":
    main()
