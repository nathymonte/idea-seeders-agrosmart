import json
import os
import random
import sys
import time
from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.observability import get_logger, log_activity, utc_iso, write_heartbeat
from services.paths import validate_field_id
from services.threshold_config import get_threshold_config

logger = get_logger("producer")
SCENARIOS = ["random", "normal", "dry", "hot", "high"]
# Pesos do cenario "mixed": maioria normal/aleatoria, com eventos de seca, calor e risco alto.
MIXED_WEIGHTS = {"normal": 0.4, "random": 0.3, "dry": 0.12, "hot": 0.12, "high": 0.06}


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
        "event_timestamp": (timestamp or datetime.now(UTC)).isoformat().replace("+00:00", "Z"),
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
    parser.add_argument("--count", type=int, default=10, help="Quantidade de eventos (0 = continuo).")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalo entre eventos em segundos.")
    parser.add_argument("--topic", default=os.getenv("KAFKA_SENSOR_TOPIC", "sensor-readings"))
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"))
    parser.add_argument("--fields", nargs="+", default=["TALHAO-01"])
    parser.add_argument("--scenario", choices=SCENARIOS + ["mixed"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invalid-rate", type=float, default=0.0,
                        help="Fracao de eventos com valor fisicamente impossivel (teste de validacao).")
    parser.add_argument("--duplicate-rate", type=float, default=0.0,
                        help="Fracao de eventos reenviados com o mesmo event_id (teste de deduplicacao).")
    return parser.parse_args()


def pick_scenario(scenario: str, rng: random.Random) -> str:
    if scenario != "mixed":
        return scenario
    return rng.choices(list(MIXED_WEIGHTS), weights=list(MIXED_WEIGHTS.values()))[0]


def corrupt_event(event: dict, rng: random.Random) -> dict:
    """Injeta uma leitura impossivel para demonstrar a camada rejected."""
    broken = dict(event)
    broken[rng.choice(["soil_moisture_percent", "air_humidity_percent"])] = round(rng.uniform(120, 180), 2)
    broken["simulation_fault"] = "out_of_range"
    return broken


def main() -> None:
    try:
        from kafka import KafkaProducer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependencia kafka-python nao encontrada. Instale com: pip install -r requirements.txt"
        ) from exc

    args = parse_args()
    if args.count < 0 or args.interval < 0:
        raise SystemExit("count nao pode ser negativo e interval nao negativo")
    if not (0 <= args.invalid_rate <= 1 and 0 <= args.duplicate_rate <= 1):
        raise SystemExit("invalid-rate e duplicate-rate devem estar entre 0 e 1")
    for field_id in args.fields:
        validate_field_id(field_id)
    # Gerador pseudoaleatorio apenas para simular sensores (sem uso criptografico)
    rng = random.Random(args.seed)  # nosec B311
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )
    started_at = utc_iso()
    stats = {"sent": 0, "faults_injected": 0, "duplicates_injected": 0, "send_errors": 0}
    logger.info("producer iniciando", extra={"topic": args.topic, "count": args.count, "scenario": args.scenario,
                                            "fields": args.fields})

    def beat(state: str, last_event: dict | None = None) -> None:
        try:
            write_heartbeat("producer", state=state, started_at=started_at, scenario=args.scenario,
                            interval_s=args.interval, fields=args.fields, **stats,
                            last_event_id=(last_event or {}).get("event_id"), last_sent_at=utc_iso())
        except OSError as exc:
            logger.warning("falha ao gravar heartbeat", extra={"error": str(exc)})

    def publish(event: dict) -> None:
        try:
            metadata = producer.send(args.topic, value=event, key=event["field_id"].encode()).get(timeout=30)
        except Exception:
            stats["send_errors"] += 1
            logger.exception("falha ao publicar evento", extra={"case_id": event.get("event_id")})
            raise
        stats["sent"] += 1
        log_activity(logger, "Publicada no Kafka", event["event_id"], field_id=event["field_id"],
                     partition=metadata.partition, offset=metadata.offset)

    index = 0
    previous = None
    try:
        while args.count == 0 or index < args.count:
            scenario = pick_scenario(args.scenario, rng)
            event = build_event(args.fields[index % len(args.fields)], scenario, rng)
            if rng.random() < args.invalid_rate:
                event = corrupt_event(event, rng)
                stats["faults_injected"] += 1
            log_activity(logger, "Leitura gerada", event["event_id"], field_id=event["field_id"],
                         scenario=scenario, fault=event.get("simulation_fault"))
            publish(event)
            if previous and rng.random() < args.duplicate_rate:
                stats["duplicates_injected"] += 1
                log_activity(logger, "Reenvio (duplicata)", previous["event_id"], field_id=previous["field_id"])
                publish(previous)
            previous = event
            beat("running", event)
            index += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("producer interrompido pelo usuario")
    finally:
        producer.flush()
        producer.close()
        beat("finished", previous)
        logger.info("producer finalizado", extra=stats)


if __name__ == "__main__":
    main()
