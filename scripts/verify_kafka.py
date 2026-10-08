"""Exercise the actual CLI consumer against Kafka using an isolated Data Lake."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from kafka import KafkaProducer
from services.data_lake import read_dataset, write_json
from services.paths import PROJECT_ROOT
from src.streaming.producer import build_event


def verify(bootstrap: str, destination: Path) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use uma pasta vazia para a verificacao Kafka")
    destination.mkdir(parents=True, exist_ok=True)
    topic = 'agrosmart-test-' + uuid4().hex[:12]
    producer = KafkaProducer(bootstrap_servers=bootstrap)
    first = build_event('TALHAO-01', 'normal')
    second = build_event('TALHAO-02', 'high')
    invalid_time = {**build_event(), 'event_timestamp': 'data-invalida'}
    payloads = [json.dumps(first).encode(), json.dumps(second).encode(),
                json.dumps(invalid_time).encode(), b'{invalid', json.dumps(first).encode()]
    for payload in payloads:
        producer.send(topic, value=payload, partition=0).get(timeout=30)
    producer.close()
    environment = {**os.environ, 'DATA_LAKE_PATH': str(destination / 'data_lake'),
                   'AGROSMART_IMAGE_CSV': str(destination / 'missing.csv')}
    command = [sys.executable, '-m', 'src.streaming.consumer', '--bootstrap', bootstrap,
               '--topic', topic, '--group-id', topic, '--max-messages', '5', '--refine-every', '10', '--refine-seconds', '1']
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment,
                               capture_output=True, text=True, timeout=60)
    (destination / 'consumer.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    lake = destination / 'data_lake'
    counts = {layer: len(read_dataset(layer, 'sensors' if layer == 'raw' else 'sensor_readings', lake))
              for layer in ['raw', 'trusted', 'rejected']}
    assert counts == {'raw': 5, 'trusted': 2, 'rejected': 3}, counts
    statuses = {p.stem: json.loads(p.read_text())['attention_level']
                for p in (lake / 'refined/field_status').glob('*.json')}
    assert statuses == {'TALHAO-01': 'normal', 'TALHAO-02': 'high'}, statuses
    result = {'topic': topic, 'counts': counts, 'statuses': statuses, 'real_kafka': True}
    write_json(destination / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', default='localhost:29092')
    parser.add_argument('--out', type=Path, default=PROJECT_ROOT / 'artifacts/kafka')
    args = parser.parse_args()
    print(json.dumps(verify(args.bootstrap, args.out), indent=2))
