from kafka import KafkaProducer
import pandas as pd
import json
import time

CSV_PATH = "output/classificacoes.csv"
TOPIC = "agrosmart-events"

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

df = pd.read_csv(CSV_PATH)

for _, row in df.iterrows():
    event = row.to_dict()

    producer.send(TOPIC, value=event)

    print(f"Enviado: {event['image_name']}")

    time.sleep(1)

producer.flush()
producer.close()

print("Finalizado")