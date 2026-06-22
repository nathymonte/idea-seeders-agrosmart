from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "agrosmart-events",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    group_id="agrosmart-consumer-v3",
    auto_offset_reset="earliest"
)

for msg in consumer:
    event = msg.value

    anomalia = event.get("anomalia", "Nenhuma")

    if anomalia != "Nenhuma":
        print(f"⚠️ ALERTA: {event.get('localidade', 'N/A')} - {anomalia}")
    else:
        print(f"OK: {event.get('image_name', 'desconhecido')} saudável")