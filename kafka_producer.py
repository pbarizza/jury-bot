# kafka_producer.py
from confluent_kafka import Producer
import json
import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "startups-created")

conf = {
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS
}

producer = Producer(**conf)

def send_startup_event(startup_dict):
    """Send a JSON event to Kafka"""
    try:
        msg = {
            "event_type": "startup_registered",
            "data": startup_dict
        }
        producer.produce(
            topic=KAFKA_TOPIC,
            value=json.dumps(msg)
        )
        producer.flush()  # Wait for delivery
        print(f"✅ Kafka event sent: {msg['data']['Startup Name']}")
    except Exception as e:
        print(f"❌ Kafka send failed: {e}")
