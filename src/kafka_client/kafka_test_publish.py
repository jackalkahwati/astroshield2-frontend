import os
import json
from kafka import KafkaProducer
from dotenv import load_dotenv
import time
from datetime import datetime, timezone
import ssl

# Load environment variables
load_dotenv()

def main():
    # Kafka configuration
    TOPIC = "ss0.sensor.heartbeat"  # Heartbeat topic
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
    KAFKA_USERNAME = os.getenv('KAFKA_PRODUCER_USERNAME')
    KAFKA_PASSWORD = os.getenv('KAFKA_PRODUCER_PASSWORD')
    
    print(f"Connecting to bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Using producer credentials: {KAFKA_USERNAME}")

    # SSL Context for SASL_SSL
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Create Kafka producer with SCRAM-SHA-512
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
        security_protocol='SASL_SSL',
        sasl_mechanism='SCRAM-SHA-512',
        ssl_context=ssl_context,
        sasl_plain_username=KAFKA_USERNAME,
        sasl_plain_password=KAFKA_PASSWORD,
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
        api_version=(0, 10, 2)
    )

    # Create a test heartbeat message
    message = {
        "timeHeartbeat": datetime.now(timezone.utc).isoformat(),
        "idSensor": "test_sensor_001",
        "status": "OPERATIONAL",
        "eo": {
            "sunlit": True,
            "overcastRatio": 0.25
        },
        "description": "Test heartbeat message from Python client"
    }

    try:
        print(f"\nSending test message to {TOPIC}:")
        print(json.dumps(message, indent=2))
        
        # Send message
        future = producer.send(TOPIC, value=message)
        # Wait for message to be delivered
        record_metadata = future.get(timeout=10)
        
        print(f"\nMessage sent successfully:")
        print(f"Topic: {record_metadata.topic}")
        print(f"Partition: {record_metadata.partition}")
        print(f"Offset: {record_metadata.offset}")
        
    except Exception as e:
        print(f"Error sending message: {e}")
    finally:
        producer.close()

if __name__ == "__main__":
    main() 