import os
import json
import uuid
import time
from datetime import datetime
from kafka import KafkaProducer
from dotenv import load_dotenv

# Add src to Python path if needed
import sys
import os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the new message header utilities if available
try:
    from src.asttroshield.common.message_headers import MessageFactory
    USING_NEW_ARCHITECTURE = True
except ImportError:
    USING_NEW_ARCHITECTURE = False
    print("Running in legacy mode (new architecture not available)")

# Load environment variables
load_dotenv()

def main():
    # Kafka configuration
    TOPIC = "test-environment.ss0"  # Test topic with read+write permissions
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
    KAFKA_USERNAME = os.getenv('KAFKA_PRODUCER_USERNAME')
    KAFKA_PASSWORD = os.getenv('KAFKA_PRODUCER_PASSWORD')
    
    # Security configuration
    SECURITY_PROTOCOL = os.getenv('KAFKA_SECURITY_PROTOCOL')
    SASL_MECHANISM = os.getenv('KAFKA_SASL_MECHANISM')

    print(f"Connecting to bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Security protocol: {SECURITY_PROTOCOL}")
    print(f"Using new architecture: {USING_NEW_ARCHITECTURE}")

    # Create Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
        security_protocol=SECURITY_PROTOCOL,
        sasl_mechanism=SASL_MECHANISM,
        sasl_plain_username=KAFKA_USERNAME,
        sasl_plain_password=KAFKA_PASSWORD,
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )

    # Example message payload
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Hello from Python Kafka producer!",
        "data": {
            "value": 42,
            "unit": "answer"
        }
    }

    # Create a message with proper header if using new architecture
    if USING_NEW_ARCHITECTURE:
        message = MessageFactory.create_message(
            message_type="example.message",
            source="kafka_publish_example",
            payload=payload
        )
        print(f"Created message with trace ID: {message['header']['traceId']}")
    else:
        # Legacy format
        message = {
            "header": {
                "messageId": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "kafka_publish_example",
                "messageType": "example.message"
            },
            "payload": payload
        }
        print(f"Created message with ID: {message['header']['messageId']}")

    try:
        # Send message
        future = producer.send(TOPIC, value=message)
        # Wait for message to be delivered
        record_metadata = future.get(timeout=10)
        
        print(f"Message sent successfully to {TOPIC}")
        print(f"Partition: {record_metadata.partition}")
        print(f"Offset: {record_metadata.offset}")
        
    except Exception as e:
        print(f"Error sending message: {e}")
    finally:
        producer.close()

if __name__ == "__main__":
    main() 