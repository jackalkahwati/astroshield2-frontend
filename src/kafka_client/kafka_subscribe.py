import os
import json
from kafka import KafkaConsumer
from jsonschema import validate
from dotenv import load_dotenv
import ssl

# Load environment variables
load_dotenv()

def load_schema(schema_path):
    with open(schema_path, 'r') as f:
        return json.load(f)

def main():
    # Kafka configuration
    TOPIC = "ss0.sensor.heartbeat"  # Example topic
    VERSION = "0.2.0"
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
    KAFKA_USERNAME = os.getenv('KAFKA_CONSUMER_USERNAME')
    KAFKA_PASSWORD = os.getenv('KAFKA_CONSUMER_PASSWORD')
    
    # Load the schema
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'welders_arc_schemas', 'schemas', 'ss0', 'sensor', 'heartbeat', '0.2.0.json')
    schema = load_schema(schema_path)
    
    # SSL Context for SASL_SSL
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Create Kafka consumer with SCRAM-SHA-512
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
        security_protocol='SASL_SSL',
        sasl_mechanism='SCRAM-SHA-512',
        ssl_context=ssl_context,
        sasl_plain_username=KAFKA_USERNAME,
        sasl_plain_password=KAFKA_PASSWORD,
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        # Comment out for testing to see all messages:
        # group_id='consumer_group_1',
        api_version=(0, 10, 2)
    )

    print(f"Starting consumer for topic: {TOPIC}")
    print(f"Using bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Security protocol: SASL_SSL")
    print(f"SASL mechanism: SCRAM-SHA-512")
    print(f"Schema version: {VERSION}")
    
    try:
        for message in consumer:
            try:
                # Process the message
                data = message.value
                print(f"\nReceived message: {json.dumps(data, indent=2)}")
                
                # Validate against schema
                validate(instance=data, schema=schema)
                print("Message validation: SUCCESS")
                
            except json.JSONDecodeError as e:
                print(f"Error decoding message: {e}")
            except Exception as e:
                print(f"Error processing message: {e}")
                continue
    
    except KeyboardInterrupt:
        print("Shutting down consumer...")
    finally:
        consumer.close()

if __name__ == "__main__":
    main() 