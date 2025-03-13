import os
import sys
import json
import signal
import logging
from kafka import KafkaConsumer
from dotenv import load_dotenv

# Add src to Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the new message header utilities if available
try:
    from src.asttroshield.common.message_headers import MessageFactory
    USING_NEW_ARCHITECTURE = True
except ImportError:
    USING_NEW_ARCHITECTURE = False
    print("Running in legacy mode (new architecture not available)")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Flag to indicate whether the consumer should continue running
running = True

def signal_handler(sig, frame):
    """Handle termination signals to gracefully shut down the consumer."""
    global running
    logger.info("Received termination signal, shutting down...")
    running = False

def process_message(message_data):
    """Process the received message."""
    if not isinstance(message_data, dict):
        logger.warning("Received message is not a dictionary")
        return
    
    # Extract header and payload
    header = message_data.get('header', {})
    payload = message_data.get('payload', {})
    
    if not header or not payload:
        logger.warning("Message missing header or payload")
        return
    
    # Extract traceability information
    message_id = header.get('messageId', 'unknown')
    trace_id = header.get('traceId', message_id)  # Default to messageId if no traceId
    message_type = header.get('messageType', 'unknown')
    source = header.get('source', 'unknown')
    
    logger.info(f"Received message: ID={message_id}, Type={message_type}, Source={source}")
    logger.info(f"Trace ID: {trace_id}")
    
    # Log payload summary (without logging potentially large data)
    if isinstance(payload, dict):
        logger.info(f"Payload keys: {', '.join(payload.keys())}")
    
    # Here you would add your business logic to process the message
    
    # If using new architecture, you could create a derived message
    if USING_NEW_ARCHITECTURE:
        # Example of creating a derived message
        # derived_message = MessageFactory.create_derived_message(
        #     parent_message=message_data,
        #     message_type="derived.message",
        #     source="kafka_consume_example",
        #     payload={"processed": True, "result": "success"}
        # )
        # logger.info(f"Created derived message with ID: {derived_message['header']['messageId']}")
        pass

def main():
    # Kafka configuration
    TOPIC = "test-environment.ss0"  # Test topic with read+write permissions
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
    KAFKA_USERNAME = os.getenv('KAFKA_CONSUMER_USERNAME')
    KAFKA_PASSWORD = os.getenv('KAFKA_CONSUMER_PASSWORD')
    
    # Security configuration
    SECURITY_PROTOCOL = os.getenv('KAFKA_SECURITY_PROTOCOL')
    SASL_MECHANISM = os.getenv('KAFKA_SASL_MECHANISM')
    GROUP_ID = "python-kafka-consumer-example"

    logger.info(f"Connecting to bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Security protocol: {SECURITY_PROTOCOL}")
    logger.info(f"Consumer group: {GROUP_ID}")
    logger.info(f"Using new architecture: {USING_NEW_ARCHITECTURE}")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Create Kafka consumer
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
            security_protocol=SECURITY_PROTOCOL,
            sasl_mechanism=SASL_MECHANISM,
            sasl_plain_username=KAFKA_USERNAME,
            sasl_plain_password=KAFKA_PASSWORD,
            group_id=GROUP_ID,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        logger.info(f"Starting to consume from {TOPIC}")
        
        while running:
            # Poll for messages with a timeout
            message_batch = consumer.poll(timeout_ms=1000)
            
            for topic_partition, messages in message_batch.items():
                for message in messages:
                    try:
                        if message.value:
                            process_message(message.value)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
        logger.info("Consumer shutting down...")
        
    except Exception as e:
        logger.error(f"Error in Kafka consumer: {e}")
    finally:
        if 'consumer' in locals():
            consumer.close()
            logger.info("Consumer closed")

if __name__ == "__main__":
    main() 