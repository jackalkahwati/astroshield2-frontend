from kafka import KafkaProducer, KafkaConsumer
from typing import Callable, Dict, Any
import json
import logging
from threading import Thread
from uuid import uuid4
from datetime import datetime
import time
import random

logger = logging.getLogger(__name__)

class EventBus:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda v: v.encode('utf-8') if v else None
            )
            self.consumers: Dict[str, KafkaConsumer] = {}
            self.consumer_threads: Dict[str, Thread] = {}
            self.initialized = True
            self.message_uuid_counter = 0
            
            # Default retry configuration
            self.retry_config = {
                'max_retries': 3,
                'initial_backoff_ms': 100,
                'backoff_multiplier': 2,
                'max_backoff_ms': 5000,
                'jitter_factor': 0.1
            }

    def configure_retries(self, max_retries: int = None, initial_backoff_ms: int = None,
                       backoff_multiplier: float = None, max_backoff_ms: int = None,
                       jitter_factor: float = None):
        """Configure the retry behavior for message publishing
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_backoff_ms: Initial backoff time in milliseconds
            backoff_multiplier: Multiplier for exponential backoff
            max_backoff_ms: Maximum backoff time in milliseconds
            jitter_factor: Random jitter factor (0-1) to add to backoff
        """
        if max_retries is not None:
            self.retry_config['max_retries'] = max_retries
        if initial_backoff_ms is not None:
            self.retry_config['initial_backoff_ms'] = initial_backoff_ms
        if backoff_multiplier is not None:
            self.retry_config['backoff_multiplier'] = backoff_multiplier
        if max_backoff_ms is not None:
            self.retry_config['max_backoff_ms'] = max_backoff_ms
        if jitter_factor is not None:
            self.retry_config['jitter_factor'] = max(0, min(1, jitter_factor))
            
        logger.info(f"Updated retry configuration: {self.retry_config}")

    def publish(self, topic: str, event: Dict[str, Any], headers=None, retry=True):
        """Publish an event to a topic with standardized headers and retry mechanism
        
        Args:
            topic: Kafka topic to publish to
            event: Event data to publish
            headers: Optional message headers
            retry: Whether to retry on failure
        """
        # Generate standard headers if not provided
        if headers is None:
            headers = self._generate_standard_headers(topic)
        
        # Include headers in Kafka message
        kafka_headers = [(k, v.encode('utf-8') if isinstance(v, str) else json.dumps(v).encode('utf-8')) 
                       for k, v in headers.items()]
        
        # Prepare message details for logging and tracking
        message_id = headers.get('message_id', str(uuid4()))
        
        # Handle retries if enabled
        attempt = 0
        max_retries = self.retry_config['max_retries'] if retry else 0
        backoff_ms = self.retry_config['initial_backoff_ms']
        
        while True:
            attempt += 1
            try:
                future = self.producer.send(
                    topic, 
                    key=message_id, 
                    value=event,
                    headers=kafka_headers
                )
                future.get(timeout=10)
                logger.info(f"Published event to {topic} (id: {message_id})")
                return True  # Success
            except Exception as e:
                if attempt <= max_retries:
                    # Calculate backoff with jitter
                    jitter = random.uniform(
                        1 - self.retry_config['jitter_factor'],
                        1 + self.retry_config['jitter_factor']
                    )
                    actual_backoff = min(
                        backoff_ms * jitter,
                        self.retry_config['max_backoff_ms']
                    )
                    
                    logger.warning(
                        f"Failed to publish event to {topic} (id: {message_id}), "
                        f"attempt {attempt}/{max_retries+1}. "
                        f"Retrying in {actual_backoff:.1f}ms. Error: {str(e)}"
                    )
                    
                    # Sleep and increase backoff for next attempt
                    time.sleep(actual_backoff / 1000)
                    backoff_ms = min(
                        backoff_ms * self.retry_config['backoff_multiplier'],
                        self.retry_config['max_backoff_ms']
                    )
                else:
                    logger.error(
                        f"Failed to publish event to {topic} (id: {message_id}) "
                        f"after {max_retries+1} attempts: {str(e)}"
                    )
                    if not retry:
                        raise
                    return False  # Failed after retries

    def _generate_standard_headers(self, topic: str) -> Dict[str, str]:
        """Generate standardized message headers according to GitLab documentation"""
        message_id = f"astroshield-{self.message_uuid_counter}"
        self.message_uuid_counter += 1
        
        return {
            "message_id": message_id,
            "source": "astroshield",
            "source_id": "astroshield_service",
            "timestamp": datetime.utcnow().isoformat(),
            "topic": topic,
            "message_type": topic.split(".")[-1],
            "content_type": "application/json",
            "version": "1.0"
        }

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any], Dict[str, Any]], None]):
        """Subscribe to a topic with a handler function that receives both data and headers"""
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=['localhost:9092'],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='astroshield-group',
                auto_offset_reset='latest'
            )
            
            def consumer_thread():
                for message in consumer:
                    try:
                        # Convert headers from bytes to string/objects
                        headers = {}
                        for key, value in message.headers:
                            try:
                                # First try to decode as string
                                headers[key] = value.decode('utf-8')
                            except UnicodeDecodeError:
                                # If it fails, try to decode as JSON
                                try:
                                    headers[key] = json.loads(value.decode('utf-8'))
                                except:
                                    # If all fails, keep as bytes
                                    headers[key] = value
                                    
                        # Call handler with both message content and headers
                        handler(message.value, headers)
                    except Exception as e:
                        logger.error(f"Error handling message from {topic}: {str(e)}")

            thread = Thread(target=consumer_thread, daemon=True)
            thread.start()

            self.consumers[topic] = consumer
            self.consumer_threads[topic] = thread
            
            logger.info(f"Subscribed to topic: {topic}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {topic}: {str(e)}")
            raise

    def unsubscribe(self, topic: str):
        """Unsubscribe from a topic"""
        try:
            if topic in self.consumers:
                self.consumers[topic].close()
                del self.consumers[topic]
                logger.info(f"Unsubscribed from topic: {topic}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {topic}: {str(e)}")
            raise

    def close(self):
        """Close all connections"""
        try:
            self.producer.close()
            for consumer in self.consumers.values():
                consumer.close()
            self.consumers.clear()
            logger.info("Event bus closed")
        except Exception as e:
            logger.error(f"Failed to close event bus: {str(e)}")
            raise

# Example usage:
# event_bus = EventBus()
# event_bus.publish("spacecraft.maneuver", {"spacecraft_id": "123", "action": "orbit_adjust"})
# event_bus.subscribe("spacecraft.telemetry", lambda event: print(f"Received telemetry: {event}"))
