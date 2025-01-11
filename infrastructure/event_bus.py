from kafka import KafkaProducer, KafkaConsumer
from typing import Callable, Dict, Any
import json
import logging
from threading import Thread

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
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            self.consumers: Dict[str, KafkaConsumer] = {}
            self.consumer_threads: Dict[str, Thread] = {}
            self.initialized = True

    def publish(self, topic: str, event: Dict[str, Any]):
        """Publish an event to a topic"""
        try:
            future = self.producer.send(topic, event)
            future.get(timeout=10)
            logger.info(f"Published event to {topic}: {event}")
        except Exception as e:
            logger.error(f"Failed to publish event to {topic}: {str(e)}")
            raise

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        """Subscribe to a topic with a handler function"""
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=['localhost:9092'],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='astroshield-group'
            )
            
            def consumer_thread():
                for message in consumer:
                    try:
                        handler(message.value)
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
