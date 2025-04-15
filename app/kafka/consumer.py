"""
Kafka consumer for processing events in Astroshield.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
import os
from aiokafka import AIOKafkaConsumer
from asyncio import Task

from app.common.logging import logger
from app.kafka.event_handlers import create_event_handler
from app.kafka.producer import KafkaProducer

class EventConsumer:
    """Kafka consumer for event-driven architecture."""
    
    def __init__(
        self,
        bootstrap_servers: str = None,
        topic_prefixes: List[str] = None,
        group_id: str = None,
        producer: Optional[KafkaProducer] = None
    ):
        """
        Initialize the Kafka consumer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers (comma separated)
            topic_prefixes: List of topic prefixes to subscribe to
            group_id: Consumer group ID
            producer: Optional Kafka producer for sending response events
        """
        # Use environment variables if not provided
        self.bootstrap_servers = bootstrap_servers or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.topic_prefixes = topic_prefixes or os.environ.get("KAFKA_TOPIC_PREFIXES", "dmd-od-update,weather-data").split(",")
        self.group_id = group_id or os.environ.get("KAFKA_CONSUMER_GROUP", "astroshield-analyzer")
        
        # Initialize consumer configuration
        self.consumer = None
        self.producer = producer
        self.running = False
        self.tasks: Set[Task] = set()
        
        # Security configuration
        self.security_protocol = os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
        self.sasl_mechanism = os.environ.get("KAFKA_SASL_MECHANISM")
        self.sasl_username = os.environ.get("KAFKA_CONSUMER_USERNAME")
        self.sasl_password = os.environ.get("KAFKA_CONSUMER_PASSWORD")
        
        logger.info(f"Initialized Kafka consumer with bootstrap servers: {self.bootstrap_servers}")
        logger.info(f"Topic prefixes: {self.topic_prefixes}")
        logger.info(f"Consumer group ID: {self.group_id}")
    
    async def start(self):
        """Start the Kafka consumer."""
        if self.running:
            logger.warning("Kafka consumer already running")
            return
        
        # Configure the Kafka consumer
        config = {
            "bootstrap_servers": self.bootstrap_servers.split(","),
            "group_id": self.group_id,
            "auto_offset_reset": "earliest",  # Start from earliest message if no committed offset
            "enable_auto_commit": True,  # Automatically commit offsets
            "value_deserializer": lambda v: json.loads(v.decode("utf-8"))
        }
        
        # Add security configuration if needed
        if self.security_protocol != "PLAINTEXT":
            config["security_protocol"] = self.security_protocol
            
            if "SASL" in self.security_protocol:
                if self.sasl_mechanism:
                    config["sasl_mechanism"] = self.sasl_mechanism
                
                if self.sasl_username and self.sasl_password:
                    config["sasl_plain_username"] = self.sasl_username
                    config["sasl_plain_password"] = self.sasl_password
        
        # Create the Kafka consumer
        self.consumer = AIOKafkaConsumer(**config)
        
        # Subscribe to topics based on prefixes
        # In a real implementation, this might use a more sophisticated topic discovery mechanism
        # For now, we just subscribe to the exact prefixes provided
        await self.consumer.start()
        
        # Get all topics and filter based on prefixes
        all_topics = await self.consumer.topics()
        subscribe_topics = []
        
        for topic in all_topics:
            for prefix in self.topic_prefixes:
                if topic.startswith(prefix):
                    subscribe_topics.append(topic)
                    break
        
        if not subscribe_topics:
            logger.warning(f"No topics found matching prefixes: {self.topic_prefixes}")
            # Fall back to using the prefixes directly
            subscribe_topics = self.topic_prefixes
        
        logger.info(f"Subscribing to topics: {subscribe_topics}")
        self.consumer.subscribe(subscribe_topics)
        
        # Start consuming messages
        self.running = True
        
        try:
            # Start the main consumer loop
            await self._consume()
        except Exception as e:
            logger.error(f"Error starting Kafka consumer: {str(e)}")
            await self.stop()
    
    async def _consume(self):
        """Main consumer loop."""
        logger.info("Starting Kafka consumer loop")
        
        while self.running:
            try:
                async for message in self.consumer:
                    # Process message asynchronously
                    task = asyncio.create_task(self._process_message(message))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
            
            except Exception as e:
                logger.error(f"Error consuming Kafka messages: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _process_message(self, message):
        """
        Process a Kafka message.
        
        Args:
            message: Kafka message
        """
        try:
            # Extract message details
            topic = message.topic
            partition = message.partition
            offset = message.offset
            value = message.value
            
            logger.info(f"Processing message from {topic} [partition={partition}, offset={offset}]")
            
            # Extract message type from topic, value, or headers
            message_type = self._extract_message_type(topic, value)
            
            if message_type:
                # Create an event handler based on message type
                handler = create_event_handler(message_type, self.producer)
                
                if handler:
                    await handler.handle_event(value)
                else:
                    logger.warning(f"No handler found for message type: {message_type}")
            else:
                logger.warning(f"Could not determine message type for topic: {topic}")
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def _extract_message_type(self, topic: str, value: Dict[str, Any]) -> Optional[str]:
        """
        Extract message type from topic or value.
        
        Args:
            topic: Message topic
            value: Message value
            
        Returns:
            Message type or None if not found
        """
        # Try to extract message type from topic first
        for prefix in self.topic_prefixes:
            if topic.startswith(prefix):
                return prefix
        
        # Try to extract message type from value
        if isinstance(value, dict):
            header = value.get("header", {})
            message_type = header.get("messageType")
            
            if message_type:
                # Map message type to handler type
                if "dmd" in message_type.lower() and "orbit" in message_type.lower():
                    return "dmd-od-update"
                elif "weather" in message_type.lower():
                    return "weather-data"
        
        return None
    
    async def stop(self):
        """Stop the Kafka consumer."""
        if not self.running:
            return
        
        logger.info("Stopping Kafka consumer")
        self.running = False
        
        # Cancel all running tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop the consumer
        if self.consumer:
            await self.consumer.stop()
        
        logger.info("Kafka consumer stopped") 