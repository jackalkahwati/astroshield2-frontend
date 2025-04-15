"""
Kafka producer for publishing events from Astroshield.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
import os
import random
import time
from aiokafka import AIOKafkaProducer

from app.common.logging import logger

class KafkaProducer:
    """Kafka producer for event-driven architecture."""
    
    def __init__(
        self,
        bootstrap_servers: str = None,
        client_id: str = None,
        max_retries: int = 3,
        retry_backoff_ms: int = 100,
        max_backoff_ms: int = 5000,
        jitter_factor: float = 0.1
    ):
        """
        Initialize the Kafka producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers (comma separated)
            client_id: Producer client ID
            max_retries: Maximum number of retry attempts
            retry_backoff_ms: Initial backoff time in milliseconds
            max_backoff_ms: Maximum backoff time in milliseconds
            jitter_factor: Random jitter factor (0-1) to add to backoff
        """
        # Use environment variables if not provided
        self.bootstrap_servers = bootstrap_servers or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.client_id = client_id or os.environ.get("KAFKA_PRODUCER_CLIENT_ID", "astroshield-producer")
        
        # Retry configuration
        self.max_retries = max_retries
        self.retry_backoff_ms = retry_backoff_ms
        self.max_backoff_ms = max_backoff_ms
        self.jitter_factor = max(0, min(1, jitter_factor))
        
        # Initialize producer
        self.producer = None
        
        # Security configuration
        self.security_protocol = os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
        self.sasl_mechanism = os.environ.get("KAFKA_SASL_MECHANISM")
        self.sasl_username = os.environ.get("KAFKA_PRODUCER_USERNAME")
        self.sasl_password = os.environ.get("KAFKA_PRODUCER_PASSWORD")
        
        logger.info(f"Initialized Kafka producer with bootstrap servers: {self.bootstrap_servers}")
    
    async def start(self):
        """Start the Kafka producer."""
        if self.producer:
            return
        
        # Configure the Kafka producer
        config = {
            "bootstrap_servers": self.bootstrap_servers.split(","),
            "client_id": self.client_id,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8")
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
        
        # Create the Kafka producer
        self.producer = AIOKafkaProducer(**config)
        await self.producer.start()
        
        logger.info("Kafka producer started")
    
    async def stop(self):
        """Stop the Kafka producer."""
        if not self.producer:
            return
        
        logger.info("Stopping Kafka producer")
        await self.producer.stop()
        self.producer = None
        
        logger.info("Kafka producer stopped")
    
    async def send_async(self, topic: str, value: Dict[str, Any], key: str = None, headers: List[tuple] = None) -> bool:
        """
        Send a message to Kafka with retry mechanism.
        
        Args:
            topic: Kafka topic
            value: Message value
            key: Message key
            headers: Message headers
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.producer:
            await self.start()
        
        # Prepare message key
        if key is not None and isinstance(key, str):
            key = key.encode("utf-8")
        
        # Handle retries
        attempt = 0
        backoff_ms = self.retry_backoff_ms
        
        while True:
            attempt += 1
            
            try:
                await self.producer.send_and_wait(topic, value, key=key, headers=headers)
                logger.info(f"Message sent to {topic} (attempt {attempt}/{self.max_retries+1})")
                return True
            
            except Exception as e:
                if attempt <= self.max_retries:
                    # Calculate backoff with jitter
                    jitter = random.uniform(
                        1 - self.jitter_factor,
                        1 + self.jitter_factor
                    )
                    actual_backoff = min(
                        backoff_ms * jitter,
                        self.max_backoff_ms
                    )
                    
                    logger.warning(
                        f"Failed to send message to {topic} (attempt {attempt}/{self.max_retries+1}), "
                        f"retrying in {actual_backoff:.1f}ms. Error: {str(e)}"
                    )
                    
                    # Sleep and increase backoff for next attempt
                    await asyncio.sleep(actual_backoff / 1000)
                    backoff_ms = min(
                        backoff_ms * 2,
                        self.max_backoff_ms
                    )
                
                else:
                    logger.error(
                        f"Failed to send message to {topic} after {self.max_retries+1} attempts: {str(e)}"
                    )
                    return False
    
    def send(self, topic: str, value: Dict[str, Any], key: str = None, headers: List[tuple] = None) -> bool:
        """
        Synchronous wrapper for send_async.
        This is a blocking call that will wait for the message to be sent.
        
        Args:
            topic: Kafka topic
            value: Message value
            key: Message key
            headers: Message headers
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        # Use asyncio.run in a try-except block
        try:
            return asyncio.run(self.send_async(topic, value, key, headers))
        except Exception as e:
            logger.error(f"Error in synchronous send: {str(e)}")
            return False 