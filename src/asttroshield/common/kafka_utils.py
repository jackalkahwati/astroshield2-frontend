"""
Kafka Utilities Module

This module provides utilities for interacting with Kafka, including publishing
and consuming standardized messages with traceability support.
"""

import os
import json
import logging
import threading
from typing import Any, Dict, Optional, List, Callable, Union
from datetime import datetime

from confluent_kafka import Producer, Consumer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic

from src.asttroshield.common.message_headers import MessageHeader, MessageFactory
from src.asttroshield.common.logging_utils import get_logger, trace_context, get_current_trace_id


logger = get_logger(__name__)


class KafkaConfig:
    """Configuration helper for Kafka connections."""
    
    def __init__(
        self,
        bootstrap_servers: str,
        security_protocol: str = "PLAINTEXT",
        sasl_mechanism: str = None,
        username: str = None,
        password: str = None,
        client_id: str = None,
        group_id: str = None,
        ssl_cafile: str = None,
    ):
        """
        Initialize Kafka configuration.
        
        Args:
            bootstrap_servers: Comma-separated list of broker addresses
            security_protocol: Security protocol (PLAINTEXT, SASL_SSL, etc.)
            sasl_mechanism: SASL mechanism (PLAIN, SCRAM-SHA-256, etc.)
            username: SASL username
            password: SASL password
            client_id: Client ID for producer/consumer
            group_id: Consumer group ID
            ssl_cafile: CA certificate file path for SSL
        """
        self.bootstrap_servers = bootstrap_servers
        self.security_protocol = security_protocol
        self.sasl_mechanism = sasl_mechanism
        self.username = username
        self.password = password
        self.client_id = client_id or f"astroshield-{os.getpid()}"
        self.group_id = group_id
        self.ssl_cafile = ssl_cafile
    
    def get_producer_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for a Kafka producer.
        
        Returns:
            Dict[str, Any]: Producer configuration
        """
        config = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': self.client_id,
        }
        
        self._add_security_config(config)
        return config
    
    def get_consumer_config(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration dictionary for a Kafka consumer.
        
        Args:
            group_id: Override the default group ID
            
        Returns:
            Dict[str, Any]: Consumer configuration
        """
        config = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': self.client_id,
            'group.id': group_id or self.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
        }
        
        self._add_security_config(config)
        return config
    
    def get_admin_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for a Kafka admin client.
        
        Returns:
            Dict[str, Any]: Admin client configuration
        """
        config = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': f"{self.client_id}-admin",
        }
        
        self._add_security_config(config)
        return config
    
    def _add_security_config(self, config: Dict[str, Any]) -> None:
        """
        Add security-related configuration to a config dictionary.
        
        Args:
            config: Configuration dictionary to modify
        """
        if self.security_protocol:
            config['security.protocol'] = self.security_protocol
        
        if self.sasl_mechanism:
            config['sasl.mechanism'] = self.sasl_mechanism
        
        if self.username and self.password:
            config['sasl.username'] = self.username
            config['sasl.password'] = self.password
        
        if self.ssl_cafile:
            config['ssl.ca.location'] = self.ssl_cafile


class AstroShieldProducer:
    """
    Producer for publishing standardized messages to Kafka.
    
    This class ensures all messages have proper headers with traceability information.
    """
    
    def __init__(self, config: Union[KafkaConfig, Dict[str, Any]], source_system: str):
        """
        Initialize the producer.
        
        Args:
            config: KafkaConfig instance or configuration dictionary
            source_system: Name of the source system for message headers
        """
        if isinstance(config, KafkaConfig):
            self.config = config.get_producer_config()
        else:
            self.config = config
        
        self.source_system = source_system
        self._producer = None
    
    def _ensure_producer(self) -> None:
        """Ensure the producer is initialized."""
        if self._producer is None:
            logger.info("Initializing Kafka producer")
            self._producer = Producer(self.config)
    
    def _delivery_report(self, err: Any, msg: Any) -> None:
        """
        Handle delivery reports from Kafka.
        
        Args:
            err: Error information (if any)
            msg: Message that was delivered
        """
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            topic = msg.topic()
            partition = msg.partition()
            offset = msg.offset()
            logger.info(f"Message delivered to {topic} [{partition}] at offset {offset}")
    
    def publish(
        self,
        topic: str,
        message_type: str,
        payload: Dict[str, Any],
        key: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_message_ids: Optional[List[str]] = None,
        headers: Optional[List[tuple]] = None,
    ) -> None:
        """
        Publish a message to Kafka with standardized header.
        
        Args:
            topic: Kafka topic to publish to
            message_type: Type of message
            payload: Message payload
            key: Message key (optional)
            trace_id: Trace ID for traceability (uses current context if None)
            parent_message_ids: List of parent message IDs
            headers: Additional Kafka headers (optional)
        """
        self._ensure_producer()
        
        # Use current trace ID from context if none provided
        trace_id = trace_id or get_current_trace_id()
        if trace_id == 'NO_TRACE':
            trace_id = None  # Let MessageFactory generate a new one
        
        # Create standardized message
        message = MessageFactory.create_message(
            message_type=message_type,
            source=self.source_system,
            payload=payload,
            trace_id=trace_id,
            parent_message_ids=parent_message_ids
        )
        
        # Prepare Kafka headers
        kafka_headers = headers or []
        kafka_headers.append(('trace_id', message['header']['traceId'].encode('utf-8')))
        
        # Encode key if provided
        key_bytes = key.encode('utf-8') if key else None
        
        # Publish message
        with trace_context(message['header']['traceId']):
            logger.info(f"Publishing {message_type} message to {topic}")
            self._producer.produce(
                topic=topic,
                key=key_bytes,
                value=json.dumps(message).encode('utf-8'),
                headers=kafka_headers,
                callback=self._delivery_report
            )
            self._producer.poll(0)  # Trigger callbacks for any previous messages
    
    def flush(self, timeout: int = 10) -> None:
        """
        Flush any outstanding messages.
        
        Args:
            timeout: Flush timeout in seconds
        """
        if self._producer is not None:
            logger.debug(f"Flushing producer (timeout: {timeout}s)")
            self._producer.flush(timeout)
    
    def close(self) -> None:
        """Close the producer and free resources."""
        self.flush()
        self._producer = None
        logger.info("Kafka producer closed")


class AstroShieldConsumer:
    """
    Consumer for processing standardized messages from Kafka.
    
    This class ensures trace context is propagated during message processing.
    """
    
    def __init__(
        self,
        config: Union[KafkaConfig, Dict[str, Any]],
        topics: List[str],
        processor: Callable[[Dict[str, Any]], None],
        group_id: Optional[str] = None,
    ):
        """
        Initialize the consumer.
        
        Args:
            config: KafkaConfig instance or configuration dictionary
            topics: List of topics to subscribe to
            processor: Callback function that processes messages
            group_id: Consumer group ID (overrides the one in config)
        """
        if isinstance(config, KafkaConfig):
            self.config = config.get_consumer_config(group_id)
        else:
            self.config = config
            if group_id:
                self.config['group.id'] = group_id
        
        self.topics = topics
        self.processor = processor
        self._consumer = None
        self._running = False
    
    def _ensure_consumer(self) -> None:
        """Ensure the consumer is initialized."""
        if self._consumer is None:
            logger.info(f"Initializing Kafka consumer for topics: {', '.join(self.topics)}")
            self._consumer = Consumer(self.config)
            self._consumer.subscribe(self.topics)
    
    def start(self, poll_timeout: float = 1.0) -> None:
        """
        Start consuming messages.
        
        Args:
            poll_timeout: Poll timeout in seconds
        """
        self._ensure_consumer()
        self._running = True
        
        logger.info(f"Starting consumer for topics: {', '.join(self.topics)}")
        
        try:
            while self._running:
                msg = self._consumer.poll(poll_timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition, not an error
                        logger.debug(f"Reached end of partition {msg.partition()}")
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                    continue
                
                try:
                    # Parse message
                    message_str = msg.value().decode('utf-8')
                    message = json.loads(message_str)
                    
                    # Extract trace ID from message or Kafka headers
                    trace_id = None
                    if message.get('header', {}).get('traceId'):
                        trace_id = message['header']['traceId']
                    else:
                        # Try to get trace ID from Kafka headers
                        for key, value in msg.headers() or []:
                            if key == 'trace_id':
                                trace_id = value.decode('utf-8')
                                break
                    
                    # Process message within trace context
                    with trace_context(trace_id):
                        logger.info(f"Processing message from {msg.topic()}")
                        self.processor(message)
                
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message as JSON: {e}")
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")
        
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        except Exception as e:
            logger.exception(f"Unexpected error in consumer: {e}")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop consuming messages."""
        self._running = False
        if self._consumer is not None:
            logger.info("Closing Kafka consumer")
            self._consumer.close()
            self._consumer = None


def list_topics(config: Union[KafkaConfig, Dict[str, Any]]) -> List[str]:
    """
    List available Kafka topics.
    
    Args:
        config: KafkaConfig instance or configuration dictionary
        
    Returns:
        List[str]: List of topic names
    """
    if isinstance(config, KafkaConfig):
        admin_config = config.get_admin_config()
    else:
        admin_config = config
    
    admin_client = AdminClient(admin_config)
    metadata = admin_client.list_topics(timeout=10)
    
    return [topic for topic in metadata.topics.keys()]


def create_topic(
    config: Union[KafkaConfig, Dict[str, Any]],
    topic: str,
    num_partitions: int = 1,
    replication_factor: int = 1
) -> bool:
    """
    Create a new Kafka topic.
    
    Args:
        config: KafkaConfig instance or configuration dictionary
        topic: Topic name
        num_partitions: Number of partitions
        replication_factor: Replication factor
        
    Returns:
        bool: True if successful, False otherwise
    """
    if isinstance(config, KafkaConfig):
        admin_config = config.get_admin_config()
    else:
        admin_config = config
    
    admin_client = AdminClient(admin_config)
    
    new_topic = NewTopic(
        topic,
        num_partitions=num_partitions,
        replication_factor=replication_factor
    )
    
    try:
        logger.info(f"Creating topic: {topic}")
        result = admin_client.create_topics([new_topic])
        
        # Wait for operation to complete
        for topic_name, future in result.items():
            future.result()  # Raises exception on failure
        
        logger.info(f"Topic {topic} created successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create topic {topic}: {e}")
        return False 