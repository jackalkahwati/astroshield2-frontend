"""
Event processor for handling Kafka messages.
"""
import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional

from ..validation import SchemaValidator

logger = logging.getLogger("astroshield.event_processing")

class EventProcessor:
    """Main event processor for handling Kafka events"""
    
    def __init__(self, consumer, producer, validator=None):
        """
        Initialize the event processor
        
        Args:
            consumer: Kafka consumer
            producer: Kafka producer
            validator: Schema validator
        """
        self.consumer = consumer
        self.producer = producer
        self.validator = validator or SchemaValidator()
        self.handlers = {}
        self.running = False
        
    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a handler for a message type
        
        Args:
            message_type: Type of message to handle
            handler: Handler function or method
        """
        self.handlers[message_type] = handler
        logger.info(f"Registered handler for {message_type}")
        
    async def process_message(self, message: Dict[str, Any]):
        """
        Process a message from Kafka
        
        Args:
            message: Message to process
        """
        message_type = message.get("header", {}).get("messageType")
        if not message_type:
            logger.warning("Message has no messageType")
            return
            
        # Validate message
        if not self.validator.validate_by_message_type(message):
            logger.warning(f"Invalid message format for {message_type}")
            return
            
        # Find and call handler
        handler = self.handlers.get(message_type)
        if not handler:
            logger.warning(f"No handler registered for {message_type}")
            return
            
        try:
            await handler(message)
            logger.debug(f"Successfully processed message of type {message_type}")
        except Exception as e:
            logger.error(f"Error processing {message_type}: {str(e)}")
            
    async def start(self):
        """Start the event processor"""
        self.running = True
        logger.info("Starting event processor")
        await self.consumer.start()
        
        while self.running:
            try:
                messages = await self.consumer.poll(timeout_ms=1000)
                for message in messages:
                    await self.process_message(message)
            except Exception as e:
                logger.error(f"Error polling messages: {str(e)}")
                await asyncio.sleep(1)  # Prevent tight loop on error
                
    async def stop(self):
        """Stop the event processor"""
        logger.info("Stopping event processor")
        self.running = False
        await self.consumer.stop()
        await self.producer.stop()
        
    def configure_handlers(self):
        """
        Configure standard event handlers
        
        This sets up the default handlers for standard message types
        """
        from .event_handlers import DMDOrbitDeterminationEventHandler, WeatherDataEventHandler
        
        # Create handlers
        dmd_handler = DMDOrbitDeterminationEventHandler(self.producer)
        weather_handler = WeatherDataEventHandler(self.producer)
        
        # Register handlers
        self.register_handler("dmd-od-update", dmd_handler.handle_event)
        self.register_handler("weather-data", weather_handler.handle_event)
        
        logger.info("Configured standard event handlers") 