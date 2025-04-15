"""
Main event processor for Astroshield.
This module initializes and starts the event-driven processing pipeline.
"""

import asyncio
import logging
import os
import signal
from typing import Dict, List, Any, Optional

from app.common.logging import setup_logging, logger
from app.kafka.producer import KafkaProducer
from app.kafka.consumer import EventConsumer

# Flag to indicate if the application should be running
running = True

def signal_handler(sig, frame):
    """Signal handler for clean shutdown."""
    global running
    logger.info(f"Received signal {sig}, shutting down...")
    running = False

async def run_event_processor():
    """Initialize and run the event processor."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize the Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS"),
        client_id=os.environ.get("KAFKA_PRODUCER_CLIENT_ID", "astroshield-maneuver-processor"),
        max_retries=int(os.environ.get("KAFKA_MAX_RETRIES", "3")),
        retry_backoff_ms=int(os.environ.get("KAFKA_RETRY_BACKOFF_MS", "100")),
        max_backoff_ms=int(os.environ.get("KAFKA_MAX_BACKOFF_MS", "5000"))
    )
    await producer.start()
    
    # Initialize the Kafka consumer
    consumer = EventConsumer(
        bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS"),
        topic_prefixes=os.environ.get("KAFKA_TOPIC_PREFIXES", "dmd-od-update,weather-data").split(","),
        group_id=os.environ.get("KAFKA_CONSUMER_GROUP", "astroshield-maneuver-processor"),
        producer=producer
    )
    
    # Start consumer in background task
    consumer_task = asyncio.create_task(consumer.start())
    
    # Wait for shutdown signal
    try:
        while running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Received cancellation request")
    finally:
        # Clean shutdown
        logger.info("Shutting down event processor...")
        await consumer.stop()
        await producer.stop()
        
        # Cancel the consumer task if still running
        if not consumer_task.done():
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event processor shutdown complete")

def main():
    """Main entry point."""
    setup_logging()
    logger.info("Starting Astroshield event processor")
    
    try:
        asyncio.run(run_event_processor())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Error in event processor: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 