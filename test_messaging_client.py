#!/usr/bin/env python3
"""
Test script for UDL Secure Messaging client with Basic Authentication
"""

import os
import sys
import logging
import time
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Add the package directory to the path
sys.path.append('/Users/jackal-kahwati/Library/Mobile Documents/com~apple~CloudDocs/ProjectBackups/asttroshield_v0/astroshield-integration-package/src')

# Import the UDL Messaging client
from asttroshield.udl_integration.messaging_client import UDLMessagingClient

def message_callback(messages):
    """Callback function for received messages"""
    logger.info(f"Received {len(messages)} messages")
    for msg in messages:
        logger.info(f"Message: {msg}")

def test_messaging_client():
    """Test the UDL Secure Messaging client with Basic Authentication"""
    
    # Get credentials from environment variables
    username = os.environ.get('UDL_USERNAME')
    password = os.environ.get('UDL_PASSWORD')
    
    if not username or not password:
        logger.error("UDL_USERNAME and UDL_PASSWORD environment variables must be set")
        return False
    
    # Create the UDL Messaging client
    client = UDLMessagingClient(
        base_url="https://unifieddatalibrary.com",
        username=username,
        password=password
    )
    
    # Test listing topics
    try:
        logger.info("Testing topic listing...")
        topics = client.list_topics()
        logger.info(f"Successfully retrieved {len(topics)} topics")
        
        # Print the topics if available
        if topics:
            for topic in topics:
                logger.info(f"Topic: {topic}")
            
            # If topics are available, try to get the latest offset for the first topic
            first_topic = topics[0]['name']
            logger.info(f"Testing latest offset for topic: {first_topic}")
            try:
                offset = client.get_latest_offset(first_topic)
                logger.info(f"Latest offset for {first_topic}: {offset}")
                
                # Try to get messages from the topic
                logger.info(f"Testing message retrieval for topic: {first_topic}")
                messages, next_offset = client.get_messages(first_topic, offset=0)
                logger.info(f"Retrieved {len(messages)} messages, next offset: {next_offset}")
                
                # Start a consumer for a short time
                logger.info(f"Testing consumer for topic: {first_topic}")
                client.start_consumer(first_topic, message_callback, start_from_latest=True)
                
                # Let the consumer run for a few seconds
                logger.info("Consumer running for 5 seconds...")
                time.sleep(5)
                
                # Stop the consumer
                client.stop_consumer(first_topic)
                logger.info("Consumer stopped")
                
            except Exception as e:
                logger.error(f"Error testing topic operations: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing UDL Messaging client: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_messaging_client()
    if success:
        logger.info("UDL Messaging client test completed successfully")
        sys.exit(0)
    else:
        logger.error("UDL Messaging client test failed")
        sys.exit(1) 