#!/usr/bin/env python3
"""
UDL Integration Example

This script demonstrates how to use the AstroShield UDL Integration package
to interact with the Unified Data Library (UDL) API.
"""

import os
import sys
import time
import logging
import json
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Add the package directory to the path
sys.path.append('/Users/jackal-kahwati/Library/Mobile Documents/com~apple~CloudDocs/ProjectBackups/asttroshield_v0/astroshield-integration-package/src')

# Import the UDL integration components
from asttroshield.udl_integration.client import UDLClient
from asttroshield.udl_integration.messaging_client import UDLMessagingClient
from asttroshield.udl_integration.integration import UDLIntegration

def example_rest_api():
    """Example of using the UDL REST API client"""
    
    # Get credentials from environment variables
    username = os.environ.get('UDL_USERNAME')
    password = os.environ.get('UDL_PASSWORD')
    
    if not username or not password:
        logger.error("UDL_USERNAME and UDL_PASSWORD environment variables must be set")
        return False
    
    # Create the UDL client
    client = UDLClient(
        base_url="https://unifieddatalibrary.com",
        username=username,
        password=password
    )
    
    # Example 1: Get state vectors
    try:
        logger.info("Example 1: Getting state vectors...")
        state_vectors = client.get_state_vectors(epoch="now", maxResults=5)
        logger.info(f"Retrieved {len(state_vectors)} state vectors")
        
        # Print the first state vector if available
        if state_vectors:
            logger.info(f"First state vector: {json.dumps(state_vectors[0], indent=2)}")
    except Exception as e:
        logger.error(f"Error getting state vectors: {str(e)}")
    
    # Example 2: Get conjunctions
    try:
        logger.info("Example 2: Getting conjunctions...")
        # Add required parameters for conjunctions API
        conjunctions = client.get_conjunctions(
            epoch="now",  # Required parameter
            maxResults=5
        )
        logger.info(f"Retrieved {len(conjunctions)} conjunctions")
        
        # Print the first conjunction if available
        if conjunctions:
            logger.info(f"First conjunction: {json.dumps(conjunctions[0], indent=2)}")
    except Exception as e:
        logger.error(f"Error getting conjunctions: {str(e)}")
    
    # Example 3: Get launch events
    try:
        logger.info("Example 3: Getting launch events...")
        # Add required parameters for launch events API
        from datetime import datetime, timedelta
        
        # Get date 30 days ago in ISO format
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        
        launch_events = client.get_launch_events(
            msg_create_date=thirty_days_ago,  # Required parameter
            maxResults=5
        )
        logger.info(f"Retrieved {len(launch_events)} launch events")
        
        # Print the first launch event if available
        if launch_events:
            logger.info(f"First launch event: {json.dumps(launch_events[0], indent=2)}")
    except Exception as e:
        logger.error(f"Error getting launch events: {str(e)}")
    
    return True

def example_secure_messaging():
    """Example of using the UDL Secure Messaging API client"""
    
    # Get credentials from environment variables
    username = os.environ.get('UDL_USERNAME')
    password = os.environ.get('UDL_PASSWORD')
    
    if not username or not password:
        logger.error("UDL_USERNAME and UDL_PASSWORD environment variables must be set")
        return False
    
    # Create the UDL Messaging client
    try:
        client = UDLMessagingClient(
            base_url="https://unifieddatalibrary.com",
            username=username,
            password=password
        )
    except Exception as e:
        logger.error(f"Error creating UDL Messaging client: {str(e)}")
        logger.warning("Make sure you have proper authorization for the Secure Messaging API")
        return False
    
    # Example 1: List available topics
    try:
        logger.info("Example 1: Listing available topics...")
        topics = client.list_topics()
        logger.info(f"Retrieved {len(topics)} topics")
        
        # Print the topics if available
        if topics:
            for topic in topics:
                logger.info(f"Topic: {topic}")
    except Exception as e:
        logger.error(f"Error listing topics: {str(e)}")
        logger.warning("This error may occur if you don't have access to the Secure Messaging API")
        return False
    
    # If we have topics, try to get messages from the first one
    if topics:
        first_topic = topics[0]['name']
        
        # Example 2: Get the latest offset for a topic
        try:
            logger.info(f"Example 2: Getting latest offset for topic {first_topic}...")
            offset = client.get_latest_offset(first_topic)
            logger.info(f"Latest offset for {first_topic}: {offset}")
        except Exception as e:
            logger.error(f"Error getting latest offset: {str(e)}")
        
        # Example 3: Get messages from a topic
        try:
            logger.info(f"Example 3: Getting messages from topic {first_topic}...")
            messages, next_offset = client.get_messages(first_topic, offset=0)
            logger.info(f"Retrieved {len(messages)} messages, next offset: {next_offset}")
            
            # Print the first message if available
            if messages:
                logger.info(f"First message: {json.dumps(messages[0], indent=2)}")
        except Exception as e:
            logger.error(f"Error getting messages: {str(e)}")
        
        # Example 4: Start a consumer for a topic
        def message_callback(messages):
            logger.info(f"Received {len(messages)} messages")
            for msg in messages:
                logger.info(f"Message: {json.dumps(msg, indent=2)}")
        
        try:
            logger.info(f"Example 4: Starting consumer for topic {first_topic}...")
            client.start_consumer(first_topic, message_callback, start_from_latest=True)
            
            # Let the consumer run for a few seconds
            logger.info("Consumer running for 10 seconds...")
            time.sleep(10)
            
            # Stop the consumer
            client.stop_consumer(first_topic)
            logger.info("Consumer stopped")
        except Exception as e:
            logger.error(f"Error with consumer: {str(e)}")
    
    return True

def example_integration():
    """Example of using the UDL Integration class"""
    
    # Get credentials from environment variables
    username = os.environ.get('UDL_USERNAME')
    password = os.environ.get('UDL_PASSWORD')
    
    if not username or not password:
        logger.error("UDL_USERNAME and UDL_PASSWORD environment variables must be set")
        return False
    
    # Create the UDL Integration
    try:
        integration = UDLIntegration(
            udl_base_url="https://unifieddatalibrary.com",
            udl_username=username,
            udl_password=password,
            kafka_bootstrap_servers="localhost:9092",  # Replace with your Kafka server
            use_secure_messaging=True  # Set to False if you don't have Secure Messaging access
        )
    except Exception as e:
        logger.error(f"Error creating UDL Integration: {str(e)}")
        return False
    
    # Example 1: Process state vectors
    try:
        logger.info("Example 1: Processing state vectors...")
        integration.process_state_vectors(epoch="now")
    except Exception as e:
        logger.error(f"Error processing state vectors: {str(e)}")
    
    # Example 2: Process conjunctions
    try:
        logger.info("Example 2: Processing conjunctions...")
        integration.process_conjunctions()
    except Exception as e:
        logger.error(f"Error processing conjunctions: {str(e)}")
    
    # Example 3: Process launch events
    try:
        logger.info("Example 3: Processing launch events...")
        integration.process_launch_events()
    except Exception as e:
        logger.error(f"Error processing launch events: {str(e)}")
    
    return True

if __name__ == "__main__":
    logger.info("=== UDL REST API Example ===")
    example_rest_api()
    
    logger.info("\n=== UDL Secure Messaging API Example ===")
    logger.info("Note: This example requires special authorization for the Secure Messaging API")
    example_secure_messaging()
    
    logger.info("\n=== UDL Integration Example ===")
    logger.info("Note: This example requires a running Kafka server")
    # Uncomment the line below to run the integration example
    # example_integration() 