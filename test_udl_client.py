#!/usr/bin/env python3
"""
Test script for UDL client with Basic Authentication
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Add the package directory to the path
sys.path.append('/Users/jackal-kahwati/Library/Mobile Documents/com~apple~CloudDocs/ProjectBackups/asttroshield_v0/astroshield-integration-package/src')

# Import the UDL client
from asttroshield.udl_integration.client import UDLClient

def test_udl_client():
    """Test the UDL client with Basic Authentication"""
    
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
    
    # Test a simple API call
    try:
        # Get state vectors with the 'now' epoch parameter
        logger.info("Testing state vector retrieval...")
        state_vectors = client.get_state_vectors(epoch="now", maxResults=5)
        logger.info(f"Successfully retrieved {len(state_vectors)} state vectors")
        
        # Print the first state vector if available
        if state_vectors:
            logger.info(f"First state vector: {state_vectors[0]}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing UDL client: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_udl_client()
    if success:
        logger.info("UDL client test completed successfully")
        sys.exit(0)
    else:
        logger.error("UDL client test failed")
        sys.exit(1) 