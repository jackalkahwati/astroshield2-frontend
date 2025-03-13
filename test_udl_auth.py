#!/usr/bin/env python3
"""
Standalone Test script for UDL authentication.

This script tests UDL authentication using credentials from the .env file.
It implements a simple version of the UDL client to make an authentication request.
"""

import json
import logging
import os
import sys
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def test_udl_auth():
    """Test UDL authentication using credentials from .env file."""
    logger.info("Testing UDL authentication with credentials from .env")
    
    # Log environment variable status (without revealing full credentials)
    username = os.environ.get("UDL_USERNAME", "")
    password = os.environ.get("UDL_PASSWORD", "")
    base_url = os.environ.get("UDL_BASE_URL", "")
    
    logger.info(f"UDL_USERNAME: {'✓' if username else '✗'} (value: {username[:2]}{'*' * (len(username) - 4)}{username[-2:] if len(username) > 3 else ''})")
    logger.info(f"UDL_PASSWORD: {'✓' if password else '✗'} (length: {len(password)})")
    logger.info(f"UDL_BASE_URL: {'✓' if base_url else '✗'} (value: {base_url})")
    
    if not (username and password):
        logger.error("Required environment variables not found. Please check your .env file.")
        return False
    
    try:
        # Create a session with retries
        session = requests.Session()
        retries = requests.adapters.Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Get a token using username/password authentication
        # Important: The auth endpoint is at the same level as the API version, not in /udl/api/v1
        auth_url = f"{base_url}/auth/token"
        
        # Fix the URL if it contains '/udl/api/v1'
        if '/udl/api/v1' in auth_url:
            auth_url = auth_url.replace('/udl/api/v1', '')
            
        logger.info(f"Authenticating to UDL at {auth_url}")
        
        auth_response = session.post(
            auth_url,
            json={"username": username, "password": password},
            timeout=30,
        )
        
        if auth_response.status_code != 200:
            logger.error(f"Authentication failed with status code {auth_response.status_code}")
            logger.error(f"Response: {auth_response.text}")
            return False
        
        token_data = auth_response.json()
        if "token" not in token_data:
            logger.error("No token in authentication response")
            logger.error(f"Response: {auth_response.text}")
            return False
        
        token = token_data["token"]
        logger.info("Successfully obtained authentication token")
        
        # Try to make a simple API call to test the token
        headers = {"Authorization": f"Bearer {token}"}
        
        # Fix the state vectors URL based on client implementation
        state_vectors_url = f"{base_url}/statevector"
        
        logger.info(f"Making test request to {state_vectors_url}")
        response = session.get(
            state_vectors_url,
            headers=headers,
            params={"limit": 1},
            timeout=30,
        )
        
        if response.status_code != 200:
            logger.error(f"API request failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
        
        state_vectors = response.json()
        if not state_vectors:
            logger.warning("No state vectors retrieved. API call worked but no data was returned.")
            logger.info("Authentication successful, but no data available.")
        else:
            logger.info(f"Successfully retrieved {len(state_vectors)} state vector(s)")
            logger.info("First state vector data:")
            logger.info(json.dumps(state_vectors[0], indent=2)[:500] + "...")
        
        logger.info("Authentication successful!")
        return True
            
    except Exception as e:
        logger.error(f"Error authenticating with UDL: {e}")
        return False


if __name__ == "__main__":
    success = test_udl_auth()
    sys.exit(0 if success else 1) 