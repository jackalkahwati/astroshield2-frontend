"""
UDL Client Service for AstroShield.

This module provides a client for communicating with the Unified Data Library (UDL).
It supports both real UDL instances and a mock UDL for development.
"""
import os
import logging
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel
from app.core.error_handling import UDLServiceException

# Configure logging
logger = logging.getLogger(__name__)

class StateVector(BaseModel):
    """State vector model representing satellite position and velocity"""
    id: str
    name: str
    epoch: str
    position: Dict[str, float]
    velocity: Dict[str, float]

class UDLClient:
    """Client for interacting with the Unified Data Library"""
    
    def __init__(self):
        """Initialize the UDL client with configuration from environment variables"""
        self.username = os.environ.get("UDL_USERNAME")
        self.password = os.environ.get("UDL_PASSWORD")
        self.base_url = os.environ.get("UDL_BASE_URL", "https://unifieddatalibrary.com/udl/api/v1")
        self.session = requests.Session()
        self.token = None
        self.token_expiry = None
        
        # Determine if we're using mock UDL
        self.mock_mode = self.base_url.startswith(("http://localhost", "https://mock"))
        if self.mock_mode:
            logger.info("UDL client initialized in mock mode")
        else:
            logger.info(f"UDL client initialized with base URL: {self.base_url}")
    
    async def authenticate(self) -> bool:
        """Authenticate with UDL to get an access token"""
        if not self.username or not self.password:
            raise UDLServiceException("UDL credentials not configured")
        
        try:
            # Determine the correct auth endpoint
            if self.mock_mode:
                # For mock server
                auth_url = self.base_url.replace("/api/v1", "") + "/auth/token"
                if not auth_url.startswith(("http://", "https://")):
                    # Handle case where base_url might be incomplete
                    auth_url = "http://localhost:8888/auth/token"
            else:
                # For real UDL
                auth_url = self.base_url.replace("/udl/api/v1", "") + "/auth/token"
            
            logger.debug(f"Authenticating to UDL at {auth_url}")
            
            auth_response = self.session.post(
                auth_url,
                json={"username": self.username, "password": self.password},
                timeout=30,
            )
            
            if auth_response.status_code != 200:
                logger.error(f"Authentication failed with status code {auth_response.status_code}")
                logger.error(f"Response: {auth_response.text}")
                return False
            
            token_data = auth_response.json()
            if "token" not in token_data:
                logger.error("No token in authentication response")
                return False
            
            self.token = token_data["token"]
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            logger.info("Successfully authenticated with UDL")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to UDL: {str(e)}")
            
            if self.mock_mode:
                # In mock mode, create a fake token
                logger.warning("Using mock authentication token")
                self.token = "mock-token-for-testing"
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                return True
                
            raise UDLServiceException(f"Failed to connect to UDL: {str(e)}")
    
    async def get_state_vectors(self, limit: int = 10) -> List[StateVector]:
        """Get state vectors from UDL"""
        if not self.token:
            await self.authenticate()
        
        try:
            if self.mock_mode:
                # For mock server
                url = "http://localhost:8888/statevector"
            else:
                # For real UDL
                url = f"{self.base_url}/statevector"
            
            params = {"epoch": "now", "limit": limit}
            
            logger.debug(f"Getting state vectors from UDL: {url}")
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Failed to get state vectors: {response.status_code}")
                logger.error(f"Response: {response.text}")
                
                if self.mock_mode:
                    # Return mock data in case of failure
                    return self._get_mock_state_vectors(limit)
                    
                raise UDLServiceException(f"Failed to get state vectors: {response.text}")
            
            data = response.json()
            
            # Handle different response formats
            if "stateVectors" in data:
                vectors_data = data["stateVectors"]
            elif isinstance(data, list):
                vectors_data = data
            else:
                logger.warning(f"Unexpected response format: {data}")
                
                if self.mock_mode:
                    return self._get_mock_state_vectors(limit)
                    
                raise UDLServiceException("Unexpected response format from UDL")
            
            # Parse into StateVector objects
            state_vectors = []
            for vector_data in vectors_data[:limit]:
                try:
                    state_vector = StateVector(**vector_data)
                    state_vectors.append(state_vector)
                except Exception as e:
                    logger.warning(f"Error parsing state vector: {str(e)}")
            
            return state_vectors
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting state vectors from UDL: {str(e)}")
            
            if self.mock_mode:
                # Return mock data in case of connection issues
                return self._get_mock_state_vectors(limit)
                
            raise UDLServiceException(f"Failed to get state vectors from UDL: {str(e)}")
    
    def _get_mock_state_vectors(self, limit: int = 10) -> List[StateVector]:
        """Generate mock state vectors for testing"""
        import random
        
        logger.info(f"Generating {limit} mock state vectors")
        
        return [
            StateVector(
                id=f"sat-{i}",
                name=f"Test Satellite {i}",
                epoch=datetime.utcnow().isoformat(),
                position={
                    "x": random.uniform(-7000, 7000),
                    "y": random.uniform(-7000, 7000),
                    "z": random.uniform(-7000, 7000)
                },
                velocity={
                    "x": random.uniform(-7, 7),
                    "y": random.uniform(-7, 7),
                    "z": random.uniform(-7, 7)
                }
            )
            for i in range(1, limit + 1)
        ]
        
    async def close(self):
        """Close the session"""
        if self.session:
            self.session.close()

# Factory for getting the UDL client
udl_client_instance = None

async def get_udl_client() -> UDLClient:
    """Get a UDL client instance (singleton)"""
    global udl_client_instance
    
    if udl_client_instance is None:
        udl_client_instance = UDLClient()
        
        # Try to authenticate, but don't fail if it doesn't work (mock mode fallback)
        try:
            await udl_client_instance.authenticate()
        except Exception as e:
            logger.warning(f"Initial UDL authentication failed: {str(e)}")
    
    return udl_client_instance 