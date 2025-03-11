"""
Vantiq Adapter Module

This module provides an adapter for interacting with the Vantiq platform.
"""

import json
import logging
import requests
import time
from typing import Dict, Any, List, Optional, Union

from backend.config.vantiq_config import get_vantiq_config

logger = logging.getLogger(__name__)

class VantiqAdapter:
    """
    Adapter for interacting with the Vantiq platform
    """
    
    def __init__(self):
        """
        Initialize the Vantiq adapter
        """
        self.config = get_vantiq_config()
        self.api_url = self.config["api_url"]
        self.api_token = self.config["api_token"]
        self.namespace = self.config["namespace"]
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self.connected = False
        
    async def connect(self) -> bool:
        """
        Connect to the Vantiq platform
        """
        try:
            response = self.session.get(f"{self.api_url}/system/status")
            if response.status_code == 200:
                self.connected = True
                logger.info("Successfully connected to Vantiq")
                return True
            else:
                logger.error(f"Failed to connect to Vantiq: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Vantiq: {str(e)}")
            return False
            
    async def disconnect(self) -> None:
        """
        Disconnect from the Vantiq platform
        """
        self.session.close()
        self.connected = False
        logger.info("Disconnected from Vantiq")
        
    async def publish_event(self, topic: str, payload: Dict[str, Any]) -> bool:
        """
        Publish an event to a Vantiq topic
        """
        try:
            if not self.connected:
                await self.connect()
                
            url = f"{self.api_url}/resources/topics/{topic}/publish"
            response = self.session.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Successfully published event to topic {topic}")
                return True
            else:
                logger.error(f"Failed to publish event to topic {topic}: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error publishing event to topic {topic}: {str(e)}")
            return False
            
    async def query_resource(self, resource_type: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query a Vantiq resource
        """
        try:
            if not self.connected:
                await self.connect()
                
            url = f"{self.api_url}/resources/{resource_type}/query"
            response = self.session.post(url, json=query)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to query resource {resource_type}: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error querying resource {resource_type}: {str(e)}")
            return []
            
    async def execute_procedure(self, procedure: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute a Vantiq procedure
        """
        try:
            if not self.connected:
                await self.connect()
                
            url = f"{self.api_url}/resources/procedures/{procedure}/execute"
            response = self.session.post(url, json=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to execute procedure {procedure}: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error executing procedure {procedure}: {str(e)}")
            return None
            
    async def send_command(self, command_type: str, params: Dict[str, Any]) -> bool:
        """
        Send a command to Vantiq
        """
        try:
            command_topic = self.config["topics"]["commands"]
            payload = {
                "type": command_type,
                "params": params,
                "timestamp": int(time.time() * 1000)
            }
            
            return await self.publish_event(command_topic, payload)
        except Exception as e:
            logger.error(f"Error sending command {command_type}: {str(e)}")
            return False 