"""
Kafka event handlers for Astroshield.
Each handler is responsible for processing a specific type of event.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional

from app.ccdm.service import DMDOrbitDeterminationClient, query_dmd_orbit_determination
from app.common.logging import logger
from app.common.weather_integration import WeatherDataService
from app.kafka.producer import KafkaProducer

class EventHandler:
    """Base class for Kafka event handlers."""
    
    def __init__(self, producer: Optional[KafkaProducer] = None):
        """Initialize the event handler."""
        self.producer = producer
    
    async def handle_event(self, event: Dict[str, Any]) -> None:
        """
        Base method for handling events.
        Should be overridden by subclasses.
        
        Args:
            event: The event to handle
        """
        raise NotImplementedError("Subclasses must implement handle_event method")


class DMDOrbitDeterminationEventHandler(EventHandler):
    """Event handler for DMD orbit determination events."""
    
    def __init__(self, producer: Optional[KafkaProducer] = None):
        """Initialize the DMD orbit determination event handler."""
        super().__init__(producer)
        self.dmd_client = DMDOrbitDeterminationClient()
    
    async def handle_event(self, event: Dict[str, Any]) -> None:
        """
        Handle DMD orbit determination events.
        Specifically designed to trigger maneuver detection when new 
        orbit determination data is available.
        
        Args:
            event: The event to handle, containing DMD catalog ID
        """
        # Extract the DMD catalog ID from the event
        # Depending on the event format, this may need to be adjusted
        try:
            # Extract catalog ID from event payload
            payload = event.get("payload", {})
            if not payload:
                logger.warning("No payload in DMD orbit determination event")
                return
            
            catalog_id = payload.get("object_id") or payload.get("catalogId") or payload.get("dmd_catalog_id")
            
            if not catalog_id:
                logger.warning("No catalog ID found in DMD orbit determination event")
                return
            
            # Log the event
            logger.info(f"Processing DMD orbit determination event for object: {catalog_id}")
            
            # Call the DMD client to detect maneuvers
            maneuver_result = await self.dmd_client.detect_maneuvers_from_states(catalog_id)
            
            # Check if a maneuver was detected
            if maneuver_result.get("detected", False):
                logger.info(f"Maneuver detected for {catalog_id}: {maneuver_result}")
                
                # If a maneuver was detected, publish a maneuver event
                if self.producer:
                    await self.publish_maneuver_event(maneuver_result)
            else:
                logger.info(f"No maneuver detected for {catalog_id}: {maneuver_result.get('reason', 'unknown reason')}")
        
        except Exception as e:
            logger.error(f"Error handling DMD orbit determination event: {str(e)}")
    
    async def publish_maneuver_event(self, maneuver_data: Dict[str, Any]) -> None:
        """
        Publish a maneuver event to Kafka.
        
        Args:
            maneuver_data: Data about the detected maneuver
        """
        if not self.producer:
            logger.warning("No Kafka producer available to publish maneuver event")
            return
        
        # Prepare the maneuver event
        maneuver_event = {
            "header": {
                "messageType": "maneuver-detected",
                "source": "dmd-od-integration",
                "timestamp": maneuver_data.get("time")
            },
            "payload": {
                "catalogId": maneuver_data.get("catalog_id"),
                "deltaV": maneuver_data.get("delta_v"),
                "confidence": maneuver_data.get("confidence"),
                "maneuverType": maneuver_data.get("maneuver_type"),
                "detectionTime": maneuver_data.get("time")
            }
        }
        
        # Publish the event
        try:
            await self.producer.send_async("maneuvers-detected", maneuver_event)
            logger.info(f"Published maneuver event for {maneuver_data.get('catalog_id')}")
        except Exception as e:
            logger.error(f"Error publishing maneuver event: {str(e)}")


class WeatherDataEventHandler(EventHandler):
    """Event handler for weather data events."""
    
    def __init__(self, producer: Optional[KafkaProducer] = None):
        """Initialize the weather data event handler."""
        super().__init__(producer)
        self.weather_service = WeatherDataService()
    
    async def handle_event(self, event: Dict[str, Any]) -> None:
        """
        Handle weather data events.
        Processes weather data and integrates it into analysis as needed.
        
        Args:
            event: The weather data event
        """
        try:
            # Extract weather data from the event
            payload = event.get("payload", {})
            if not payload:
                logger.warning("No payload in weather data event")
                return
            
            # Log the event
            logger.info("Processing weather data event")
            
            # Extract object info if available
            object_info = None
            if "targetObject" in payload:
                object_info = {
                    "catalog_id": payload["targetObject"].get("catalogId", "UNKNOWN"),
                    "altitude_km": payload["targetObject"].get("altitude", 0.0),
                    "type": payload["targetObject"].get("objectType", "UNKNOWN")
                }
            
            # Analyze observation conditions
            analysis_result = self.weather_service.analyze_observation_conditions(payload, object_info)
            
            # If analysis indicates favorable conditions, publish an event
            if analysis_result["observation_quality"]["recommendation"] == "GO":
                await self.publish_observation_recommendation(analysis_result)
        
        except Exception as e:
            logger.error(f"Error handling weather data event: {str(e)}")
    
    async def publish_observation_recommendation(self, analysis_result: Dict[str, Any]) -> None:
        """
        Publish an observation recommendation event based on weather analysis.
        
        Args:
            analysis_result: Results from the weather analysis
        """
        if not self.producer:
            logger.warning("No Kafka producer available to publish observation recommendation")
            return
        
        # Prepare the recommendation event
        recommendation_event = {
            "header": {
                "messageType": "observation-window-recommended",
                "source": "weather-integration",
                "timestamp": analysis_result["analysis_time"]
            },
            "payload": {
                "location": analysis_result["location"],
                "qualityScore": analysis_result["observation_quality"]["score"],
                "qualityCategory": analysis_result["observation_quality"]["category"],
                "recommendation": analysis_result["observation_quality"]["recommendation"]
            }
        }
        
        # Add observation window if available
        if "observation_window" in analysis_result:
            recommendation_event["payload"]["observationWindow"] = analysis_result["observation_window"]
        
        # Add object info if available
        if "object_info" in analysis_result:
            recommendation_event["payload"]["targetObject"] = analysis_result["object_info"]
        
        # Publish the event
        try:
            await self.producer.send_async("observation-windows", recommendation_event)
            logger.info(f"Published observation recommendation with quality {analysis_result['observation_quality']['category']}")
        except Exception as e:
            logger.error(f"Error publishing observation recommendation: {str(e)}")


# Factory to create event handlers based on message type
def create_event_handler(message_type: str, producer: Optional[KafkaProducer] = None) -> Optional[EventHandler]:
    """
    Create an event handler based on the message type.
    
    Args:
        message_type: Type of message to handle
        producer: Optional Kafka producer for publishing events
        
    Returns:
        EventHandler instance or None if no handler is available
    """
    handlers = {
        "dmd-od-update": DMDOrbitDeterminationEventHandler(producer),
        "weather-data": WeatherDataEventHandler(producer)
    }
    
    return handlers.get(message_type) 