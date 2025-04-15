"""
TMDB (Target Model Database) Handler

This module provides real-time monitoring and processing of TMDB object updates
and insertions, as discussed in the technical meetings.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)

class TMDBUpdateHandler:
    """Handler for real-time updates from the Target Model Database"""
    
    def __init__(self):
        """Initialize the TMDB update handler"""
        self.event_bus = EventBus()
        self.callbacks = {
            "object_inserted": [],
            "object_updated": [],
            "attribute_changed": [],
            "maneuver_detected": []  # New callback type for maneuvers
        }
        self.red_objects = set()  # Set of object IDs that are flagged as "red"
        self.is_running = False
        
        # Track recent maneuvers
        self.recent_maneuvers = []
        self.max_recent_maneuvers = 50  # Store up to 50 recent maneuvers
    
    def start(self):
        """Start monitoring TMDB updates"""
        if self.is_running:
            logger.warning("TMDB update handler is already running")
            return
            
        logger.info("Starting TMDB update handler")
        
        # Subscribe to TMDB update topics as mentioned in the meeting
        self.event_bus.subscribe("tmdb.objects.inserted", self._handle_object_inserted)
        self.event_bus.subscribe("tmdb.objects.updated", self._handle_object_updated)
        self.event_bus.subscribe("tmdb.objects.attribute", self._handle_attribute_changed)
        
        # Subscribe to red object list updates
        self.event_bus.subscribe("tmdb.red_objects", self._handle_red_objects_update)
        
        # Subscribe to maneuvers-detected topic (mentioned by multiple teams)
        self.event_bus.subscribe("maneuvers-detected", self._handle_maneuver_detected)
        
        self.is_running = True
        logger.info("TMDB update handler is now running")
    
    def stop(self):
        """Stop monitoring TMDB updates"""
        if not self.is_running:
            logger.warning("TMDB update handler is not running")
            return
            
        logger.info("Stopping TMDB update handler")
        
        # Unsubscribe from topics
        self.event_bus.unsubscribe("tmdb.objects.inserted")
        self.event_bus.unsubscribe("tmdb.objects.updated")
        self.event_bus.unsubscribe("tmdb.objects.attribute")
        self.event_bus.unsubscribe("tmdb.red_objects")
        self.event_bus.unsubscribe("maneuvers-detected")
        
        self.is_running = False
        logger.info("TMDB update handler has been stopped")
    
    def register_callback(self, event_type: str, callback: Callable[[Dict[str, Any], Dict[str, Any]], None]):
        """
        Register a callback function for a specific event type
        
        Args:
            event_type: Type of event ('object_inserted', 'object_updated', 'attribute_changed', 'maneuver_detected')
            callback: Callback function that takes (event_data, headers) as arguments
        """
        if event_type not in self.callbacks:
            raise ValueError(f"Invalid event type: {event_type}")
            
        self.callbacks[event_type].append(callback)
        logger.info(f"Registered callback for event type: {event_type}")
    
    def _handle_object_inserted(self, data: Dict[str, Any], headers: Dict[str, Any]):
        """Handle object insertion event"""
        object_id = data.get("object_id")
        object_type = data.get("object_type")
        
        logger.info(f"TMDB object inserted: {object_id} (Type: {object_type})")
        
        # Check if this is a red object
        if object_id in self.red_objects:
            logger.warning(f"Inserted object {object_id} is in the red objects list")
        
        # Execute registered callbacks
        for callback in self.callbacks["object_inserted"]:
            try:
                callback(data, headers)
            except Exception as e:
                logger.error(f"Error in object_inserted callback: {str(e)}")
    
    def _handle_object_updated(self, data: Dict[str, Any], headers: Dict[str, Any]):
        """Handle object update event"""
        object_id = data.get("object_id")
        update_type = data.get("update_type")
        
        logger.info(f"TMDB object updated: {object_id} (Update: {update_type})")
        
        # Check if this is a red object
        if object_id in self.red_objects:
            logger.warning(f"Updated object {object_id} is in the red objects list")
        
        # Execute registered callbacks
        for callback in self.callbacks["object_updated"]:
            try:
                callback(data, headers)
            except Exception as e:
                logger.error(f"Error in object_updated callback: {str(e)}")
    
    def _handle_attribute_changed(self, data: Dict[str, Any], headers: Dict[str, Any]):
        """Handle attribute change event"""
        object_id = data.get("object_id")
        attribute = data.get("attribute")
        
        logger.info(f"TMDB object attribute changed: {object_id} (Attribute: {attribute})")
        
        # Check if this is a red object
        if object_id in self.red_objects:
            logger.warning(f"Object {object_id} with changed attribute is in the red objects list")
        
        # Execute registered callbacks
        for callback in self.callbacks["attribute_changed"]:
            try:
                callback(data, headers)
            except Exception as e:
                logger.error(f"Error in attribute_changed callback: {str(e)}")
    
    def _handle_red_objects_update(self, data: Dict[str, Any], headers: Dict[str, Any]):
        """Handle updates to the red objects list"""
        if "objects" not in data:
            logger.error("Red objects update missing 'objects' field")
            return
            
        # Update the red objects set
        self.red_objects = set(data["objects"])
        
        logger.info(f"Updated red objects list with {len(self.red_objects)} objects")
    
    def _handle_maneuver_detected(self, data: Dict[str, Any], headers: Dict[str, Any]):
        """
        Handle maneuver detection events from the maneuvers-detected topic
        that multiple teams mentioned in the meeting
        """
        object_id = data.get("object_id", data.get("satellite_id", data.get("spacecraft_id")))
        maneuver_type = data.get("maneuver_type", data.get("type", "UNKNOWN"))
        confidence = data.get("confidence", 0.0)
        
        if not object_id:
            logger.error("Maneuver detection missing object identifier")
            return
            
        timestamp = data.get("timestamp", data.get("detection_time", datetime.utcnow().isoformat()))
        
        logger.info(f"Maneuver detected: {object_id} (Type: {maneuver_type}, Confidence: {confidence:.2f})")
        
        # Check if this is a red object
        if object_id in self.red_objects:
            logger.warning(f"Maneuvering object {object_id} is in the red objects list")
        
        # Store in recent maneuvers
        maneuver_record = {
            "object_id": object_id,
            "maneuver_type": maneuver_type,
            "confidence": confidence,
            "timestamp": timestamp,
            "source": headers.get("source", "UNKNOWN"),
            "details": data.get("details", {}),
            "raw_data": data
        }
        
        self.recent_maneuvers.insert(0, maneuver_record)
        
        # Trim list if needed
        if len(self.recent_maneuvers) > self.max_recent_maneuvers:
            self.recent_maneuvers = self.recent_maneuvers[:self.max_recent_maneuvers]
        
        # Execute registered callbacks
        for callback in self.callbacks["maneuver_detected"]:
            try:
                callback(data, headers)
            except Exception as e:
                logger.error(f"Error in maneuver_detected callback: {str(e)}")
                
        # Check if this maneuver warrants an object update in TMDB
        if confidence >= 0.75:  # High confidence maneuvers
            self._update_object_with_maneuver(object_id, maneuver_record)
    
    def _update_object_with_maneuver(self, object_id: str, maneuver_data: Dict[str, Any]):
        """
        Update TMDB object information based on detected maneuvers
        
        Args:
            object_id: The object identifier
            maneuver_data: Maneuver detection data
        """
        logger.info(f"Updating TMDB object {object_id} with maneuver information")
        
        # In a real implementation, this would update TMDB via API
        # For now, log the intent
        
        # Example: Create attribute update to publish to TMDB
        update_data = {
            "object_id": object_id,
            "attribute": "last_maneuver",
            "value": {
                "timestamp": maneuver_data["timestamp"],
                "type": maneuver_data["maneuver_type"],
                "confidence": maneuver_data["confidence"],
                "source": maneuver_data["source"],
                "details": maneuver_data.get("details", {})
            }
        }
        
        # In a real implementation, would publish to TMDB update topic
        logger.info(f"Would update TMDB with: {update_data}")
    
    def get_recent_maneuvers(self, object_id: Optional[str] = None, 
                          max_count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent maneuvers, optionally filtered by object ID
        
        Args:
            object_id: Optional object ID to filter by
            max_count: Maximum number of maneuvers to return
            
        Returns:
            List of recent maneuvers
        """
        if object_id:
            filtered = [m for m in self.recent_maneuvers if m["object_id"] == object_id]
            return filtered[:max_count]
        
        return self.recent_maneuvers[:max_count]
    
    def is_red_object(self, object_id: str) -> bool:
        """Check if an object is in the red objects list"""
        return object_id in self.red_objects
    
    def get_red_objects(self) -> List[str]:
        """Get the current list of red objects"""
        return list(self.red_objects)
        
    def fetch_object_details(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch details for a specific object from TMDB
        
        In a real implementation, this would query the TMDB API
        """
        logger.info(f"Fetching details for object: {object_id}")
        
        # This would be replaced with an actual API call
        # Return placeholder data for now
        
        if not object_id:
            return None
            
        # Simulated data
        return {
            "object_id": object_id,
            "name": f"Object {object_id}",
            "type": "SATELLITE",
            "status": "ACTIVE",
            "country": "USA" if object_id.startswith("USA") else "RUS" if object_id.startswith("COSMOS") else "PRC",
            "launch_date": "2022-01-01T00:00:00Z",
            "orbit": {
                "apogee": 35786,
                "perigee": 35784,
                "inclination": 0.1,
                "period": 1436
            },
            "last_updated": datetime.utcnow().isoformat(),
            "recent_maneuvers": self.get_recent_maneuvers(object_id, 3)
        } 