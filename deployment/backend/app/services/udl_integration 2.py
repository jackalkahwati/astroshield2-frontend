"""UDL Integration Service for Event Processors.

This service handles the integration between UDL data feeds and the event processors.
It subscribes to relevant UDL topics, converts incoming messages to the appropriate
format for event detection, and forwards them to the event processors.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Type
from datetime import datetime
import yaml

# UDL integration imports
try:
    from asttroshield.udl_integration.client import UDLClient
    from asttroshield.udl_integration.messaging_client import UDLMessagingClient
except ImportError:
    # Add fallback for development environment
    import sys
    sys.path.append('/Users/jackal-kahwati/asttroshield_v0 2/astroshield-integration-package/src')
    try:
        from asttroshield.udl_integration.client import UDLClient
        from asttroshield.udl_integration.messaging_client import UDLMessagingClient
    except ImportError:
        # Create stubs for development without UDL
        class UDLClient:
            def __init__(self, *args, **kwargs):
                pass
                
        class UDLMessagingClient:
            def __init__(self, *args, **kwargs):
                pass
                
            def start_consumer(self, *args, **kwargs):
                pass

# Event processor imports
from app.services.event_processor_base import EventProcessorBase
from app.services.processors.launch_processor import LaunchProcessor
from app.services.processors.reentry_processor import ReentryProcessor
from app.services.processors.maneuver_processor import ManeuverProcessor
from app.services.processors.separation_processor import SeparationProcessor
from app.services.processors.proximity_processor import ProximityProcessor
from app.services.processors.link_change_processor import LinkChangeProcessor
from app.services.processors.attitude_change_processor import AttitudeChangeProcessor

logger = logging.getLogger(__name__)

class UDLIntegrationService:
    """Service for integrating UDL data with event processors."""
    
    # UDL topic to processor mapping
    TOPIC_TO_PROCESSOR = {
        "launch": LaunchProcessor,
        "reentry": ReentryProcessor,
        "maneuver": ManeuverProcessor,
        "separation": SeparationProcessor,
        "conjunction": ProximityProcessor,
        "link-status": LinkChangeProcessor,
        "attitude": AttitudeChangeProcessor
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the UDL integration service.
        
        Args:
            config_path: Path to UDL configuration file
        """
        self.config = self._load_config(config_path)
        self.processors = {}
        self.udl_client = None
        self.messaging_client = None
        self.running = False
        self.callback_queue = asyncio.Queue()
        self.event_callback = None
        
        # Initialize UDL clients if configuration is valid
        if self._is_config_valid():
            self._init_udl_clients()
            
        # Initialize processors
        self._init_processors()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load UDL configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config = {
            "base_url": os.environ.get("UDL_BASE_URL", "https://unifieddatalibrary.com"),
            "username": os.environ.get("UDL_USERNAME", ""),
            "password": os.environ.get("UDL_PASSWORD", ""),
            "api_key": os.environ.get("UDL_API_KEY", ""),
            "topics": {
                "launch": "udl.launchevent",
                "reentry": "udl.reentry",
                "maneuver": "udl.maneuver",
                "separation": "udl.separation",
                "conjunction": "udl.conjunction",
                "link-status": "udl.link-status",
                "attitude": "udl.attitude"
            },
            "enabled": False  # Default to disabled
        }
        
        # Try to load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                    
                    # Update config with file values
                    config.update(file_config)
                    logger.info(f"Loaded UDL configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading UDL configuration from {config_path}: {str(e)}")
        
        return config
        
    def _is_config_valid(self) -> bool:
        """
        Check if UDL configuration is valid for connection.
        
        Returns:
            True if valid, False otherwise
        """
        if not self.config.get("enabled", False):
            logger.info("UDL integration is disabled in configuration.")
            return False
            
        if not self.config.get("base_url"):
            logger.warning("UDL base URL not configured.")
            return False
            
        if not (self.config.get("api_key") or 
                (self.config.get("username") and self.config.get("password"))):
            logger.warning("UDL credentials not configured.")
            return False
            
        return True
        
    def _init_udl_clients(self):
        """Initialize UDL clients with configuration."""
        try:
            # Initialize REST API client
            self.udl_client = UDLClient(
                base_url=self.config.get("base_url"),
                api_key=self.config.get("api_key"),
                username=self.config.get("username"),
                password=self.config.get("password")
            )
            
            # Initialize messaging client
            self.messaging_client = UDLMessagingClient(
                base_url=self.config.get("base_url"),
                username=self.config.get("username"),
                password=self.config.get("password")
            )
            
            logger.info("UDL clients initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing UDL clients: {str(e)}")
            self.udl_client = None
            self.messaging_client = None
            
    def _init_processors(self):
        """Initialize event processors."""
        for key, processor_class in self.TOPIC_TO_PROCESSOR.items():
            self.processors[key] = processor_class()
        logger.info(f"Initialized {len(self.processors)} event processors.")
        
    def register_event_callback(self, callback: Callable[[Dict[str, Any], str], None]):
        """
        Register a callback function to receive events.
        
        Args:
            callback: Function to call when an event is detected
        """
        self.event_callback = callback
        logger.info("Registered event callback function.")
        
    async def process_message(self, message: Dict[str, Any], topic_key: str):
        """
        Process a message from UDL.
        
        Args:
            message: Message received from UDL
            topic_key: Key for the topic processor map
        """
        try:
            if topic_key not in self.processors:
                logger.warning(f"No processor found for topic {topic_key}")
                return
                
            processor = self.processors[topic_key]
            
            # Convert message to proper format for detection
            detection_data = self._convert_message_for_detection(message, topic_key)
            if not detection_data:
                return
                
            # Check entry criteria
            detection = await processor.detect_entry_criteria(detection_data)
            if detection:
                logger.info(f"Detected {topic_key} event for object {detection.object_id}")
                
                # Create event
                event = processor.create_event(detection)
                
                # Call event callback if registered
                if self.event_callback:
                    await self.callback_queue.put((event, topic_key))
        except Exception as e:
            logger.error(f"Error processing UDL message for {topic_key}: {str(e)}")
            
    def _convert_message_for_detection(self, message: Dict[str, Any], topic_key: str) -> Dict[str, Any]:
        """
        Convert UDL message to format required by event processor detection.
        
        Args:
            message: Message from UDL
            topic_key: Topic processor key
            
        Returns:
            Dictionary in format expected by event processor
        """
        if not message:
            return None
            
        # Common fields
        result = {
            "object_id": message.get("objectId", message.get("object_id", "")),
            "confidence": 0.9,  # UDL data is considered reliable
            "detection_time": message.get("epoch", datetime.utcnow().isoformat()),
            "sensor_id": message.get("sensorId", message.get("sensor_id", "UDL"))
        }
        
        # Topic-specific conversions
        if topic_key == "launch":
            result.update({
                "launch_site": message.get("launchSite", {}),
                "initial_trajectory": message.get("initialTrajectory", [0, 0, 0]),
                "launch_time": message.get("launchTime", message.get("epoch", "")),
                "velocity": message.get("velocity", 0),
                "altitude": message.get("altitude", 0),
                "thermal_signature": message.get("thermalSignature", {})
            })
            
        elif topic_key == "reentry":
            result.update({
                "altitude_km": message.get("altitude", 0) / 1000,  # Convert to km
                "prediction_confidence": message.get("predictionConfidence", 0.8),
                "predicted_location": message.get("predictedLocation", {"lat": 0, "lon": 0}),
                "predicted_time": message.get("predictedTime", ""),
                "velocity_km_s": message.get("velocity", 0) / 1000  # Convert to km/s
            })
            
        elif topic_key == "maneuver":
            result.update({
                "delta_v": message.get("deltaV", 0),
                "direction": message.get("direction", [0, 0, 0]),
                "timestamp": message.get("maneuverTime", message.get("epoch", "")),
                "initial_orbit": message.get("initialOrbit", {}),
                "final_orbit": message.get("finalOrbit", {})
            })
            
        elif topic_key == "separation":
            result.update({
                "parent_object_id": message.get("parentObjectId", ""),
                "child_object_id": message.get("childObjectId", ""),
                "relative_velocity": message.get("relativeVelocity", 0),
                "separation_time": message.get("separationTime", message.get("epoch", "")),
                "separation_distance": message.get("separationDistance", 0),
                "relative_trajectory": message.get("relativeTrajectory", [0, 0, 0])
            })
            
        elif topic_key == "conjunction":
            result.update({
                "primary_object_id": message.get("primaryObjectId", message.get("objectId", "")),
                "secondary_object_id": message.get("secondaryObjectId", ""),
                "minimum_distance": message.get("minimumDistance", 10000),
                "relative_velocity": message.get("relativeVelocity", 0),
                "closest_approach_time": message.get("closestApproachTime", message.get("epoch", "")),
                "radial_separation": message.get("radialSeparation", 0),
                "in_track_separation": message.get("inTrackSeparation", 0),
                "cross_track_separation": message.get("crossTrackSeparation", 0)
            })
            
        elif topic_key == "link-status":
            result.update({
                "link_type": message.get("linkType", "unknown"),
                "previous_state": message.get("previousState", "inactive"),
                "current_state": message.get("currentState", "active"),
                "link_change_time": message.get("linkChangeTime", message.get("epoch", "")),
                "frequency_band": message.get("frequencyBand", "unknown"),
                "signal_characteristics": message.get("signalCharacteristics", {})
            })
            
        elif topic_key == "attitude":
            result.update({
                "change_magnitude": message.get("changeMagnitude", 0),
                "previous_attitude": message.get("previousAttitude", [0, 0, 0, 0]),
                "current_attitude": message.get("currentAttitude", [0, 0, 0, 0]),
                "attitude_change_time": message.get("attitudeChangeTime", message.get("epoch", "")),
                "change_rate": message.get("changeRate", 0),
                "axis_of_rotation": message.get("axisOfRotation", [0, 0, 0])
            })
            
        return result
        
    def start(self):
        """Start UDL integration and message consumers."""
        if not self._is_config_valid() or not self.messaging_client:
            logger.warning("UDL integration not started due to invalid configuration.")
            return
            
        if self.running:
            logger.warning("UDL integration already running.")
            return
            
        try:
            # Start consumers for each topic
            for topic_key, topic_name in self.config.get("topics", {}).items():
                if topic_key in self.processors:
                    # Define message handler
                    def create_message_handler(topic_key):
                        def handle_message(message):
                            asyncio.create_task(self.process_message(message, topic_key))
                        return handle_message
                    
                    # Start consumer
                    handler = create_message_handler(topic_key)
                    self.messaging_client.start_consumer(
                        topic=topic_name,
                        callback_fn=handler,
                        start_from_latest=True
                    )
                    logger.info(f"Started consumer for {topic_name} with {topic_key} processor")
            
            # Start callback processor
            asyncio.create_task(self._process_callbacks())
            
            self.running = True
            logger.info("UDL integration started successfully.")
        except Exception as e:
            logger.error(f"Error starting UDL integration: {str(e)}")
            
    async def _process_callbacks(self):
        """Background task to process callbacks from queue."""
        while True:
            try:
                event, topic_key = await self.callback_queue.get()
                if self.event_callback:
                    self.event_callback(event, topic_key)
                self.callback_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing callback: {str(e)}")
            await asyncio.sleep(0.1)
            
    def stop(self):
        """Stop UDL integration and message consumers."""
        if not self.running:
            return
            
        try:
            # Stop all consumers
            if self.messaging_client:
                self.messaging_client.stop_all_consumers()
                
            self.running = False
            logger.info("UDL integration stopped.")
        except Exception as e:
            logger.error(f"Error stopping UDL integration: {str(e)}")
            
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of UDL integration.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "enabled": self.config.get("enabled", False),
            "running": self.running,
            "topics": list(self.config.get("topics", {}).keys()),
            "processors": list(self.processors.keys()),
            "callback_queue_size": self.callback_queue.qsize() if self.callback_queue else 0
        }
        
        if self.messaging_client:
            try:
                # Add messaging client metrics if available
                metrics = self.messaging_client.get_overall_metrics()
                status["metrics"] = metrics
            except Exception as e:
                status["metrics_error"] = str(e)
                
        return status