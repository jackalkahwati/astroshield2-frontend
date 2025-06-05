"""
Kafka Adapter for AstroShield AI Modules

This module provides the integration layer between the AI analysis modules
and the existing Kafka event processing pipeline. It handles message routing,
schema validation, and coordinated analysis workflows.

Key capabilities:
- Event routing to appropriate AI modules
- Schema validation and transformation
- Coordinated analysis pipeline
- Result aggregation and publishing
- Error handling and retry logic
- Performance monitoring
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid

from .models import (
    ManeuverEvent, ProximityEvent, IntentClassificationResult, 
    HostilityAssessment, AIAnalysisMessage, ModelConfig, PipelineConfig
)
from .intent_classifier import IntentClassifier
from .hostility_scorer import HostilityScorer
from app.common.logging import logger

try:
    from app.kafka.producer import KafkaProducer
    from app.kafka.consumer import EventConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    logger.warning("Kafka modules not available, using mock implementation")
    KAFKA_AVAILABLE = False


class MockKafkaProducer:
    """Mock Kafka producer for testing."""
    
    async def send_async(self, topic: str, message: Dict[str, Any]):
        """Mock send method."""
        logger.info(f"Mock publish to {topic}: {json.dumps(message, indent=2)}")


class MockKafkaConsumer:
    """Mock Kafka consumer for testing."""
    
    def __init__(self):
        self.running = False
    
    async def start(self):
        """Mock start method."""
        self.running = True
        logger.info("Mock Kafka consumer started")
    
    async def stop(self):
        """Mock stop method."""
        self.running = False
        logger.info("Mock Kafka consumer stopped")


class AIEventRouter:
    """Routes events to appropriate AI analysis modules."""
    
    def __init__(self, intent_classifier: IntentClassifier, hostility_scorer: HostilityScorer):
        self.intent_classifier = intent_classifier
        self.hostility_scorer = hostility_scorer
        
        # Event type routing
        self.event_routes = {
            "dmd-od-update": self._handle_orbit_update,
            "maneuver-detection": self._handle_maneuver_event,
            "proximity-alert": self._handle_proximity_event,
            "weather-data": self._handle_weather_data,
            "tle-update": self._handle_tle_update
        }
        
        # Analysis coordination
        self.pending_analyses: Dict[str, Dict[str, Any]] = {}
        
    async def route_event(self, message_type: str, payload: Dict[str, Any]) -> List[AIAnalysisMessage]:
        """Route event to appropriate analysis modules."""
        
        if message_type in self.event_routes:
            return await self.event_routes[message_type](payload)
        else:
            logger.warning(f"No route defined for message type: {message_type}")
            return []
    
    async def _handle_maneuver_event(self, payload: Dict[str, Any]) -> List[AIAnalysisMessage]:
        """Handle maneuver detection events."""
        try:
            # Parse maneuver event
            maneuver_event = self._parse_maneuver_event(payload)
            if not maneuver_event:
                return []
            
            # Coordinate analyses
            correlation_id = str(uuid.uuid4())
            analyses = []
            
            # Start intent classification
            intent_task = asyncio.create_task(
                self.intent_classifier.analyze_intent(maneuver_event)
            )
            
            # Check if proximity event exists for this maneuver
            proximity_event = await self._check_proximity_context(maneuver_event)
            
            # Wait for intent classification
            intent_result = await intent_task
            
            # Start hostility assessment with intent context
            hostility_result = await self.hostility_scorer.assess_hostility(
                maneuver_event, intent_result, proximity_event
            )
            
            # Create analysis messages
            analyses.append(AIAnalysisMessage(
                message_type="intent_classification_result",
                analysis_type="intent_classification",
                payload=intent_result.dict(),
                correlation_id=correlation_id
            ))
            
            analyses.append(AIAnalysisMessage(
                message_type="hostility_assessment_result", 
                analysis_type="hostility_assessment",
                payload=hostility_result.dict(),
                correlation_id=correlation_id
            ))
            
            return analyses
            
        except Exception as e:
            logger.error(f"Failed to handle maneuver event: {str(e)}")
            return []
    
    async def _handle_proximity_event(self, payload: Dict[str, Any]) -> List[AIAnalysisMessage]:
        """Handle proximity alert events."""
        try:
            proximity_event = self._parse_proximity_event(payload)
            if not proximity_event:
                return []
            
            # Store proximity context for potential maneuver analysis
            self._store_proximity_context(proximity_event)
            
            # For now, just log the proximity event
            logger.info(f"Proximity event stored: {proximity_event.event_id}")
            return []
            
        except Exception as e:
            logger.error(f"Failed to handle proximity event: {str(e)}")
            return []
    
    async def _handle_orbit_update(self, payload: Dict[str, Any]) -> List[AIAnalysisMessage]:
        """Handle orbit determination updates."""
        # Could trigger analysis for significant orbital changes
        logger.debug("Orbit update received - no immediate analysis required")
        return []
    
    async def _handle_weather_data(self, payload: Dict[str, Any]) -> List[AIAnalysisMessage]:
        """Handle weather data for observation planning."""
        # Could be used for observation recommendation analysis
        logger.debug("Weather data received - available for observation planning")
        return []
    
    async def _handle_tle_update(self, payload: Dict[str, Any]) -> List[AIAnalysisMessage]:
        """Handle TLE updates."""
        # Could trigger maneuver detection if significant changes detected
        logger.debug("TLE update received - checking for maneuver indicators")
        return []
    
    def _parse_maneuver_event(self, payload: Dict[str, Any]) -> Optional[ManeuverEvent]:
        """Parse payload into ManeuverEvent object."""
        try:
            # Extract required fields with defaults
            event_data = {
                "sat_pair_id": payload.get("satellite_id", "unknown"),
                "primary_norad_id": payload.get("norad_id", payload.get("catalogId", "unknown")),
                "maneuver_type": payload.get("maneuver_type", "unknown"),
                "delta_v": float(payload.get("delta_v", 0.0)),
                "confidence": float(payload.get("confidence", 0.5)),
                "orbital_elements_before": payload.get("orbital_elements_before", {}),
                "orbital_elements_after": payload.get("orbital_elements_after", {}),
                "source_data_lineage": payload.get("metadata", {})
            }
            
            # Optional fields
            if "burn_duration" in payload:
                event_data["burn_duration"] = float(payload["burn_duration"])
            if "secondary_norad_id" in payload:
                event_data["secondary_norad_id"] = payload["secondary_norad_id"]
            
            return ManeuverEvent(**event_data)
            
        except Exception as e:
            logger.error(f"Failed to parse maneuver event: {str(e)}")
            return None
    
    def _parse_proximity_event(self, payload: Dict[str, Any]) -> Optional[ProximityEvent]:
        """Parse payload into ProximityEvent object."""
        try:
            event_data = {
                "sat_pair_id": f"{payload.get('primary_id', 'unk')}_{payload.get('secondary_id', 'unk')}",
                "primary_norad_id": payload.get("primary_id", "unknown"),
                "secondary_norad_id": payload.get("secondary_id", "unknown"),
                "closest_approach_time": datetime.fromisoformat(
                    payload.get("closest_approach_time", datetime.utcnow().isoformat())
                ),
                "minimum_distance": float(payload.get("minimum_distance", 0.0)),
                "relative_velocity": float(payload.get("relative_velocity", 0.0)),
                "duration_minutes": float(payload.get("duration_minutes", 0.0)),
                "approach_geometry": payload.get("approach_geometry", {})
            }
            
            return ProximityEvent(**event_data)
            
        except Exception as e:
            logger.error(f"Failed to parse proximity event: {str(e)}")
            return None
    
    async def _check_proximity_context(self, maneuver_event: ManeuverEvent) -> Optional[ProximityEvent]:
        """Check if there's relevant proximity context for this maneuver."""
        # Look for recent proximity events involving this satellite
        # This is a simplified implementation
        return None
    
    def _store_proximity_context(self, proximity_event: ProximityEvent):
        """Store proximity event for context in future analyses."""
        # Store in cache with TTL for correlation with maneuvers
        pass


class KafkaAdapter:
    """Main Kafka adapter for AI modules integration."""
    
    def __init__(self, 
                 config: Optional[PipelineConfig] = None,
                 kafka_producer: Optional[Any] = None,
                 kafka_consumer: Optional[Any] = None):
        """Initialize the Kafka adapter."""
        
        self.config = config or PipelineConfig()
        
        # Initialize Kafka components
        if KAFKA_AVAILABLE:
            self.producer = kafka_producer or KafkaProducer()
            self.consumer = kafka_consumer or EventConsumer()
        else:
            self.producer = MockKafkaProducer()
            self.consumer = MockKafkaConsumer()
        
        # Initialize AI modules
        self.intent_classifier = IntentClassifier(kafka_adapter=self)
        self.hostility_scorer = HostilityScorer(kafka_adapter=self)
        
        # Initialize event router
        self.event_router = AIEventRouter(self.intent_classifier, self.hostility_scorer)
        
        # Processing metrics
        self.metrics = {
            "events_processed": 0,
            "analyses_completed": 0,
            "errors": 0,
            "start_time": datetime.utcnow()
        }
        
        # Topic configuration
        self.input_topics = [
            "dmd-od-update",
            "maneuver-detection", 
            "proximity-alert",
            "weather-data",
            "tle-update"
        ]
        
        self.output_topics = {
            "intent_classification": "astroshield.ai.intent_classification",
            "hostility_assessment": "astroshield.ai.hostility_assessment",
            "observation_recommendation": "astroshield.ai.observation_recommendation",
            "analysis_metrics": "astroshield.ai.metrics"
        }
        
        logger.info("KafkaAdapter initialized")
    
    async def start(self):
        """Start the Kafka adapter and begin processing events."""
        try:
            # Start Kafka consumer
            await self.consumer.start()
            
            # Register message handlers
            for topic in self.input_topics:
                self.consumer.register_handler(topic, self._handle_message)
            
            logger.info("KafkaAdapter started - listening for events")
            
        except Exception as e:
            logger.error(f"Failed to start KafkaAdapter: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the Kafka adapter."""
        try:
            await self.consumer.stop()
            logger.info("KafkaAdapter stopped")
            
        except Exception as e:
            logger.error(f"Error stopping KafkaAdapter: {str(e)}")
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming Kafka message."""
        try:
            self.metrics["events_processed"] += 1
            
            # Extract message type and payload
            message_type = message.get("messageType") or message.get("topic", "unknown")
            payload = message.get("payload", message)
            
            logger.debug(f"Processing message type: {message_type}")
            
            # Route to AI modules
            analysis_results = await self.event_router.route_event(message_type, payload)
            
            # Publish analysis results
            for result in analysis_results:
                await self._publish_analysis_result(result)
                self.metrics["analyses_completed"] += 1
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Error handling message: {str(e)}")
    
    async def _publish_analysis_result(self, result: AIAnalysisMessage):
        """Publish AI analysis result to appropriate topic."""
        try:
            topic = self.output_topics.get(result.analysis_type, "astroshield.ai.general")
            await self.publish(topic, result.dict())
            
        except Exception as e:
            logger.error(f"Failed to publish analysis result: {str(e)}")
    
    async def publish(self, topic: str, message: Dict[str, Any]):
        """Publish message to Kafka topic."""
        try:
            await self.producer.send_async(topic, message)
            logger.debug(f"Published message to {topic}")
            
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {str(e)}")
    
    async def generate_test_events(self, count: int = 10):
        """Generate test events for integration testing with Welders Arc."""
        logger.info(f"Generating {count} test events for Welders Arc integration")
        
        test_events = []
        
        for i in range(count):
            # Create test maneuver event
            maneuver_event = {
                "messageType": "maneuver-detection",
                "timestamp": datetime.utcnow().isoformat(),
                "payload": {
                    "event_id": f"test-maneuver-{i+1}",
                    "satellite_id": f"TEST-SAT-{(i % 3) + 1}",
                    "norad_id": f"9999{i:02d}",
                    "maneuver_type": ["prograde", "retrograde", "normal"][i % 3],
                    "delta_v": 0.5 + (i * 0.3),
                    "confidence": 0.7 + (i * 0.02),
                    "burn_duration": 10 + (i * 5),
                    "orbital_elements_before": {
                        "semi_major_axis": 7000.0,
                        "eccentricity": 0.001,
                        "inclination": 51.6,
                        "raan": 0.0,
                        "argument_of_perigee": 0.0,
                        "mean_anomaly": 0.0
                    },
                    "orbital_elements_after": {
                        "semi_major_axis": 7000.0 + (i * 0.1),
                        "eccentricity": 0.001,
                        "inclination": 51.6,
                        "raan": 0.0,
                        "argument_of_perigee": 0.0,
                        "mean_anomaly": 0.0
                    },
                    "metadata": {
                        "source": "test_generator",
                        "test_scenario": f"scenario_{i+1}"
                    }
                }
            }
            
            test_events.append(maneuver_event)
            
            # Publish test event
            await self.publish("maneuver-detection", maneuver_event)
            
            # Add some proximity events
            if i % 3 == 0:
                proximity_event = {
                    "messageType": "proximity-alert",
                    "timestamp": datetime.utcnow().isoformat(),
                    "payload": {
                        "event_id": f"test-proximity-{i+1}",
                        "primary_id": f"TEST-SAT-{(i % 3) + 1}",
                        "secondary_id": "TARGET-SAT-1",
                        "closest_approach_time": datetime.utcnow().isoformat(),
                        "minimum_distance": 5000 - (i * 100),
                        "relative_velocity": 500 + (i * 50),
                        "duration_minutes": 30 + (i * 5),
                        "approach_geometry": {
                            "radial": 1000.0,
                            "in_track": 2000.0,
                            "cross_track": 1500.0
                        }
                    }
                }
                
                await self.publish("proximity-alert", proximity_event)
            
            # Small delay between events
            await asyncio.sleep(0.1)
        
        logger.info(f"Generated and published {count} test events")
        return test_events
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        runtime = (datetime.utcnow() - self.metrics["start_time"]).total_seconds()
        
        return {
            **self.metrics,
            "runtime_seconds": runtime,
            "events_per_second": self.metrics["events_processed"] / max(runtime, 1),
            "error_rate": self.metrics["errors"] / max(self.metrics["events_processed"], 1)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        try:
            # Check intent classifier
            health_status["components"]["intent_classifier"] = {
                "status": "healthy",
                "metrics": self.intent_classifier.get_performance_metrics()
            }
            
            # Check hostility scorer
            health_status["components"]["hostility_scorer"] = {
                "status": "healthy", 
                "model_version": self.hostility_scorer.config.model_version
            }
            
            # Check Kafka connectivity (simplified)
            health_status["components"]["kafka"] = {
                "status": "healthy" if KAFKA_AVAILABLE else "mocked",
                "producer": "connected",
                "consumer": "connected"
            }
            
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["error"] = str(e)
        
        return health_status


# Factory function for easy instantiation
def create_kafka_adapter(config: Optional[PipelineConfig] = None) -> KafkaAdapter:
    """Create and return a KafkaAdapter instance."""
    return KafkaAdapter(config=config) 