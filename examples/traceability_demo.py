#!/usr/bin/env python3
"""
AstroShield Traceability Demo

This script demonstrates the full power of the AstroShield message tracing architecture
by simulating a complete workflow through multiple subsystems, from data ingestion to
threat assessment, with full message traceability.

Usage:
    python traceability_demo.py

Requirements:
    - All AstroShield common utilities installed
    - Kafka cluster configuration in .env file (or use --simulate flag)
"""

import os
import sys
import json
import uuid
import time
import logging
import argparse
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import AstroShield components
try:
    from src.asttroshield.common.message_headers import MessageFactory
    from src.asttroshield.common.kafka_utils import KafkaConfig, AstroShieldProducer, AstroShieldConsumer
    from src.asttroshield.common.subsystem import SubsystemBase, Subsystem0, Subsystem1, Subsystem2, Subsystem4, Subsystem6
except ImportError:
    print("Error: AstroShield common components not found. Please run from the root directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("traceability_demo")

# Load environment variables
load_dotenv()

# Topic names for the demo
TOPICS = {
    "ss0_sensor_data": "demo.ss0.sensor.data",
    "ss1_target_model": "demo.ss1.target.model",
    "ss2_state_estimate": "demo.ss2.state.estimate",
    "ss4_ccdm_detection": "demo.ss4.ccdm.detection", 
    "ss6_threat_assessment": "demo.ss6.threat.assessment"
}

class SimulatedProducer:
    """Simulated Kafka producer for local testing."""
    
    def __init__(self, callback=None):
        self.messages = []
        self.callback = callback
        
    def publish(self, topic, message):
        """Publish a message to the simulated queue."""
        logger.info(f"Published to {topic}: {message['header']['messageId']}")
        self.messages.append((topic, message))
        
        # If there's a callback registered (for the consumer), call it
        if self.callback:
            self.callback(topic, message)
        
        return True

class SimulatedConsumer:
    """Simulated Kafka consumer for local testing."""
    
    def __init__(self, topics, group_id, process_message):
        self.topics = topics
        self.group_id = group_id
        self.process_message = process_message
        self.running = False
        self.producers = []
        
    def register_producer(self, producer):
        """Register a producer to receive messages from."""
        producer.callback = self._receive_message
        self.producers.append(producer)
        
    def _receive_message(self, topic, message):
        """Receive a message from a producer."""
        if topic in self.topics:
            # Process the message if it's for one of our subscribed topics
            self.process_message(message)
    
    def start(self):
        """Start the consumer."""
        self.running = True
        logger.info(f"Started consumer for topics: {', '.join(self.topics)}")
        
    def stop(self):
        """Stop the consumer."""
        self.running = False
        logger.info("Stopped consumer")

def simulate_sensor_data():
    """Generate simulated sensor data for Space Domain Awareness."""
    object_id = f"SAT-{uuid.uuid4().hex[:8]}"
    observation_time = datetime.utcnow()
    
    return {
        "objectId": object_id,
        "timestamp": observation_time.isoformat(),
        "sensorId": "SDA-SENSOR-001",
        "observationType": "OPTICAL",
        "rawData": {
            "azimuth": 234.56,
            "elevation": 45.67,
            "range": 35786.2,
            "rangeRate": 0.05,
            "signalToNoise": 12.3,
            "brightness": 8.7
        },
        "metadata": {
            "observationDuration": 2.5,
            "weatherConditions": "CLEAR",
            "calibrationStatus": "NOMINAL"
        }
    }

class DataIngestionSubsystem(Subsystem0):
    """Subsystem 0: Data Ingestion from sensors and external sources."""
    
    def __init__(self, kafka_producer):
        super().__init__(name="Data Ingestion")
        self.producer = kafka_producer
        self.register_output_topic(TOPICS["ss0_sensor_data"])
    
    def ingest_sensor_data(self):
        """Ingest data from a simulated sensor."""
        # Generate simulated sensor data
        sensor_data = simulate_sensor_data()
        
        # Create a message with the sensor data
        message = MessageFactory.create_message(
            message_type="ss0.sensor.data",
            source="sda_sensor_network",
            payload=sensor_data
        )
        
        # Log the message creation with traceability info
        logger.info(f"SS0: Created sensor data message with ID {message['header']['messageId']}")
        logger.info(f"SS0: Trace ID: {message['header']['traceId']}")
        
        # Publish the message
        self.publish_message(TOPICS["ss0_sensor_data"], message)
        
        return message

class TargetModelingSubsystem(Subsystem1):
    """Subsystem 1: Target Modeling from ingested data."""
    
    def __init__(self, kafka_producer):
        super().__init__(name="Target Modeling")
        self.producer = kafka_producer
        self.register_input_topic(TOPICS["ss0_sensor_data"])
        self.register_output_topic(TOPICS["ss1_target_model"])
    
    def process_message(self, message):
        """Process sensor data and create/update a target model."""
        header = message.get('header', {})
        payload = message.get('payload', {})
        
        logger.info(f"SS1: Processing message {header.get('messageId')} from {header.get('source')}")
        logger.info(f"SS1: Trace ID: {header.get('traceId')}")
        
        # Extract information from the payload
        object_id = payload.get('objectId')
        raw_data = payload.get('rawData', {})
        
        if not object_id or not raw_data:
            logger.warning("SS1: Missing required data in message payload")
            return
        
        # Process the raw data to create a target model
        # In a real system, this would involve sophisticated algorithms
        target_model = {
            "objectId": object_id,
            "objectType": "SATELLITE",
            "creationTime": datetime.utcnow().isoformat(),
            "dimensions": {
                "length": 5.2 + (raw_data.get('signalToNoise', 10) / 100),
                "width": 2.8,
                "height": 3.0
            },
            "estimatedMass": 750.0,
            "crossSectionalArea": 14.5,
            "surfaceMaterials": ["SOLAR_PANELS", "ALUMINUM", "MULTI_LAYER_INSULATION"],
            "estimatedCapabilities": {
                "propulsion": True,
                "attitude_control": True,
                "power": "SOLAR",
                "communication": ["S_BAND", "X_BAND"]
            },
            "confidenceScore": 0.85,
            "dataSource": header.get('source')
        }
        
        # Create a derived message with the target model
        derived_message = self.derive_message(
            message,
            "ss1.target.model",
            target_model
        )
        
        # Log the derived message creation
        logger.info(f"SS1: Created target model message with ID {derived_message['header']['messageId']}")
        logger.info(f"SS1: Parent message ID: {header.get('messageId')}")
        logger.info(f"SS1: Maintained trace ID: {derived_message['header']['traceId']}")
        
        # Publish the derived message
        self.publish_message(TOPICS["ss1_target_model"], derived_message)
        
        return derived_message

class StateEstimationSubsystem(Subsystem2):
    """Subsystem 2: State Estimation for tracking objects."""
    
    def __init__(self, kafka_producer):
        super().__init__(name="State Estimation")
        self.producer = kafka_producer
        self.register_input_topic(TOPICS["ss1_target_model"])
        self.register_output_topic(TOPICS["ss2_state_estimate"])
    
    def process_message(self, message):
        """Process target model and create state estimates."""
        header = message.get('header', {})
        payload = message.get('payload', {})
        
        logger.info(f"SS2: Processing message {header.get('messageId')} from {header.get('source')}")
        logger.info(f"SS2: Trace ID: {header.get('traceId')}")
        
        # Extract information from the payload
        object_id = payload.get('objectId')
        object_type = payload.get('objectType')
        
        if not object_id or not object_type:
            logger.warning("SS2: Missing required data in message payload")
            return
        
        # Create a state estimate based on the target model
        # In a real system, this would use orbital mechanics and filtering algorithms
        state_estimate = {
            "objectId": object_id,
            "objectType": object_type,
            "timestamp": datetime.utcnow().isoformat(),
            "position": [42164.0, 0.0, 0.0],  # Geostationary orbit example (km)
            "velocity": [0.0, 3.075, 0.0],    # Orbital velocity (km/s)
            "covariance": [
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.001, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
            ],
            "orbitParameters": {
                "semimajorAxis": 42164.0,
                "eccentricity": 0.0,
                "inclination": 0.05,
                "raan": 180.0,
                "argumentOfPerigee": 0.0,
                "meanAnomaly": 90.0
            },
            "covarianceFrame": "RTN",
            "estimationAlgorithm": "UKF",
            "confidenceScore": 0.92
        }
        
        # Create a derived message with the state estimate
        derived_message = self.derive_message(
            message,
            "ss2.state.estimate",
            state_estimate
        )
        
        # Log the derived message creation
        logger.info(f"SS2: Created state estimate message with ID {derived_message['header']['messageId']}")
        logger.info(f"SS2: Parent message ID: {header.get('messageId')}")
        logger.info(f"SS2: Maintained trace ID: {derived_message['header']['traceId']}")
        
        # Publish the derived message
        self.publish_message(TOPICS["ss2_state_estimate"], derived_message)
        
        return derived_message

class CCDMDetectionSubsystem(Subsystem4):
    """Subsystem 4: CCDM Detection for identifying suspicious activities."""
    
    def __init__(self, kafka_producer):
        super().__init__(name="CCDM Detection")
        self.producer = kafka_producer
        self.register_input_topic(TOPICS["ss2_state_estimate"])
        self.register_output_topic(TOPICS["ss4_ccdm_detection"])
        
        # For the demo, sometimes we'll inject anomalies
        self.inject_anomaly = True
    
    def process_message(self, message):
        """Process state estimates and detect CCDM activities."""
        header = message.get('header', {})
        payload = message.get('payload', {})
        
        logger.info(f"SS4: Processing message {header.get('messageId')} from {header.get('source')}")
        logger.info(f"SS4: Trace ID: {header.get('traceId')}")
        
        # Extract information from the payload
        object_id = payload.get('objectId')
        position = payload.get('position', [])
        velocity = payload.get('velocity', [])
        
        if not object_id or not position or not velocity:
            logger.warning("SS4: Missing required data in message payload")
            return
        
        # For the demo, we'll sometimes detect an anomaly
        if self.inject_anomaly:
            # Create a CCDM detection for a simulated maneuver
            detection = {
                "objectId": object_id,
                "detectionTime": datetime.utcnow().isoformat(),
                "ccdmType": "MANEUVER",
                "confidence": 0.78,
                "indicators": [
                    "DELTA_V_DETECTED",
                    "ORBITAL_PLANE_CHANGE"
                ],
                "details": {
                    "deltaV": 0.05,  # km/s
                    "burnDuration": 120,  # seconds
                    "burnDirection": [0.866, 0.5, 0.0],  # unit vector
                    "startTime": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                    "endTime": (datetime.utcnow() - timedelta(minutes=3)).isoformat()
                },
                "relatedObjects": [],
                "evidenceData": {
                    "beforeState": {
                        "position": [42164.0, 0.0, 0.0],
                        "velocity": [0.0, 3.075, 0.0]
                    },
                    "afterState": {
                        "position": [42164.0, 0.0, 0.0], 
                        "velocity": [0.0, 3.125, 0.0]  # Increased velocity
                    }
                },
                "assessment": {
                    "expectedManeuver": False,
                    "anomalyScore": 0.65,
                    "potentialIntent": ["STATION_KEEPING", "APPROACH"]
                },
                "detectionMetadata": {
                    "algorithmVersion": "1.3.0",
                    "detectionLatency": 120,  # seconds
                    "analysisWindow": 1800  # seconds
                }
            }
            
            # Create a derived message with the CCDM detection
            derived_message = self.derive_message(
                message,
                "ss4.ccdm.detection",
                detection
            )
            
            # Log the derived message creation
            logger.info(f"SS4: Created CCDM detection message with ID {derived_message['header']['messageId']}")
            logger.info(f"SS4: Parent message ID: {header.get('messageId')}")
            logger.info(f"SS4: Maintained trace ID: {derived_message['header']['traceId']}")
            
            # Publish the derived message
            self.publish_message(TOPICS["ss4_ccdm_detection"], derived_message)
            
            # Turn off anomaly detection for next time
            self.inject_anomaly = False
            
            return derived_message
        else:
            logger.info("SS4: No CCDM activity detected")
            return None

class ThreatAssessmentSubsystem(Subsystem6):
    """Subsystem 6: Threat Assessment for evaluating detected activities."""
    
    def __init__(self, kafka_producer):
        super().__init__(name="Threat Assessment")
        self.producer = kafka_producer
        self.register_input_topic(TOPICS["ss4_ccdm_detection"])
        self.register_output_topic(TOPICS["ss6_threat_assessment"])
    
    def process_message(self, message):
        """Process CCDM detections and assess threat levels."""
        header = message.get('header', {})
        payload = message.get('payload', {})
        
        logger.info(f"SS6: Processing message {header.get('messageId')} from {header.get('source')}")
        logger.info(f"SS6: Trace ID: {header.get('traceId')}")
        
        # Extract information from the payload
        object_id = payload.get('objectId')
        ccdm_type = payload.get('ccdmType')
        confidence = payload.get('confidence', 0)
        indicators = payload.get('indicators', [])
        
        if not object_id or not ccdm_type:
            logger.warning("SS6: Missing required data in message payload")
            return
        
        # Determine the threat level based on the CCDM detection
        # In a real system, this would use sophisticated threat models
        if ccdm_type == "MANEUVER" and confidence > 0.7:
            threat_level = "MEDIUM"
            if "APPROACH" in payload.get('assessment', {}).get('potentialIntent', []):
                threat_level = "HIGH"
        else:
            threat_level = "LOW"
        
        # Create a threat assessment
        assessment = {
            "objectId": object_id,
            "assessmentTime": datetime.utcnow().isoformat(),
            "threatLevel": threat_level,
            "confidence": min(confidence + 0.05, 1.0),  # Slightly increase confidence
            "source": payload.get('ccdmType'),
            "details": {
                "activityType": ccdm_type,
                "indicators": indicators,
                "assessmentFactors": [
                    "HISTORICAL_BEHAVIOR",
                    "CURRENT_GEOPOLITICAL_CONTEXT",
                    "TECHNICAL_CAPABILITIES"
                ]
            },
            "recommendations": [
                "INCREASE_MONITORING",
                "NOTIFY_OPERATORS" if threat_level == "MEDIUM" else None,
                "ALERT_COMMAND_STAFF" if threat_level == "HIGH" else None
            ],
            "relatedDetections": [header.get('messageId')],
            "assessmentMetadata": {
                "algorithmVersion": "2.1.0",
                "assessmentLatency": 5,  # seconds
                "confidenceModel": "BAYESIAN_NETWORK"
            }
        }
        
        # Filter out None values from recommendations
        assessment["recommendations"] = [r for r in assessment["recommendations"] if r]
        
        # Create a derived message with the threat assessment
        derived_message = self.derive_message(
            message,
            "ss6.threat.assessment",
            assessment
        )
        
        # Log the derived message creation
        logger.info(f"SS6: Created threat assessment message with ID {derived_message['header']['messageId']}")
        logger.info(f"SS6: Parent message ID: {header.get('messageId')}")
        logger.info(f"SS6: Maintained trace ID: {derived_message['header']['traceId']}")
        logger.info(f"SS6: Threat level: {threat_level}")
        
        # Publish the derived message
        self.publish_message(TOPICS["ss6_threat_assessment"], derived_message)
        
        return derived_message

def get_producers_and_consumers(simulate=False):
    """Get Kafka producers and consumers based on configuration."""
    if simulate:
        # Create simulated Kafka producers for each subsystem
        producer_ss0 = SimulatedProducer()
        producer_ss1 = SimulatedProducer()
        producer_ss2 = SimulatedProducer()
        producer_ss4 = SimulatedProducer()
        producer_ss6 = SimulatedProducer()
        
        # Create subsystem instances
        ss0 = DataIngestionSubsystem(producer_ss0)
        ss1 = TargetModelingSubsystem(producer_ss1)
        ss2 = StateEstimationSubsystem(producer_ss2)
        ss4 = CCDMDetectionSubsystem(producer_ss4)
        ss6 = ThreatAssessmentSubsystem(producer_ss6)
        
        # Create simulated consumers
        consumer_ss1 = SimulatedConsumer([TOPICS["ss0_sensor_data"]], "ss1-group", ss1.process_message)
        consumer_ss2 = SimulatedConsumer([TOPICS["ss1_target_model"]], "ss2-group", ss2.process_message)
        consumer_ss4 = SimulatedConsumer([TOPICS["ss2_state_estimate"]], "ss4-group", ss4.process_message)
        consumer_ss6 = SimulatedConsumer([TOPICS["ss4_ccdm_detection"]], "ss6-group", ss6.process_message)
        
        # Register producers with consumers
        consumer_ss1.register_producer(producer_ss0)
        consumer_ss2.register_producer(producer_ss1)
        consumer_ss4.register_producer(producer_ss2)
        consumer_ss6.register_producer(producer_ss4)
        
        # Start consumers
        consumer_ss1.start()
        consumer_ss2.start()
        consumer_ss4.start()
        consumer_ss6.start()
        
        return {
            "ss0": {"subsystem": ss0, "producer": producer_ss0, "consumer": None},
            "ss1": {"subsystem": ss1, "producer": producer_ss1, "consumer": consumer_ss1},
            "ss2": {"subsystem": ss2, "producer": producer_ss2, "consumer": consumer_ss2},
            "ss4": {"subsystem": ss4, "producer": producer_ss4, "consumer": consumer_ss4},
            "ss6": {"subsystem": ss6, "producer": producer_ss6, "consumer": consumer_ss6}
        }
    else:
        # Create real Kafka configuration
        kafka_config = KafkaConfig(
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
            security_protocol=os.getenv('KAFKA_SECURITY_PROTOCOL'),
            sasl_mechanism=os.getenv('KAFKA_SASL_MECHANISM'),
            producer_username=os.getenv('KAFKA_PRODUCER_USERNAME'),
            producer_password=os.getenv('KAFKA_PRODUCER_PASSWORD'),
            consumer_username=os.getenv('KAFKA_CONSUMER_USERNAME'),
            consumer_password=os.getenv('KAFKA_CONSUMER_PASSWORD')
        )
        
        # Create real Kafka producers and consumers
        producer = AstroShieldProducer(kafka_config)
        
        # Create subsystem instances
        ss0 = DataIngestionSubsystem(producer)
        ss1 = TargetModelingSubsystem(producer)
        ss2 = StateEstimationSubsystem(producer)
        ss4 = CCDMDetectionSubsystem(producer)
        ss6 = ThreatAssessmentSubsystem(producer)
        
        # Create real Kafka consumers
        consumer_ss1 = AstroShieldConsumer(kafka_config, [TOPICS["ss0_sensor_data"]], "ss1-group", ss1.process_message)
        consumer_ss2 = AstroShieldConsumer(kafka_config, [TOPICS["ss1_target_model"]], "ss2-group", ss2.process_message)
        consumer_ss4 = AstroShieldConsumer(kafka_config, [TOPICS["ss2_state_estimate"]], "ss4-group", ss4.process_message)
        consumer_ss6 = AstroShieldConsumer(kafka_config, [TOPICS["ss4_ccdm_detection"]], "ss6-group", ss6.process_message)
        
        return {
            "ss0": {"subsystem": ss0, "producer": producer, "consumer": None},
            "ss1": {"subsystem": ss1, "producer": producer, "consumer": consumer_ss1},
            "ss2": {"subsystem": ss2, "producer": producer, "consumer": consumer_ss2},
            "ss4": {"subsystem": ss4, "producer": producer, "consumer": consumer_ss4},
            "ss6": {"subsystem": ss6, "producer": producer, "consumer": consumer_ss6}
        }

def run_demo(simulate=False):
    """Run the traceability demo."""
    logger.info("Starting AstroShield Traceability Demo")
    logger.info(f"Mode: {'Simulated' if simulate else 'Real Kafka'}")
    
    # Get producers, consumers, and subsystem instances
    components = get_producers_and_consumers(simulate)
    
    # Start consumers if using real Kafka
    if not simulate:
        for ss_name, comp in components.items():
            if comp["consumer"]:
                comp["consumer"].start()
    
    try:
        # Simulate data ingestion
        logger.info("Ingesting sensor data...")
        initial_message = components["ss0"]["subsystem"].ingest_sensor_data()
        
        # In simulation mode, we need to wait for processing
        if simulate:
            time.sleep(1)  # Wait for messages to propagate through the chain
        else:
            # In real Kafka mode, we just need to wait for messages to be processed
            logger.info("Waiting for messages to be processed...")
            time.sleep(10)
        
        # Print the trace summary
        logger.info("\n--- Trace Summary ---")
        logger.info(f"Trace ID: {initial_message['header']['traceId']}")
        
        # In a real application, you would query Kafka or a database for all messages with this trace ID
        
        logger.info("Demo completed successfully")
            
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Error in demo: {e}")
    finally:
        # Stop consumers if using real Kafka
        if not simulate:
            for ss_name, comp in components.items():
                if comp["consumer"]:
                    comp["consumer"].stop()

def main():
    """Main entry point for the traceability demo."""
    parser = argparse.ArgumentParser(description="AstroShield Traceability Demo")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode (no Kafka needed)")
    
    args = parser.parse_args()
    
    run_demo(simulate=args.simulate)

if __name__ == "__main__":
    main() 