#!/usr/bin/env python3
"""
AstroShield Example Application

This example demonstrates how to set up multiple subsystems with proper message
flow and traceability between them.
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
from datetime import datetime
from typing import Dict, Any, List

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.asttroshield.common.logging_utils import configure_logging, get_logger, trace_context
from src.asttroshield.common.kafka_utils import KafkaConfig
from src.asttroshield.common.subsystem import (
    Subsystem0, Subsystem1, Subsystem2, Subsystem4, Subsystem5, Subsystem6
)


# Configure logging
configure_logging()
logger = get_logger(__name__)


def load_kafka_config() -> KafkaConfig:
    """
    Load Kafka configuration from environment variables.
    
    Returns:
        KafkaConfig: Kafka configuration
    """
    # Try to load from environment variables
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    security_protocol = os.environ.get('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT')
    sasl_mechanism = os.environ.get('KAFKA_SASL_MECHANISM')
    username = os.environ.get('KAFKA_USERNAME')
    password = os.environ.get('KAFKA_PASSWORD')
    
    return KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        security_protocol=security_protocol,
        sasl_mechanism=sasl_mechanism,
        username=username,
        password=password,
        client_id='astroshield-example-app'
    )


def run_subsystem0(kafka_config: KafkaConfig) -> None:
    """
    Run the Data Ingestion subsystem.
    
    Args:
        kafka_config: Kafka configuration
    """
    subsystem = Subsystem0(kafka_config)
    subsystem.initialize()
    
    # Simulate sensor observations
    try:
        while True:
            # Create a sensor observation
            with trace_context():
                logger.info("Generating sensor observation")
                
                # Example observation payload
                payload = {
                    "observationId": f"obs-{int(time.time())}",
                    "sensorId": "ground-radar-1",
                    "timestamp": datetime.utcnow().isoformat(),
                    "targetId": f"target-{int(time.time()) % 10}",
                    "azimuth": 45.0 + (time.time() % 10),
                    "elevation": 30.0 + (time.time() % 5),
                    "range": 1000.0 + (time.time() % 100),
                    "rangeRate": 0.5 + (time.time() % 1),
                    "signalToNoise": 15.0 + (time.time() % 5),
                    "dataQuality": "GOOD"
                }
                
                # Publish observation
                subsystem.publish_message(
                    topic="ss0.sensor.observation",
                    message_type="sensor.observation",
                    payload=payload,
                    key=payload["targetId"]
                )
                
                # Also publish a heartbeat (different topic)
                heartbeat_payload = {
                    "sensorId": "ground-radar-1",
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "OPERATIONAL",
                    "uptime": int(time.time() % 86400),
                    "batteryLevel": 0.95
                }
                
                subsystem.publish_message(
                    topic="ss0.sensor.heartbeat",
                    message_type="sensor.heartbeat",
                    payload=heartbeat_payload
                )
            
            # Sleep before generating next observation
            time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("Data Ingestion subsystem stopped by user")
    finally:
        subsystem.stop()


def run_subsystem1(kafka_config: KafkaConfig) -> None:
    """
    Run the Target Modeling subsystem.
    
    Args:
        kafka_config: Kafka configuration
    """
    # Override process_message to implement actual functionality
    class CustomSubsystem1(Subsystem1):
        @trace_context
        def process_message(self, message: Dict[str, Any]) -> None:
            header = message.get("header", {})
            payload = message.get("payload", {})
            message_type = header.get("messageType", "unknown")
            
            logger.info(f"Processing {message_type} message in Target Modeling subsystem")
            
            if message_type == "sensor.observation":
                # Create object identification message
                target_id = payload.get("targetId")
                
                if target_id:
                    object_payload = {
                        "objectId": f"obj-{target_id}",
                        "catalogId": f"cat-{hash(target_id) % 10000}",
                        "objectType": "SATELLITE",
                        "name": f"Object {target_id}",
                        "confidence": 0.95,
                        "lastObservationTime": payload.get("timestamp"),
                        "metadata": {
                            "origin": "example-app",
                            "sensorId": payload.get("sensorId")
                        }
                    }
                    
                    # Publish derived message (maintains traceability)
                    self.derive_message(
                        parent_message=message,
                        message_type="object.identification",
                        payload=object_payload,
                        topic="ss1.object.identification",
                        key=object_payload["objectId"]
                    )
    
    # Create and run the subsystem
    subsystem = CustomSubsystem1(kafka_config)
    subsystem.initialize()
    
    try:
        # Start consuming messages
        subsystem.start_consuming()
    except KeyboardInterrupt:
        logger.info("Target Modeling subsystem stopped by user")
    finally:
        subsystem.stop()


def run_subsystem2(kafka_config: KafkaConfig) -> None:
    """
    Run the State Estimation subsystem.
    
    Args:
        kafka_config: Kafka configuration
    """
    # Override process_message to implement actual functionality
    class CustomSubsystem2(Subsystem2):
        @trace_context
        def process_message(self, message: Dict[str, Any]) -> None:
            header = message.get("header", {})
            payload = message.get("payload", {})
            message_type = header.get("messageType", "unknown")
            
            logger.info(f"Processing {message_type} message in State Estimation subsystem")
            
            if message_type == "object.identification":
                # Create state vector message
                object_id = payload.get("objectId")
                
                if object_id:
                    # Generate a simulated state vector
                    now = datetime.utcnow()
                    
                    state_vector_payload = {
                        "stateVectorId": f"sv-{int(time.time())}",
                        "objectId": object_id,
                        "epoch": now.isoformat(),
                        "referenceFrame": "GCRF",
                        "position": {
                            "x": 42164.0 + (hash(object_id) % 100),  # GEO orbit in km
                            "y": (hash(object_id) % 1000) - 500,
                            "z": (hash(object_id) % 1000) - 500
                        },
                        "velocity": {
                            "x": (hash(object_id) % 10) * 0.1 - 0.5,
                            "y": 3.075 + (hash(object_id) % 10) * 0.01,  # GEO velocity in km/s
                            "z": (hash(object_id) % 10) * 0.01 - 0.05
                        },
                        "covariance": [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.001, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
                        ],
                        "confidence": 0.9,
                        "source": "EXAMPLE_APP"
                    }
                    
                    # Publish derived message (maintains traceability)
                    self.derive_message(
                        parent_message=message,
                        message_type="state.vector",
                        payload=state_vector_payload,
                        topic="ss2.data.state-vector",
                        key=object_id
                    )
    
    # Create and run the subsystem
    subsystem = CustomSubsystem2(kafka_config)
    subsystem.initialize()
    
    try:
        # Start consuming messages
        subsystem.start_consuming()
    except KeyboardInterrupt:
        logger.info("State Estimation subsystem stopped by user")
    finally:
        subsystem.stop()


def run_subsystem4(kafka_config: KafkaConfig) -> None:
    """
    Run the CCDM Detection subsystem.
    
    Args:
        kafka_config: Kafka configuration
    """
    # Override process_message to implement actual functionality
    class CustomSubsystem4(Subsystem4):
        # Track state vectors for CCDM detection
        _state_vectors = {}
        
        @trace_context
        def process_message(self, message: Dict[str, Any]) -> None:
            header = message.get("header", {})
            payload = message.get("payload", {})
            message_type = header.get("messageType", "unknown")
            
            logger.info(f"Processing {message_type} message in CCDM Detection subsystem")
            
            if message_type == "state.vector":
                object_id = payload.get("objectId")
                
                if not object_id:
                    return
                
                # Check if we have a previous state vector for this object
                if object_id in self._state_vectors:
                    prev_sv = self._state_vectors[object_id]
                    
                    # Calculate velocity change (simple detection)
                    prev_vx = prev_sv["payload"]["velocity"]["x"]
                    prev_vy = prev_sv["payload"]["velocity"]["y"]
                    prev_vz = prev_sv["payload"]["velocity"]["z"]
                    
                    curr_vx = payload["velocity"]["x"]
                    curr_vy = payload["velocity"]["y"]
                    curr_vz = payload["velocity"]["z"]
                    
                    # Calculate delta-V (very simplified)
                    delta_v = ((curr_vx - prev_vx) ** 2 + 
                               (curr_vy - prev_vy) ** 2 + 
                               (curr_vz - prev_vz) ** 2) ** 0.5
                    
                    # Check if delta-V is significant (arbitrary threshold)
                    if delta_v > 0.1:
                        # Detected a maneuver - generate CCDM detection
                        ccdm_payload = {
                            "detectionId": f"ccdm-{int(time.time())}",
                            "objectId": object_id,
                            "detectionTime": datetime.utcnow().isoformat(),
                            "ccdmType": "MANEUVERING",
                            "confidence": min(delta_v * 5, 0.95),  # Arbitrary confidence
                            "indicators": [
                                {
                                    "indicatorId": f"ind-{int(time.time())}",
                                    "indicatorType": "UNEXPECTED_MANEUVER",
                                    "value": delta_v,
                                    "confidence": min(delta_v * 5, 0.95),
                                    "description": f"Delta-V of {delta_v:.3f} km/s detected"
                                }
                            ],
                            "evidenceData": {
                                "stateVectorIds": [
                                    prev_sv["payload"]["stateVectorId"],
                                    payload["stateVectorId"]
                                ]
                            },
                            "assessment": {
                                "threatLevel": "MEDIUM",
                                "intentAssessment": "UNCERTAIN"
                            }
                        }
                        
                        # Publish CCDM detection
                        self.derive_message(
                            parent_message=message,
                            message_type="ccdm.detection",
                            payload=ccdm_payload,
                            topic="ss4.ccdm.detection",
                            key=object_id
                        )
                
                # Update the stored state vector
                self._state_vectors[object_id] = message
    
    # Create and run the subsystem
    subsystem = CustomSubsystem4(kafka_config)
    subsystem.initialize()
    
    try:
        # Start consuming messages
        subsystem.start_consuming()
    except KeyboardInterrupt:
        logger.info("CCDM Detection subsystem stopped by user")
    finally:
        subsystem.stop()


def run_subsystem6(kafka_config: KafkaConfig) -> None:
    """
    Run the Threat Assessment subsystem.
    
    Args:
        kafka_config: Kafka configuration
    """
    # Override process_message to implement actual functionality
    class CustomSubsystem6(Subsystem6):
        @trace_context
        def process_message(self, message: Dict[str, Any]) -> None:
            header = message.get("header", {})
            payload = message.get("payload", {})
            message_type = header.get("messageType", "unknown")
            
            logger.info(f"Processing {message_type} message in Threat Assessment subsystem")
            
            if message_type == "ccdm.detection":
                object_id = payload.get("objectId")
                ccdm_type = payload.get("ccdmType")
                confidence = payload.get("confidence", 0.5)
                
                if not object_id:
                    return
                
                # Generate threat assessment based on CCDM detection
                threat_level = "LOW"
                if ccdm_type == "MANEUVERING":
                    if confidence > 0.8:
                        threat_level = "HIGH"
                    elif confidence > 0.5:
                        threat_level = "MEDIUM"
                
                # Generate recommended actions
                recommended_actions = ["Continue monitoring"]
                if threat_level == "MEDIUM":
                    recommended_actions.append("Schedule follow-up observations")
                    recommended_actions.append("Alert space operations center")
                elif threat_level == "HIGH":
                    recommended_actions.append("Increase monitoring frequency")
                    recommended_actions.append("Alert space operations center")
                    recommended_actions.append("Analyze recent trajectory changes")
                
                # Create threat assessment payload
                assessment_payload = {
                    "assessmentId": f"ta-{int(time.time())}",
                    "objectId": object_id,
                    "assessmentTime": datetime.utcnow().isoformat(),
                    "threatLevel": threat_level,
                    "confidence": confidence,
                    "source": "CCDM_ANALYSIS",
                    "description": f"Threat assessment based on {ccdm_type} detection",
                    "relatedDetections": [
                        {
                            "detectionId": payload.get("detectionId"),
                            "detectionType": "CCDM",
                            "weight": 1.0
                        }
                    ],
                    "recommendedActions": recommended_actions
                }
                
                # Publish the threat assessment
                self.derive_message(
                    parent_message=message,
                    message_type="threat.assessment",
                    payload=assessment_payload,
                    topic="ss6.threat.assessment",
                    key=object_id
                )
    
    # Create and run the subsystem
    subsystem = CustomSubsystem6(kafka_config)
    subsystem.initialize()
    
    try:
        # Start consuming messages
        subsystem.start_consuming()
    except KeyboardInterrupt:
        logger.info("Threat Assessment subsystem stopped by user")
    finally:
        subsystem.stop()


def main() -> None:
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AstroShield Example Application')
    parser.add_argument(
        '--subsystem',
        type=int,
        choices=[0, 1, 2, 4, 6],
        help='Run a specific subsystem (0=Data Ingestion, 1=Target Modeling, 2=State Estimation, 4=CCDM, 6=Threat Assessment)'
    )
    
    args = parser.parse_args()
    
    # Load Kafka configuration
    kafka_config = load_kafka_config()
    
    # Run the selected subsystem or all if none specified
    if args.subsystem == 0:
        logger.info("Running Data Ingestion subsystem")
        run_subsystem0(kafka_config)
    elif args.subsystem == 1:
        logger.info("Running Target Modeling subsystem")
        run_subsystem1(kafka_config)
    elif args.subsystem == 2:
        logger.info("Running State Estimation subsystem")
        run_subsystem2(kafka_config)
    elif args.subsystem == 4:
        logger.info("Running CCDM Detection subsystem")
        run_subsystem4(kafka_config)
    elif args.subsystem == 6:
        logger.info("Running Threat Assessment subsystem")
        run_subsystem6(kafka_config)
    else:
        # Run all subsystems in separate threads
        logger.info("Running all subsystems")
        
        threads = [
            threading.Thread(target=run_subsystem0, args=(kafka_config,), name="Subsystem0"),
            threading.Thread(target=run_subsystem1, args=(kafka_config,), name="Subsystem1"),
            threading.Thread(target=run_subsystem2, args=(kafka_config,), name="Subsystem2"),
            threading.Thread(target=run_subsystem4, args=(kafka_config,), name="Subsystem4"),
            threading.Thread(target=run_subsystem6, args=(kafka_config,), name="Subsystem6"),
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        try:
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
            sys.exit(0)


if __name__ == "__main__":
    main() 