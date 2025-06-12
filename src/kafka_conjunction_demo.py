#!/usr/bin/env python3
"""
AstroShield Kafka Conjunction Assessment Demo
Demonstrates minimal topic set for delivering TBD capability
"""

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
from confluent_kafka import Consumer, Producer, KafkaError
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StateVector:
    """State vector from ss2.data.state-vector"""
    object_id: str
    epoch: str
    position: List[float]  # [x, y, z] in km
    velocity: List[float]  # [vx, vy, vz] in km/s
    covariance: Optional[List[float]] = None

@dataclass
class ObjectMetadata:
    """Object info from ss1.tmdb.object-updated"""
    object_id: str
    object_type: str
    rcs: float  # Radar cross section
    mass: float
    owner: str
    maneuverable: bool

@dataclass
class ConjunctionPrediction:
    """Output to ss5.pez-wez-prediction.conjunction"""
    primary_object: str
    secondary_object: str
    tca: str  # Time of closest approach
    probability: float
    miss_distance: float  # km
    relative_velocity: float  # km/s
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL

@dataclass
class ProximityAlert:
    """Output to ss4.indicators.proximity-events-valid-remote-sense"""
    event_id: str
    objects: List[str]
    detection_time: str
    closest_approach: float  # km
    validated: bool
    confidence: float

@dataclass
class UIEvent:
    """Output to ui.event"""
    event_type: str
    severity: str
    title: str
    description: str
    recommended_action: str
    data: Dict

class ConjunctionAssessmentDemo:
    def __init__(self, kafka_config: Dict):
        """Initialize Kafka consumers and producers"""
        self.consumer_config = {
            **kafka_config,
            'group.id': 'astroshield-conjunction',
            'auto.offset.reset': 'latest'
        }
        self.producer_config = kafka_config
        
        # Subscribe to input topics
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([
            'ss2.data.state-vector',
            'ss2.data.observation-track',
            'ss1.tmdb.object-updated'
        ])
        
        self.producer = Producer(self.producer_config)
        
        # State storage
        self.state_vectors: Dict[str, StateVector] = {}
        self.object_metadata: Dict[str, ObjectMetadata] = {}
        
    def process_messages(self):
        """Main processing loop"""
        logger.info("Starting conjunction assessment processing...")
        
        while True:
            msg = self.consumer.poll(1.0)
            
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"Kafka error: {msg.error()}")
                    break
                    
            # Process based on topic
            topic = msg.topic()
            data = json.loads(msg.value().decode('utf-8'))
            
            if topic == 'ss2.data.state-vector':
                self.process_state_vector(data)
            elif topic == 'ss1.tmdb.object-updated':
                self.process_object_update(data)
                
            # Check for conjunctions after each update
            self.check_conjunctions()
            
    def process_state_vector(self, data: Dict):
        """Process incoming state vector"""
        sv = StateVector(**data)
        self.state_vectors[sv.object_id] = sv
        logger.debug(f"Updated state vector for {sv.object_id}")
        
    def process_object_update(self, data: Dict):
        """Process object metadata update"""
        obj = ObjectMetadata(**data)
        self.object_metadata[obj.object_id] = obj
        logger.debug(f"Updated metadata for {obj.object_id}")
        
    def check_conjunctions(self):
        """Check all object pairs for potential conjunctions"""
        objects = list(self.state_vectors.keys())
        
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1_id = objects[i]
                obj2_id = objects[j]
                
                if obj1_id in self.state_vectors and obj2_id in self.state_vectors:
                    self.assess_conjunction(
                        self.state_vectors[obj1_id],
                        self.state_vectors[obj2_id]
                    )
                    
    def assess_conjunction(self, sv1: StateVector, sv2: StateVector):
        """Assess conjunction between two objects"""
        # Calculate relative position and velocity
        rel_pos = np.array(sv1.position) - np.array(sv2.position)
        rel_vel = np.array(sv1.velocity) - np.array(sv2.velocity)
        
        # Simple closest approach calculation
        miss_distance = np.linalg.norm(rel_pos)
        relative_velocity = np.linalg.norm(rel_vel)
        
        # Time to closest approach (simplified)
        if relative_velocity > 0:
            tca_seconds = -np.dot(rel_pos, rel_vel) / (relative_velocity ** 2)
            if tca_seconds > 0 and tca_seconds < 86400:  # Within 24 hours
                # Calculate probability (simplified model)
                probability = self.calculate_collision_probability(
                    miss_distance, sv1, sv2
                )
                
                if probability > 1e-6:  # Threshold for concern
                    self.generate_alerts(
                        sv1, sv2, miss_distance, 
                        relative_velocity, probability, tca_seconds
                    )
                    
    def calculate_collision_probability(self, miss_distance: float, 
                                      sv1: StateVector, sv2: StateVector) -> float:
        """Calculate collision probability using simplified 2D Pc method"""
        # Combined object radius (simplified)
        combined_radius = 0.010  # 10 meters in km
        
        # Simplified probability based on miss distance
        if miss_distance < combined_radius:
            return 1.0
        elif miss_distance < 1.0:  # Within 1 km
            return np.exp(-miss_distance ** 2 / (2 * 0.1 ** 2))
        else:
            return 0.0
            
    def generate_alerts(self, sv1: StateVector, sv2: StateVector,
                       miss_distance: float, relative_velocity: float,
                       probability: float, tca_seconds: float):
        """Generate and publish alerts for conjunction"""
        
        # Determine risk level
        if probability > 1e-3:
            risk_level = "CRITICAL"
            severity = "critical"
        elif probability > 1e-4:
            risk_level = "HIGH"
            severity = "high"
        elif probability > 1e-5:
            risk_level = "MEDIUM"
            severity = "medium"
        else:
            risk_level = "LOW"
            severity = "low"
            
        current_time = datetime.now(timezone.utc)
        tca_time = current_time.timestamp() + tca_seconds
        
        # Create conjunction prediction
        conjunction = ConjunctionPrediction(
            primary_object=sv1.object_id,
            secondary_object=sv2.object_id,
            tca=datetime.fromtimestamp(tca_time, timezone.utc).isoformat(),
            probability=probability,
            miss_distance=miss_distance,
            relative_velocity=relative_velocity,
            risk_level=risk_level
        )
        
        # Create proximity alert
        alert = ProximityAlert(
            event_id=f"CONJ-{sv1.object_id}-{sv2.object_id}-{int(time.time())}",
            objects=[sv1.object_id, sv2.object_id],
            detection_time=current_time.isoformat(),
            closest_approach=miss_distance,
            validated=True,
            confidence=0.95
        )
        
        # Create UI event
        ui_event = UIEvent(
            event_type="conjunction_alert",
            severity=severity,
            title=f"Conjunction Alert: {sv1.object_id} - {sv2.object_id}",
            description=f"Potential conjunction in {tca_seconds/3600:.1f} hours. "
                       f"Miss distance: {miss_distance:.3f} km, "
                       f"Probability: {probability:.2e}",
            recommended_action=self.get_recommended_action(risk_level),
            data={
                "conjunction": asdict(conjunction),
                "metadata": {
                    "obj1_type": self.object_metadata.get(sv1.object_id, {}).get("object_type", "UNKNOWN"),
                    "obj2_type": self.object_metadata.get(sv2.object_id, {}).get("object_type", "UNKNOWN")
                }
            }
        )
        
        # Publish to Kafka topics
        self.publish_message('ss5.pez-wez-prediction.conjunction', asdict(conjunction))
        self.publish_message('ss4.indicators.proximity-events-valid-remote-sense', asdict(alert))
        self.publish_message('ui.event', asdict(ui_event))
        
        logger.info(f"Published {risk_level} conjunction alert: {sv1.object_id} - {sv2.object_id}")
        
    def get_recommended_action(self, risk_level: str) -> str:
        """Get recommended action based on risk level"""
        actions = {
            "CRITICAL": "IMMEDIATE ACTION REQUIRED: Initiate collision avoidance maneuver",
            "HIGH": "Prepare collision avoidance maneuver options",
            "MEDIUM": "Monitor closely and refine orbit determination",
            "LOW": "Continue routine monitoring"
        }
        return actions.get(risk_level, "Monitor situation")
        
    def publish_message(self, topic: str, data: Dict):
        """Publish message to Kafka topic"""
        self.producer.produce(
            topic,
            key=data.get('object_id', data.get('event_id', '')),
            value=json.dumps(data).encode('utf-8'),
            callback=self.delivery_report
        )
        self.producer.poll(0)
        
    def delivery_report(self, err, msg):
        """Callback for message delivery confirmation"""
        if err is not None:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.debug(f'Message delivered to {msg.topic()} [{msg.partition()}]')
            
    def run(self):
        """Run the demonstration"""
        try:
            self.process_messages()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.consumer.close()
            self.producer.flush()

def main():
    """Main entry point"""
    # Kafka configuration
    kafka_config = {
        'bootstrap.servers': 'localhost:9092',
        'security.protocol': 'SASL_SSL',
        'sasl.mechanism': 'PLAIN',
        'sasl.username': 'astroshield',
        'sasl.password': 'your-password-here'
    }
    
    # Create and run demo
    demo = ConjunctionAssessmentDemo(kafka_config)
    demo.run()

if __name__ == "__main__":
    main() 