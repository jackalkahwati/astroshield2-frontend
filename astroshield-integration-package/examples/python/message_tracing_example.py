#!/usr/bin/env python3
"""
Message Tracing Example

This example demonstrates how to implement message tracing across multiple subsystems in AstroShield.
It shows how to maintain trace IDs and parent-child relationships between messages as they flow
through the processing chain from initial detection to final assessment.
"""

import json
import uuid
import logging
from datetime import datetime
from confluent_kafka import Consumer, Producer, KafkaError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_CONFIG = {
    'bootstrap.servers': 'kafka.astroshield.com:9093',
    'security.protocol': 'SASL_SSL',
    'sasl.mechanism': 'PLAIN',
    'sasl.username': 'your-username',  # Replace with your username
    'sasl.password': 'your-password',  # Replace with your password
    'group.id': 'message-tracing-example',
    'auto.offset.reset': 'earliest'
}

# Topics for different subsystems
TOPICS = {
    'ss2_state_vector': 'ss2.data.state-vector',
    'ss4_ccdm_detection': 'ss4.ccdm.detection',
    'ss6_threat_assessment': 'ss6.threat.assessment'
}

def delivery_report(err, msg):
    """Callback for message delivery reports."""
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

def generate_trace_id():
    """Generate a new trace ID for starting a new processing chain."""
    return str(uuid.uuid4())

def generate_message_id():
    """Generate a new message ID."""
    return str(uuid.uuid4())

def create_state_vector_message(object_id, trace_id=None):
    """
    Create a state vector message (Subsystem 2).
    This would typically be the start of a processing chain.
    """
    if trace_id is None:
        trace_id = generate_trace_id()
        
    message = {
        "header": {
            "messageId": generate_message_id(),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "message-tracing-example",
            "messageType": "state.vector",
            "traceId": trace_id,
            "parentMessageIds": []  # Empty since this is the start of the chain
        },
        "payload": {
            "stateVectorId": f"sv-{generate_message_id()}",
            "objectId": object_id,
            "epoch": datetime.utcnow().isoformat(),
            "referenceFrame": "GCRF",
            "position": {
                "x": 42164.0,  # Example GEO position in km
                "y": 0.0,
                "z": 0.0
            },
            "velocity": {
                "x": 0.0,
                "y": 3.075,  # Example GEO velocity in km/s
                "z": 0.0
            },
            "covariance": [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.001, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
            ],
            "metadata": {
                "source": "example-generator",
                "confidenceLevel": 0.95
            }
        }
    }
    
    logger.info(f"Created state vector message with traceId: {trace_id}")
    return message

def process_state_vector_to_ccdm(state_vector_message):
    """
    Process a state vector message to produce a CCDM detection (Subsystem 4).
    This demonstrates maintaining the trace ID and adding the parent message ID.
    """
    # Extract trace ID and message ID from the incoming message
    trace_id = state_vector_message["header"]["traceId"]
    parent_message_id = state_vector_message["header"]["messageId"]
    object_id = state_vector_message["payload"]["objectId"]
    
    # Create a CCDM detection message with the same trace ID
    ccdm_message = {
        "header": {
            "messageId": generate_message_id(),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "message-tracing-example",
            "messageType": "ccdm.detection",
            "traceId": trace_id,  # Maintain the same trace ID
            "parentMessageIds": [parent_message_id]  # Reference the parent message
        },
        "payload": {
            "detectionId": f"ccdm-{generate_message_id()}",
            "objectId": object_id,
            "detectionTime": datetime.utcnow().isoformat(),
            "ccdmType": "MANEUVERING",
            "confidence": 0.85,
            "indicators": [
                {
                    "indicatorId": f"ind-{generate_message_id()}",
                    "indicatorType": "UNEXPECTED_MANEUVER",
                    "value": 0.92,
                    "confidence": 0.85,
                    "description": "Unexpected delta-V detected"
                }
            ],
            "evidenceData": {
                "stateVectorIds": [state_vector_message["payload"]["stateVectorId"]]
            },
            "assessment": {
                "threatLevel": "MEDIUM",
                "intentAssessment": "UNCERTAIN"
            }
        }
    }
    
    logger.info(f"Processed state vector to CCDM detection with traceId: {trace_id}")
    logger.info(f"Parent message ID: {parent_message_id}")
    return ccdm_message

def process_ccdm_to_threat_assessment(ccdm_message):
    """
    Process a CCDM detection to produce a threat assessment (Subsystem 6).
    This demonstrates maintaining the trace chain across three subsystems.
    """
    # Extract trace ID and message ID from the incoming message
    trace_id = ccdm_message["header"]["traceId"]
    parent_message_id = ccdm_message["header"]["messageId"]
    parent_message_ids = ccdm_message["header"]["parentMessageIds"]
    object_id = ccdm_message["payload"]["objectId"]
    
    # Create a threat assessment message with the same trace ID
    threat_assessment = {
        "header": {
            "messageId": generate_message_id(),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "message-tracing-example",
            "messageType": "threat.assessment",
            "traceId": trace_id,  # Maintain the same trace ID
            "parentMessageIds": [parent_message_id] + parent_message_ids  # Include all ancestors
        },
        "payload": {
            "assessmentId": f"threat-{generate_message_id()}",
            "objectId": object_id,
            "assessmentTime": datetime.utcnow().isoformat(),
            "threatLevel": "MEDIUM",
            "confidence": 0.80,
            "source": "CCDM_ANALYSIS",
            "description": "Threat assessment based on MANEUVERING detection",
            "relatedDetections": [
                {
                    "detectionId": ccdm_message["payload"]["detectionId"],
                    "detectionType": "CCDM",
                    "weight": 1.0
                }
            ],
            "recommendedActions": [
                "Increase monitoring frequency",
                "Schedule follow-up observations",
                "Predict possible trajectories"
            ]
        }
    }
    
    logger.info(f"Processed CCDM detection to threat assessment with traceId: {trace_id}")
    logger.info(f"Parent message ID: {parent_message_id}")
    logger.info(f"Complete ancestry chain: {threat_assessment['header']['parentMessageIds']}")
    return threat_assessment

def trace_message_chain(producer=None):
    """
    Demonstrate a complete message trace chain across three subsystems.
    
    1. Generate a state vector (Subsystem 2)
    2. Process it to create a CCDM detection (Subsystem 4)
    3. Process that to create a threat assessment (Subsystem 6)
    
    All with proper tracing via trace IDs and parent message IDs.
    """
    # Create an object ID for our example
    object_id = f"SAT-{uuid.uuid4().hex[:8].upper()}"
    
    # Start with a state vector message
    state_vector = create_state_vector_message(object_id)
    
    # If a producer was provided, publish the message
    if producer:
        producer.produce(
            TOPICS['ss2_state_vector'],
            key=object_id,
            value=json.dumps(state_vector).encode('utf-8'),
            callback=delivery_report
        )
        producer.flush()
    
    # Process it to create a CCDM detection
    ccdm_detection = process_state_vector_to_ccdm(state_vector)
    
    # If a producer was provided, publish the message
    if producer:
        producer.produce(
            TOPICS['ss4_ccdm_detection'],
            key=object_id,
            value=json.dumps(ccdm_detection).encode('utf-8'),
            callback=delivery_report
        )
        producer.flush()
    
    # Process it to create a threat assessment
    threat_assessment = process_ccdm_to_threat_assessment(ccdm_detection)
    
    # If a producer was provided, publish the message
    if producer:
        producer.produce(
            TOPICS['ss6_threat_assessment'],
            key=object_id,
            value=json.dumps(threat_assessment).encode('utf-8'),
            callback=delivery_report
        )
        producer.flush()
    
    # Return the complete chain for inspection
    return {
        "state_vector": state_vector,
        "ccdm_detection": ccdm_detection,
        "threat_assessment": threat_assessment
    }

def trace_visualization(message_chain):
    """
    Visualize the message trace chain in a human-readable format.
    This is useful for debugging and understanding the flow.
    """
    print("\n" + "="*80)
    print("MESSAGE TRACE VISUALIZATION")
    print("="*80)
    
    state_vector = message_chain["state_vector"]
    ccdm_detection = message_chain["ccdm_detection"]
    threat_assessment = message_chain["threat_assessment"]
    
    trace_id = state_vector["header"]["traceId"]
    
    print(f"\nTrace ID: {trace_id}")
    print("\nProcessing Chain:")
    print(f"  1. State Vector (Subsystem 2)")
    print(f"     Message ID: {state_vector['header']['messageId']}")
    print(f"     Object ID: {state_vector['payload']['objectId']}")
    print(f"     Timestamp: {state_vector['header']['timestamp']}")
    print(f"     Parent IDs: {state_vector['header']['parentMessageIds'] or 'None (origin message)'}")
    
    print(f"\n  2. CCDM Detection (Subsystem 4)")
    print(f"     Message ID: {ccdm_detection['header']['messageId']}")
    print(f"     Detection ID: {ccdm_detection['payload']['detectionId']}")
    print(f"     CCDM Type: {ccdm_detection['payload']['ccdmType']}")
    print(f"     Timestamp: {ccdm_detection['header']['timestamp']}")
    print(f"     Parent IDs: {ccdm_detection['header']['parentMessageIds']}")
    
    print(f"\n  3. Threat Assessment (Subsystem 6)")
    print(f"     Message ID: {threat_assessment['header']['messageId']}")
    print(f"     Assessment ID: {threat_assessment['payload']['assessmentId']}")
    print(f"     Threat Level: {threat_assessment['payload']['threatLevel']}")
    print(f"     Timestamp: {threat_assessment['header']['timestamp']}")
    print(f"     Parent IDs: {threat_assessment['header']['parentMessageIds']}")
    
    print("\nComplete Trace Summary:")
    print(f"  Origin: State Vector {state_vector['header']['messageId']}")
    print(f"  ↓")
    print(f"  Process: State Vector → CCDM Detection {ccdm_detection['header']['messageId']}")
    print(f"  ↓")
    print(f"  Final: CCDM Detection → Threat Assessment {threat_assessment['header']['messageId']}")
    
    print("\nRecommended Actions:")
    for action in threat_assessment["payload"]["recommendedActions"]:
        print(f"  - {action}")
    
    print("="*80 + "\n")

def main():
    """Main function to run the message tracing example."""
    try:
        # Create a Producer instance
        producer_config = KAFKA_CONFIG.copy()
        producer_config.pop('group.id', None)
        producer_config.pop('auto.offset.reset', None)
        
        # Comment out the next line to run in "simulation mode" without actually producing messages
        # producer = Producer(producer_config)
        producer = None  # Simulation mode
        
        # Demonstrate message tracing
        message_chain = trace_message_chain(producer)
        
        # Visualize the trace chain
        trace_visualization(message_chain)
        
        # Output the complete message chain to a file for reference
        with open('message_trace_example.json', 'w') as f:
            json.dump(message_chain, f, indent=2)
            logger.info("Wrote complete message chain to message_trace_example.json")
        
        logger.info("Message tracing example completed successfully.")
    
    except Exception as e:
        logger.error(f"Error in message tracing example: {e}")
    
    finally:
        if producer is not None:
            # Ensure all messages are delivered before exiting
            producer.flush()

if __name__ == "__main__":
    main() 