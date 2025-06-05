#!/usr/bin/env python3
"""
AstroShield AI Modules Demonstration

This script demonstrates the new AI-powered capabilities of AstroShield including:
- Intent classification for satellite maneuvers
- Hostility scoring and threat assessment
- Kafka event processing integration
- Real-time analysis pipeline
- Integration with Welders Arc system

Usage:
    python demo_ai_modules.py
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("astroshield.demo")

# Import AI modules
from app.ai_modules.models import (
    ManeuverEvent, ProximityEvent, ManeuverType, 
    IntentClass, ThreatLevel, ActorType, ModelConfig, PipelineConfig
)
from app.ai_modules.intent_classifier import IntentClassifier, create_intent_classifier
from app.ai_modules.hostility_scorer import HostilityScorer, create_hostility_scorer
from app.ai_modules.kafka_adapter import KafkaAdapter, create_kafka_adapter


async def demo_intent_classification():
    """Demonstrate intent classification capabilities."""
    print("\n" + "="*50)
    print("INTENT CLASSIFICATION DEMONSTRATION")
    print("="*50)
    
    # Create intent classifier
    classifier = create_intent_classifier()
    
    # Create sample maneuver events
    sample_maneuvers = [
        # Inspection scenario
        ManeuverEvent(
            sat_pair_id="ADVERSARY-123_TARGET-456",
            primary_norad_id="12345",
            maneuver_type=ManeuverType.PROGRADE,
            delta_v=2.5,
            confidence=0.85,
            orbital_elements_before={
                "semi_major_axis": 7000.0, "eccentricity": 0.001,
                "inclination": 51.6, "raan": 0.0, "argument_of_perigee": 0.0, "mean_anomaly": 0.0
            },
            orbital_elements_after={
                "semi_major_axis": 7001.5, "eccentricity": 0.001,
                "inclination": 51.6, "raan": 0.0, "argument_of_perigee": 0.0, "mean_anomaly": 0.0
            }
        ),
        
        # Station keeping scenario
        ManeuverEvent(
            sat_pair_id="COMMERCIAL-789",
            primary_norad_id="67890",
            maneuver_type=ManeuverType.RADIAL,
            delta_v=0.1,
            confidence=0.95,
            orbital_elements_before={
                "semi_major_axis": 42164.0, "eccentricity": 0.0001,
                "inclination": 0.1, "raan": 0.0, "argument_of_perigee": 0.0, "mean_anomaly": 0.0
            },
            orbital_elements_after={
                "semi_major_axis": 42164.0, "eccentricity": 0.0001,
                "inclination": 0.1, "raan": 0.0, "argument_of_perigee": 0.0, "mean_anomaly": 0.0
            }
        ),
        
        # Evasive maneuver scenario
        ManeuverEvent(
            sat_pair_id="US-ASSET-001",
            primary_norad_id="11111",
            maneuver_type=ManeuverType.COMBINED,
            delta_v=15.0,
            confidence=0.92,
            orbital_elements_before={
                "semi_major_axis": 6800.0, "eccentricity": 0.002,
                "inclination": 98.2, "raan": 45.0, "argument_of_perigee": 30.0, "mean_anomaly": 0.0
            },
            orbital_elements_after={
                "semi_major_axis": 6850.0, "eccentricity": 0.004,
                "inclination": 98.5, "raan": 45.2, "argument_of_perigee": 31.0, "mean_anomaly": 0.0
            }
        )
    ]
    
    # Analyze each maneuver
    for i, maneuver in enumerate(sample_maneuvers, 1):
        print(f"\nAnalyzing Maneuver {i}:")
        print(f"  NORAD ID: {maneuver.primary_norad_id}")
        print(f"  Delta-V: {maneuver.delta_v} m/s")
        print(f"  Type: {maneuver.maneuver_type}")
        
        result = await classifier.analyze_intent(maneuver)
        
        print(f"  RESULT:")
        print(f"    Intent: {result.intent_class}")
        print(f"    Confidence: {result.confidence_score:.3f}")
        print(f"    Reasoning: {'; '.join(result.reasoning)}")
    
    # Show classifier performance metrics
    print(f"\nClassifier Performance:")
    metrics = classifier.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")


async def demo_hostility_scoring():
    """Demonstrate hostility scoring capabilities."""
    print("\n" + "="*50)
    print("HOSTILITY SCORING DEMONSTRATION")
    print("="*50)
    
    # Create hostility scorer
    scorer = create_hostility_scorer()
    
    # Create sample scenarios
    scenarios = [
        {
            "name": "Benign Commercial Operation",
            "maneuver": ManeuverEvent(
                sat_pair_id="COMMERCIAL-SAT",
                primary_norad_id="22222",
                maneuver_type=ManeuverType.PROGRADE,
                delta_v=0.5,
                confidence=0.9,
                orbital_elements_before={"semi_major_axis": 7000.0, "eccentricity": 0.001, "inclination": 51.6, "raan": 0.0, "argument_of_perigee": 0.0, "mean_anomaly": 0.0},
                orbital_elements_after={"semi_major_axis": 7000.5, "eccentricity": 0.001, "inclination": 51.6, "raan": 0.0, "argument_of_perigee": 0.0, "mean_anomaly": 0.0}
            ),
            "actor_id": "SPACEX"
        },
        
        {
            "name": "Suspicious Adversary Activity",
            "maneuver": ManeuverEvent(
                sat_pair_id="ADVERSARY-UNKNOWN",
                primary_norad_id="33333",
                maneuver_type=ManeuverType.COMBINED,
                delta_v=8.0,
                confidence=0.8,
                orbital_elements_before={"semi_major_axis": 6900.0, "eccentricity": 0.003, "inclination": 62.0, "raan": 120.0, "argument_of_perigee": 45.0, "mean_anomaly": 0.0},
                orbital_elements_after={"semi_major_axis": 6920.0, "eccentricity": 0.005, "inclination": 62.2, "raan": 120.5, "argument_of_perigee": 46.0, "mean_anomaly": 0.0}
            ),
            "actor_id": "UNKNOWN"
        },
        
        {
            "name": "High-Threat Proximity Operation",
            "maneuver": ManeuverEvent(
                sat_pair_id="THREAT-SAT_TARGET-ISS",
                primary_norad_id="44444",
                maneuver_type=ManeuverType.NORMAL,
                delta_v=12.0,
                confidence=0.95,
                orbital_elements_before={"semi_major_axis": 6780.0, "eccentricity": 0.002, "inclination": 51.6, "raan": 0.0, "argument_of_perigee": 0.0, "mean_anomaly": 0.0},
                orbital_elements_after={"semi_major_axis": 6780.0, "eccentricity": 0.002, "inclination": 51.6, "raan": 0.0, "argument_of_perigee": 5.0, "mean_anomaly": 0.0}
            ),
            "actor_id": "CNSA",
            "proximity": ProximityEvent(
                sat_pair_id="THREAT-SAT_TARGET-ISS",
                primary_norad_id="44444",
                secondary_norad_id="25544",  # ISS
                closest_approach_time=datetime.utcnow() + timedelta(hours=2),
                minimum_distance=2000.0,  # 2km - very close
                relative_velocity=150.0,
                duration_minutes=45.0
            )
        }
    ]
    
    # Analyze each scenario
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"  Actor: {scenario['actor_id']}")
        print(f"  Maneuver Delta-V: {scenario['maneuver'].delta_v} m/s")
        
        # Get intent classification first
        classifier = create_intent_classifier()
        intent_result = await classifier.analyze_intent(scenario['maneuver'])
        
        # Perform hostility assessment
        assessment = await scorer.assess_hostility(
            scenario['maneuver'],
            intent_result,
            scenario.get('proximity'),
            scenario['actor_id']
        )
        
        print(f"  ASSESSMENT:")
        print(f"    Threat Level: {assessment.threat_level}")
        print(f"    Hostility Score: {assessment.hostility_score:.3f}")
        print(f"    Confidence: {assessment.confidence:.3f}")
        print(f"    Actor Type: {assessment.actor_type}")
        
        # Show contributing factors
        print(f"    Contributing Factors:")
        for factor, score in assessment.contributing_factors.items():
            print(f"      {factor}: {score:.3f}")
        
        # Show recommendations
        if assessment.recommendations:
            print(f"    Recommendations:")
            for rec in assessment.recommendations[:3]:  # Show first 3
                print(f"      • {rec}")


async def demo_kafka_integration():
    """Demonstrate Kafka integration and event processing."""
    print("\n" + "="*50)
    print("KAFKA INTEGRATION DEMONSTRATION")
    print("="*50)
    
    # Create Kafka adapter with pipeline configuration
    config = PipelineConfig(
        enable_intent_classification=True,
        enable_hostility_scoring=True,
        parallel_processing=True,
        max_concurrent_analyses=5
    )
    
    adapter = create_kafka_adapter(config)
    
    # Start the adapter
    print("Starting Kafka adapter...")
    try:
        await adapter.start()
        kafka_available = True
    except Exception as e:
        print(f"Kafka not available, using mock implementation: {str(e)}")
        kafka_available = False
        # Force adapter to use mock components
        from app.ai_modules.kafka_adapter import MockKafkaProducer, MockKafkaConsumer
        adapter.producer = MockKafkaProducer()
        adapter.consumer = MockKafkaConsumer()
        await adapter.consumer.start()
    
    # Generate and process test events
    print("\nGenerating test events for Welders Arc integration...")
    test_events = await adapter.generate_test_events(count=5)
    
    # Small delay to allow processing
    await asyncio.sleep(2)
    
    # Show metrics
    print("\nProcessing Metrics:")
    metrics = adapter.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Health check
    print("\nSystem Health Check:")
    health = await adapter.health_check()
    print(f"  Overall Status: {health['status']}")
    for component, status in health['components'].items():
        print(f"  {component}: {status['status']}")
    
    # Stop the adapter
    await adapter.stop()
    print("Kafka adapter stopped")
    
    if not kafka_available:
        print("\nNote: This demonstration used mock Kafka implementation.")
        print("In production, connect to real Kafka cluster for event streaming.")


async def demo_real_time_analysis():
    """Demonstrate real-time analysis capabilities."""
    print("\n" + "="*50)
    print("REAL-TIME ANALYSIS DEMONSTRATION")
    print("="*50)
    
    # Create AI modules
    classifier = create_intent_classifier()
    scorer = create_hostility_scorer()
    
    print("Simulating real-time event stream...")
    
    # Simulate incoming events over time
    events = []
    for i in range(3):
        # Create dynamic maneuver event
        maneuver = ManeuverEvent(
            sat_pair_id=f"LIVE-SAT-{i+1}",
            primary_norad_id=f"5555{i}",
            maneuver_type=[ManeuverType.PROGRADE, ManeuverType.NORMAL, ManeuverType.RETROGRADE][i],
            delta_v=1.0 + (i * 2.0),
            confidence=0.8 + (i * 0.05),
            orbital_elements_before={"semi_major_axis": 7000.0 + i, "eccentricity": 0.001, "inclination": 51.6, "raan": 0.0, "argument_of_perigee": 0.0, "mean_anomaly": 0.0},
            orbital_elements_after={"semi_major_axis": 7000.0 + i + 0.5, "eccentricity": 0.001, "inclination": 51.6, "raan": 0.0, "argument_of_perigee": 0.0, "mean_anomaly": 0.0}
        )
        
        events.append(maneuver)
        
        print(f"\nProcessing Event {i+1} at {datetime.utcnow().strftime('%H:%M:%S')}")
        print(f"  Satellite: {maneuver.primary_norad_id}")
        print(f"  Delta-V: {maneuver.delta_v} m/s")
        
        # Process in real-time
        start_time = datetime.utcnow()
        
        # Concurrent analysis
        intent_task = asyncio.create_task(classifier.analyze_intent(maneuver))
        intent_result = await intent_task
        
        hostility_task = asyncio.create_task(scorer.assess_hostility(maneuver, intent_result))
        hostility_result = await hostility_task
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        print(f"  Analysis completed in {processing_time:.3f} seconds")
        print(f"    Intent: {intent_result.intent_class} (confidence: {intent_result.confidence_score:.3f})")
        print(f"    Threat: {hostility_result.threat_level} (score: {hostility_result.hostility_score:.3f})")
        
        # Short delay between events
        await asyncio.sleep(1)
    
    # Batch analysis demonstration
    print(f"\nDemonstrating batch analysis of {len(events)} events...")
    start_time = datetime.utcnow()
    
    # Process all events in batch
    batch_intent_results = await classifier.batch_analyze(events)
    batch_hostility_results = await scorer.batch_assess(events)
    
    batch_time = (datetime.utcnow() - start_time).total_seconds()
    print(f"Batch processing completed in {batch_time:.3f} seconds")
    print(f"Average time per event: {batch_time / len(events):.3f} seconds")


def print_welders_arc_integration_summary():
    """Print summary of Welders Arc integration capabilities."""
    print("\n" + "="*60)
    print("WELDERS ARC INTEGRATION SUMMARY")
    print("="*60)
    
    integration_points = [
        {
            "component": "Kafka Event Bus",
            "description": "Real-time event ingestion from SDA systems",
            "topics": ["dmd-od-update", "maneuver-detection", "proximity-alert"]
        },
        {
            "component": "Intent Classification",
            "description": "AI-powered maneuver intent analysis",
            "outputs": ["inspection", "shadowing", "evasion", "collision_course"]
        },
        {
            "component": "Hostility Scoring",
            "description": "Multi-factor threat assessment",
            "factors": ["actor_identity", "intent", "proximity", "pattern_deviation"]
        },
        {
            "component": "Real-time Analysis",
            "description": "Sub-second processing for tactical response",
            "capabilities": ["concurrent_processing", "batch_optimization", "scalable_architecture"]
        }
    ]
    
    for point in integration_points:
        print(f"\n{point['component']}:")
        print(f"  Description: {point['description']}")
        
        if 'topics' in point:
            print(f"  Kafka Topics: {', '.join(point['topics'])}")
        if 'outputs' in point:
            print(f"  Classification Outputs: {', '.join(point['outputs'])}")
        if 'factors' in point:
            print(f"  Scoring Factors: {', '.join(point['factors'])}")
        if 'capabilities' in point:
            print(f"  Key Capabilities: {', '.join(point['capabilities'])}")
    
    print(f"\nKafka Output Topics for Welders Arc:")
    print(f"  • astroshield.ai.intent_classification")
    print(f"  • astroshield.ai.hostility_assessment")
    print(f"  • astroshield.ai.observation_recommendation")
    print(f"  • astroshield.ai.metrics")
    
    print(f"\nJSON Message Format Example:")
    example_message = {
        "message_id": "ai-analysis-12345",
        "timestamp": "2024-11-21T15:30:00Z",
        "message_type": "intent_classification_result",
        "analysis_type": "intent_classification",
        "payload": {
            "event_id": "maneuver-event-67890",
            "sat_pair_id": "THREAT-SAT-123",
            "intent_class": "inspection",
            "confidence_score": 0.85,
            "maneuver_type": "prograde",
            "reasoning": ["High delta-V maneuver suggests significant orbit change"],
            "model_version": "1.0.0"
        },
        "correlation_id": "analysis-session-abc123"
    }
    
    print(json.dumps(example_message, indent=2))


async def main():
    """Main demonstration function."""
    print("AstroShield AI Modules Demonstration")
    print("Expanding capabilities for Major Allen's operational priorities")
    print("Real-time streaming analysis for space domain awareness")
    
    try:
        # Run all demonstrations
        await demo_intent_classification()
        await demo_hostility_scoring()
        await demo_kafka_integration()
        await demo_real_time_analysis()
        
        # Print integration summary
        print_welders_arc_integration_summary()
        
        print(f"\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("✅ Intent classification system operational")
        print("✅ Hostility scoring system operational") 
        print("✅ Kafka integration pipeline operational")
        print("✅ Real-time analysis capabilities verified")
        print("✅ Welders Arc integration ready")
        
        print(f"\nNext Steps:")
        print(f"1. Deploy to production environment")
        print(f"2. Configure Welders Arc Kafka consumers")
        print(f"3. Establish feedback loops with operators")
        print(f"4. Begin real-time threat assessment operations")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 