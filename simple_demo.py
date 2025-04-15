#!/usr/bin/env python3
"""
Simple demo of event-driven architecture for Astroshield.
This is a self-contained script that simulates the event processing system.
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("astroshield")

#
# Mock classes to simulate the full system
#

class MockProducer:
    """Mock Kafka producer that prints messages to console."""
    
    def __init__(self):
        self.sent_messages = []
    
    async def send_async(self, topic, value, key=None, headers=None):
        """Print message details and add to sent_messages."""
        self.sent_messages.append({
            'topic': topic,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"\n📨 Message published to {topic}:")
        print(f"  {json.dumps(value, indent=2)[:300]}...")
        return True

class DMDOrbitDeterminationClient:
    """Simplified mock client for DMD OD API."""
    
    async def detect_maneuvers_from_states(self, catalog_id, time_window=24):
        """Mock implementation of maneuver detection."""
        # Simulate processing delay
        await asyncio.sleep(0.5)
        
        # 30% chance of detecting a maneuver
        detected = random.random() < 0.3
        
        if not detected:
            logger.info(f"No maneuver detected for {catalog_id}")
            return {
                "detected": False,
                "reason": random.choice(["no_significant_changes", "insufficient_data"]),
                "catalog_id": catalog_id
            }
        
        # Generate realistic maneuver data
        maneuver_types = [
            ("STATIONKEEPING", 0.005, 0.6),
            ("ORBIT_MAINTENANCE", 0.05, 0.7),
            ("ORBIT_ADJUSTMENT", 0.2, 0.85),
            ("MAJOR_MANEUVER", 0.5, 0.95)
        ]
        
        maneuver_type, base_delta_v, base_confidence = random.choice(maneuver_types)
        
        # Add some random variation
        delta_v = base_delta_v * random.uniform(0.8, 1.2)
        confidence = min(0.95, base_confidence * random.uniform(0.9, 1.1))
        
        logger.info(f"Detected {maneuver_type} for {catalog_id} with delta-v: {delta_v:.6f}")
        
        return {
            "detected": True,
            "catalog_id": catalog_id,
            "delta_v": delta_v,
            "time": (datetime.utcnow() - timedelta(hours=random.uniform(1, time_window))).isoformat(),
            "maneuver_type": maneuver_type,
            "confidence": confidence,
            "analysis_window_hours": time_window
        }

class WeatherDataService:
    """Simplified mock weather data service."""
    
    def analyze_observation_conditions(self, weather_data, object_data=None):
        """Analyze observation conditions from weather data."""
        # Extract key weather metrics
        location = weather_data.get("location", {})
        conditions = weather_data.get("conditions", {})
        
        cloud_cover = conditions.get("clouds", {}).get("coverage", 0.0)
        visibility_km = conditions.get("visibility", {}).get("value", 10.0)
        
        # Calculate quality factors
        cloud_factor = 1.0 - (cloud_cover / 100.0) if cloud_cover <= 100 else 0.0
        visibility_factor = min(1.0, visibility_km / 10.0)
        
        # Calculate overall quality score
        quality_score = (cloud_factor * 0.7 + visibility_factor * 0.3)
        
        # Determine quality category
        quality_category = "EXCELLENT" if quality_score > 0.8 else \
                          "GOOD" if quality_score > 0.6 else \
                          "FAIR" if quality_score > 0.4 else \
                          "POOR"
        
        # Determine go/no-go recommendation
        recommendation = "GO" if quality_score > 0.5 else "NO_GO"
        
        logger.info(f"Weather analysis: {quality_category} (score: {quality_score:.2f})")
        
        # Prepare results
        results = {
            "analysis_time": datetime.utcnow().isoformat(),
            "location": location,
            "observation_quality": {
                "score": quality_score,
                "category": quality_category,
                "recommendation": recommendation
            }
        }
        
        # Add observation window if conditions are favorable
        if quality_score > 0.5:
            now = datetime.utcnow()
            window_start = now + timedelta(minutes=30)
            window_duration = timedelta(minutes=max(10, int(quality_score * 60)))
            window_end = window_start + window_duration
            
            results["observation_window"] = {
                "start_time": window_start.isoformat(),
                "end_time": window_end.isoformat(),
                "duration_minutes": window_duration.total_seconds() / 60
            }
            
        # Add object info if provided
        if object_data:
            results["object_info"] = object_data
            
        return results

#
# Event handlers
#

class DMDOrbitDeterminationEventHandler:
    """Handler for DMD orbit determination events."""
    
    def __init__(self, producer):
        self.producer = producer
        self.dmd_client = DMDOrbitDeterminationClient()
    
    async def handle_event(self, event):
        """Handle DMD orbit determination events."""
        try:
            # Extract catalog ID from event payload
            payload = event.get("payload", {})
            if not payload:
                logger.warning("No payload in DMD orbit determination event")
                return
            
            catalog_id = payload.get("object_id") or payload.get("catalogId")
            
            if not catalog_id:
                logger.warning("No catalog ID found in DMD orbit determination event")
                return
            
            # Log the event
            logger.info(f"Processing DMD orbit determination event for object: {catalog_id}")
            
            # Call the DMD client to detect maneuvers
            maneuver_result = await self.dmd_client.detect_maneuvers_from_states(catalog_id)
            
            # Check if a maneuver was detected
            if maneuver_result.get("detected", False):
                logger.info(f"Maneuver detected for {catalog_id}: {maneuver_result.get('maneuver_type')}")
                
                # If a maneuver was detected, publish a maneuver event
                if self.producer:
                    await self.publish_maneuver_event(maneuver_result)
            else:
                logger.info(f"No maneuver detected for {catalog_id}: {maneuver_result.get('reason', 'unknown reason')}")
        
        except Exception as e:
            logger.error(f"Error handling DMD orbit determination event: {str(e)}")
    
    async def publish_maneuver_event(self, maneuver_data):
        """Publish a maneuver event."""
        # Prepare the maneuver event
        maneuver_event = {
            "header": {
                "messageType": "maneuver-detected",
                "source": "dmd-od-integration",
                "timestamp": datetime.utcnow().isoformat()
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
        await self.producer.send_async("maneuvers-detected", maneuver_event)

class WeatherDataEventHandler:
    """Handler for weather data events."""
    
    def __init__(self, producer):
        self.producer = producer
        self.weather_service = WeatherDataService()
    
    async def handle_event(self, event):
        """Handle weather data events."""
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
                    "altitude_km": payload["targetObject"].get("altitude", 0.0)
                }
            
            # Analyze observation conditions
            analysis_result = self.weather_service.analyze_observation_conditions(payload, object_info)
            
            # If analysis indicates favorable conditions, publish an event
            if analysis_result["observation_quality"]["recommendation"] == "GO":
                await self.publish_observation_recommendation(analysis_result)
        
        except Exception as e:
            logger.error(f"Error handling weather data event: {str(e)}")
    
    async def publish_observation_recommendation(self, analysis_result):
        """Publish an observation recommendation event."""
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
        await self.producer.send_async("observation-windows", recommendation_event)

#
# Event generators
#

def generate_dmd_update_event(object_id=None):
    """Generate a sample DMD object update event."""
    if not object_id:
        object_id = f"DMD-{random.randint(1000, 9999)}"
    
    return {
        "header": {
            "messageType": "dmd-object-update",
            "source": "dmd-catalog",
            "timestamp": datetime.utcnow().isoformat()
        },
        "payload": {
            "object_id": object_id,
            "catalogId": object_id,
            "updateType": random.choice(["NEW_OBSERVATION", "ORBIT_UPDATED", "STATE_VECTOR_UPDATED"]),
            "updateTime": datetime.utcnow().isoformat(),
            "source": "DMD"
        }
    }

def generate_weather_data_event(location=None, include_target=True):
    """Generate a sample weather data event."""
    if not location:
        # Random location in continental US
        latitude = random.uniform(25.0, 49.0)
        longitude = random.uniform(-125.0, -66.0)
    else:
        latitude, longitude = location
    
    # Generate cloud cover with some bias toward clear or overcast
    cloud_distribution = [0, 10, 20, 30, 70, 80, 90, 100]  # More likely to be very clear or very cloudy
    cloud_cover = random.choice(cloud_distribution)
    
    event = {
        "header": {
            "messageType": "weather-data-update",
            "source": "earthcast-api",
            "timestamp": datetime.utcnow().isoformat()
        },
        "payload": {
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "locationName": f"Site-{random.randint(1, 100)}"
            },
            "conditions": {
                "clouds": {
                    "coverage": cloud_cover,
                    "type": "CUMULUS" if cloud_cover < 50 else "STRATUS"
                },
                "visibility": {
                    "value": random.uniform(1.0, 15.0),
                    "units": "km"
                },
                "precipitation": {
                    "type": "NONE" if cloud_cover < 60 else random.choice(["NONE", "DRIZZLE", "RAIN"]),
                    "intensity": 0 if cloud_cover < 60 else random.uniform(0, 5.0)
                }
            }
        }
    }
    
    # Optionally add target object
    if include_target:
        event["payload"]["targetObject"] = {
            "catalogId": f"SAT-{random.randint(1000, 9999)}",
            "altitude": random.uniform(400, 1200),
            "objectType": random.choice(["PAYLOAD", "ROCKET_BODY", "DEBRIS"])
        }
    
    return event

#
# Main demo function
#

async def run_demo(num_events=5, delay_seconds=2):
    """Run the event processing demo."""
    # Create producer and handlers
    producer = MockProducer()
    dmd_handler = DMDOrbitDeterminationEventHandler(producer)
    weather_handler = WeatherDataEventHandler(producer)
    
    print(f"""
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ 🛰️  Astroshield Event-Driven Architecture Demo       ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """)
    
    print(f"🚀 Starting event processing simulation ({num_events} events)")
    print("=" * 80)
    
    for i in range(num_events):
        print(f"\n[Event {i+1}/{num_events}]")
        
        # Alternate between DMD and weather events
        if i % 2 == 0:
            event = generate_dmd_update_event()
            print(f"Processing DMD update for {event['payload']['object_id']}")
            await dmd_handler.handle_event(event)
        else:
            event = generate_weather_data_event()
            target_info = event['payload'].get('targetObject', {})
            if target_info:
                print(f"Processing weather data for {target_info.get('catalogId', 'Unknown')}")
            else:
                print("Processing general weather data update")
            await weather_handler.handle_event(event)
        
        # Wait between events
        if i < num_events - 1:
            print(f"\nWaiting {delay_seconds} seconds for next event...")
            await asyncio.sleep(delay_seconds)
    
    print("\n" + "=" * 80)
    print(f"✅ Simulation complete: Processed {num_events} events")
    
    # Summarize results
    maneuver_events = [m for m in producer.sent_messages if m['topic'] == 'maneuvers-detected']
    window_events = [m for m in producer.sent_messages if m['topic'] == 'observation-windows']
    
    print(f"\n📊 Results Summary:")
    print(f"  - Maneuvers detected: {len(maneuver_events)}")
    print(f"  - Observation windows recommended: {len(window_events)}")

if __name__ == "__main__":
    # Get number of events from command line if provided
    num_events = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    delay = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    # Run the demo
    asyncio.run(run_demo(num_events, delay)) 