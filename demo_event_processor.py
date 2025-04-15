#!/usr/bin/env python3
"""
Demo script to demonstrate real-time event processing in Astroshield.
This script simulates Kafka messages and shows how the system processes them.
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app.kafka.event_handlers import DMDOrbitDeterminationEventHandler, WeatherDataEventHandler
from app.kafka.producer import KafkaProducer

# Mock producer for demonstration
class DemoProducer:
    """Mock producer that prints messages to console."""
    
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
        print(f"  Payload: {json.dumps(value['payload'], indent=2)[:200]}...")
        return True
    
    async def start(self):
        """Mock start method."""
        print("Demo producer started")
    
    async def stop(self):
        """Mock stop method."""
        print("Demo producer stopped")

# Sample event generators
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
                    "type": "CUMULUS" if cloud_cover < 50 else "STRATUS",
                    "ceiling": {
                        "value": random.uniform(1000, 5000),
                        "units": "meters"
                    }
                },
                "visibility": {
                    "value": random.uniform(1.0, 15.0),
                    "units": "km"
                },
                "precipitation": {
                    "type": "NONE" if cloud_cover < 60 else random.choice(["NONE", "DRIZZLE", "RAIN"]),
                    "intensity": 0 if cloud_cover < 60 else random.uniform(0, 5.0),
                    "units": "mm/hr"
                },
                "temperature": {
                    "value": random.uniform(10, 30),
                    "units": "celsius"
                },
                "wind": {
                    "speed": random.uniform(0, 30),
                    "direction": random.uniform(0, 360),
                    "units": "kph"
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

async def process_events(num_events=5, delay_seconds=2):
    """Process a series of simulated events."""
    # Create handlers with demo producer
    producer = DemoProducer()
    await producer.start()
    
    dmd_handler = DMDOrbitDeterminationEventHandler(producer)
    weather_handler = WeatherDataEventHandler(producer)
    
    # Patch the DMD client to return realistic responses
    dmd_handler.dmd_client.detect_maneuvers_from_states = mock_detect_maneuvers
    
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
    print(f"✅ Processed {num_events} events successfully")
    await producer.stop()

# Mock method for DMD maneuver detection
async def mock_detect_maneuvers(catalog_id, time_window=24):
    """Mock implementation of detect_maneuvers_from_states."""
    # 30% chance of detecting a maneuver
    detected = random.random() < 0.3
    
    if not detected:
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
    
    # Simulate processing delay
    await asyncio.sleep(0.5)
    
    return {
        "detected": True,
        "catalog_id": catalog_id,
        "delta_v": delta_v,
        "time": (datetime.utcnow() - timedelta(hours=random.uniform(1, time_window))).isoformat(),
        "maneuver_type": maneuver_type,
        "confidence": confidence,
        "analysis_window_hours": time_window
    }

if __name__ == "__main__":
    print("""
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ 🛰️  Astroshield Event-Driven Architecture Demo       ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """)
    
    # Get number of events from command line if provided
    num_events = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    delay = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    # Run the demo
    asyncio.run(process_events(num_events, delay)) 