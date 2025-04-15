import aiohttp
import asyncio
import json
import sys
from datetime import datetime

async def register_event_callback(api_url, event_type, callback_url, api_key):
    """
    Register a callback URL for event notifications
    
    Args:
        api_url: Base API URL
        event_type: Type of event to subscribe to
        callback_url: URL to call when events occur
        api_key: API key for authentication
    """
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "eventType": event_type,
            "callbackUrl": callback_url,
            "format": "json"
        }
        
        response = await session.post(
            f"{api_url}/api/v1/events/subscribe",
            json=payload,
            headers=headers
        )
        
        if response.status == 201:
            result = await response.json()
            print(f"Successfully registered for {event_type} events. Subscription ID: {result['subscriptionId']}")
            return result
        else:
            error_text = await response.text()
            print(f"Failed to register: {response.status} - {error_text}")
            return None

async def list_event_subscriptions(api_url, api_key):
    """
    List all event subscriptions
    
    Args:
        api_url: Base API URL
        api_key: API key for authentication
    """
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
        
        response = await session.get(
            f"{api_url}/api/v1/events/subscriptions",
            headers=headers
        )
        
        if response.status == 200:
            subscriptions = await response.json()
            print(f"Found {len(subscriptions)} event subscriptions:")
            for sub in subscriptions:
                print(f"  - ID: {sub['subscriptionId']}, Type: {sub['eventType']}, URL: {sub['callbackUrl']}")
            return subscriptions
        else:
            error_text = await response.text()
            print(f"Failed to list subscriptions: {response.status} - {error_text}")
            return []

async def unsubscribe_from_events(api_url, subscription_id, api_key):
    """
    Unsubscribe from events
    
    Args:
        api_url: Base API URL
        subscription_id: ID of the subscription to delete
        api_key: API key for authentication
    """
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        response = await session.delete(
            f"{api_url}/api/v1/events/subscriptions/{subscription_id}",
            headers=headers
        )
        
        if response.status == 204:
            print(f"Successfully unsubscribed from events (ID: {subscription_id})")
            return True
        else:
            error_text = await response.text()
            print(f"Failed to unsubscribe: {response.status} - {error_text}")
            return False

async def handle_incoming_event(request):
    """
    Example handler for incoming event webhooks
    
    This would be implemented in your web server to receive callbacks
    """
    event_data = await request.json()
    
    # Log the event
    print(f"Received event at {datetime.utcnow().isoformat()}Z")
    print(json.dumps(event_data, indent=2))
    
    # Process based on event type
    event_type = event_data.get("header", {}).get("messageType")
    
    if event_type == "maneuver-detected":
        # Handle maneuver detection
        process_maneuver_event(event_data)
    elif event_type == "observation-window-recommended":
        # Handle observation window recommendation
        process_observation_window(event_data)
    
    # Return success response
    return {"status": "processed"}

def process_maneuver_event(event_data):
    """Process a maneuver detection event"""
    payload = event_data.get("payload", {})
    
    catalog_id = payload.get("catalogId")
    maneuver_type = payload.get("maneuverType")
    delta_v = payload.get("deltaV")
    confidence = payload.get("confidence")
    
    print(f"Processing maneuver for object {catalog_id}:")
    print(f"  - Type: {maneuver_type}")
    print(f"  - Delta-V: {delta_v} km/s")
    print(f"  - Confidence: {confidence}")
    
    # In a real application, you would:
    # 1. Update your database
    # 2. Alert operators
    # 3. Trigger follow-up observations

def process_observation_window(event_data):
    """Process an observation window recommendation event"""
    payload = event_data.get("payload", {})
    
    location = payload.get("location", {})
    quality = payload.get("qualityScore")
    recommendation = payload.get("recommendation")
    window = payload.get("observationWindow", {})
    target = payload.get("targetObject", {})
    
    print(f"Processing observation window:")
    print(f"  - Location: {location.get('latitude')}, {location.get('longitude')}")
    print(f"  - Quality: {quality} ({payload.get('qualityCategory')})")
    print(f"  - Recommendation: {recommendation}")
    print(f"  - Window: {window.get('start_time')} to {window.get('end_time')} ({window.get('duration_minutes')} min)")
    print(f"  - Target: {target.get('catalog_id')} at {target.get('altitude_km')} km")
    
    # In a real application, you would:
    # 1. Schedule observations
    # 2. Alert operators
    # 3. Update observation plans

async def main():
    """Example of event-driven integration"""
    # Configuration
    api_url = "https://api.asttroshield.com"
    api_key = "your_api_key_here"
    callback_url = "https://your-app.example.com/webhook/asttroshield"
    
    # Register for events
    maneuver_subscription = await register_event_callback(
        api_url, 
        "maneuver-detected", 
        callback_url, 
        api_key
    )
    
    observation_subscription = await register_event_callback(
        api_url,
        "observation-window-recommended",
        callback_url,
        api_key
    )
    
    # List all subscriptions
    await list_event_subscriptions(api_url, api_key)
    
    # In a real application, you would keep your server running to receive callbacks
    print("\nIn a real application, your server would now listen for incoming webhooks.")
    print("For demonstration purposes, we'll pause for a moment then unsubscribe.")
    
    await asyncio.sleep(5)
    
    # Unsubscribe (cleanup for the example)
    if maneuver_subscription:
        await unsubscribe_from_events(api_url, maneuver_subscription["subscriptionId"], api_key)
    
    if observation_subscription:
        await unsubscribe_from_events(api_url, observation_subscription["subscriptionId"], api_key)

if __name__ == "__main__":
    asyncio.run(main()) 