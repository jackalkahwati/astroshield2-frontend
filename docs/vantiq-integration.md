# Vantiq Integration Guide

This document provides information on integrating AstroShield with the Vantiq platform.

## Overview

AstroShield integrates with Vantiq to provide real-time event processing, command and control, and data visualization capabilities. The integration uses Vantiq's REST API and webhook mechanisms to exchange data between the systems.

## Setup

### Prerequisites

- Vantiq account with appropriate permissions
- API token for authentication
- Network connectivity between AstroShield and Vantiq

### Configuration

1. Copy the `.env.vantiq` file to `.env` or add its contents to your existing `.env` file:

```bash
cp .env.vantiq .env
```

2. Edit the `.env` file to set your Vantiq credentials and configuration:

```
VANTIQ_API_URL=https://your-vantiq-instance.com/api/v1
VANTIQ_API_TOKEN=your_api_token_here
VANTIQ_NAMESPACE=your_namespace
```

3. Configure the webhook endpoint in your Vantiq instance to point to your AstroShield deployment:

```
https://your-astroshield-instance.com/api/vantiq/webhook
```

## Integration Points

### Data Flow

1. **Trajectory Updates**: AstroShield sends trajectory data to Vantiq for processing and visualization.
2. **Threat Detections**: AstroShield sends threat detection events to Vantiq for alerting and response.
3. **Commands**: Vantiq sends commands to AstroShield to control satellite operations.

### API Endpoints

#### AstroShield to Vantiq

- `POST /resources/topics/TRAJECTORY_UPDATES/publish` - Send trajectory updates
- `POST /resources/topics/THREAT_DETECTIONS/publish` - Send threat detection events

#### Vantiq to AstroShield

- `POST /api/vantiq/webhook` - Receive events from Vantiq
- `GET /api/vantiq/status` - Check integration status
- `POST /api/vantiq/command` - Send commands to AstroShield

## Usage Examples

### Sending a Trajectory Update to Vantiq

```python
from backend.integrations.vantiq import VantiqAdapter

async def send_trajectory_update(satellite_id, position, velocity):
    adapter = VantiqAdapter()
    
    payload = {
        "satelliteId": satellite_id,
        "position": position,
        "velocity": velocity,
        "timestamp": int(time.time() * 1000)
    }
    
    success = await adapter.publish_event("TRAJECTORY_UPDATES", payload)
    return success
```

### Receiving a Command from Vantiq

```python
@router.post("/api/vantiq/webhook")
async def vantiq_webhook(request: Request):
    payload = await request.json()
    
    if payload.get("eventType") == "COMMAND":
        command_type = payload.get("commandType")
        params = payload.get("params", {})
        
        # Process the command
        if command_type == "MANEUVER":
            # Execute maneuver command
            pass
        elif command_type == "SCAN":
            # Execute scan command
            pass
            
    return {"status": "success"}
```

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check network connectivity
   - Verify API token is valid
   - Ensure Vantiq instance is running

2. **Authentication Errors**
   - Check credentials in `.env` file
   - Verify token has not expired
   - Ensure proper permissions are set

3. **Webhook Issues**
   - Verify webhook URL is correctly configured in Vantiq
   - Check webhook secret matches between systems
   - Ensure AstroShield webhook endpoint is publicly accessible

### Logging

Enable debug logging to troubleshoot integration issues:

```python
import logging
logging.getLogger('backend.integrations.vantiq').setLevel(logging.DEBUG)
```

## References

- [Vantiq API Documentation](https://dev.vantiq.com/docs/api)
- [AstroShield API Documentation](https://astroshield.com/docs/api) 