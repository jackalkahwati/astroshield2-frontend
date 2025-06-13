# Mapbox Integration Setup

## Overview

AstroShield now includes a real Mapbox-based orbital tracking system that provides:

- **Real Earth Map**: Satellite imagery with geographic context
- **Interactive Satellite Tracking**: Click satellites to view details and fly to location
- **Ground Station Network**: Military, DSN, and commercial ground stations
- **Orbital Ground Tracks**: Visual satellite paths across Earth
- **Coverage Areas**: Satellite communication and sensor coverage zones
- **Real-time Updates**: Live position updates and threat status

## Setup Instructions

### 1. Get a Mapbox Access Token

1. Go to [mapbox.com](https://mapbox.com) and create an account
2. Navigate to your Account Dashboard
3. Click "Create a token" 
4. Copy your access token

### 2. Configure Environment Variable

Create a `.env.local` file in the frontend directory:

```bash
# astroshield-production/frontend/.env.local
NEXT_PUBLIC_MAPBOX_TOKEN=your_mapbox_token_here
```

### 3. Features Available

#### Interactive Map
- **Globe Projection**: 3D globe view for space context
- **Satellite Imagery**: Real Earth surface from space
- **Navigation Controls**: Zoom, pan, rotate controls
- **Fly-to Animation**: Smooth camera movements to objects

#### Satellite Tracking
- **Real-time Positions**: Live satellite coordinates over Earth
- **Threat Level Indicators**: Color-coded threat status
- **Interactive Markers**: Click for detailed information
- **Ground Track Visualization**: Orbital paths over surface

#### Ground Stations
- **DSN Network**: Deep Space Network stations (blue)
- **Military Stations**: Space Force facilities (red) 
- **Commercial Stations**: Private sector stations (green)
- **Status Monitoring**: Active/offline status tracking

#### Space Operations Features
- **Multi-object Tracking**: Monitor multiple satellites simultaneously
- **Coverage Analysis**: Sensor and communication coverage areas
- **Conjunction Monitoring**: Close approach event tracking
- **Threat Assessment**: Real-time risk evaluation

## Development Notes

The component (`OrbitalMapView`) includes:
- SSR-safe implementation with dynamic imports
- Responsive design for different screen sizes
- Dark theme integration matching AstroShield UI
- Professional space operations styling
- Real-time data update capabilities

## Fallback Behavior

If no Mapbox token is provided, the component will:
- Display a loading state with clear messaging
- Log error information to console
- Gracefully degrade to show tracking data without map

This provides a much more valuable orbital situational awareness tool for space operators compared to the previous simple animation. 