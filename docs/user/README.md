# AstroShield User Guide

Welcome to AstroShield, the advanced satellite threat assessment platform. This guide will help you navigate the system and make the most of its features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Satellite Tracking](#satellite-tracking)
4. [Threat Assessment](#threat-assessment)
5. [Historical Analysis](#historical-analysis)
6. [Shape Change Detection](#shape-change-detection)
7. [Alerts and Notifications](#alerts-and-notifications)
8. [API Integration](#api-integration)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### System Requirements

To use AstroShield, you need:

- A modern web browser (Chrome, Firefox, Safari, or Edge)
- Internet connection
- User account with appropriate permissions

### Logging In

1. Navigate to [https://asttroshield.com](https://asttroshield.com)
2. Enter your username and password
3. Click "Sign In"

## Dashboard Overview

The main dashboard provides a comprehensive view of your satellite fleet with:

- **Status Overview**: Quick view of all satellites by status
- **Recent Alerts**: Latest threat notifications
- **Activity Timeline**: Recent events and detections
- **System Health**: Critical system metrics

## Satellite Tracking

### Viewing Satellites

1. Click on "Satellites" in the main navigation
2. Browse the list of satellites or use the search function
3. Click on a satellite to view detailed information

### Satellite Details

The satellite detail page shows:

- Basic information (NORAD ID, name, status)
- Current position and trajectory
- Historical data
- Threat assessment status

## Threat Assessment

The Threat Assessment module evaluates potential risks to your satellites:

### Threat Levels

- **NONE**: No unusual activity detected
- **LOW**: Minor anomalies detected, monitoring advised
- **MEDIUM**: Significant anomalies detected, investigation recommended
- **HIGH**: Serious anomalies detected, immediate attention required
- **CRITICAL**: Severe anomalies detected, immediate action required

### Viewing Threat Assessments

1. Navigate to "CCDM" > "Threat Assessment"
2. Select a satellite from the dropdown
3. View the current threat level and supporting evidence

## Historical Analysis

The Historical Analysis tool allows you to review past threat assessments:

1. Navigate to "CCDM" > "Historical Analysis"
2. Select a satellite from the dropdown
3. Choose between the timeline chart or data table view
4. Optionally, specify a custom date range

### Understanding the Timeline Chart

- The Y-axis shows threat levels (NONE to CRITICAL)
- The X-axis represents time
- Each point indicates an assessment at that time
- Hover over points to see detailed information

## Shape Change Detection

This module detects physical changes to satellites:

1. Navigate to "CCDM" > "Shape Detection"
2. Select a satellite from the dropdown
3. View detected changes and confidence levels

### Change Types

- **EXTENSION**: The satellite has deployed components
- **CONTRACTION**: The satellite has retracted components
- **SEPARATION**: A part has detached from the satellite
- **ATTACHMENT**: An object has attached to the satellite
- **UNKNOWN**: Changes detected but type cannot be determined

## Alerts and Notifications

AstroShield provides real-time alerts for critical events:

### Alert Types

- **Threat Level Changes**: When a satellite's threat level changes
- **Shape Changes**: When physical changes are detected
- **System Events**: Important system notifications

### Configuring Alerts

1. Go to "Settings" > "Notifications"
2. Choose which alerts you want to receive
3. Configure delivery methods (email, SMS, in-app)

## API Integration

AstroShield provides a comprehensive REST API for integration with other systems:

1. Go to "Settings" > "API Access"
2. Generate an API key
3. View the API documentation at [https://asttroshield.com/api/docs](https://asttroshield.com/api/docs)

See the [API Documentation](../api/openapi.yml) for detailed information.

## Troubleshooting

### Common Issues

**Problem**: Dashboard isn't loading
**Solution**: Clear your browser cache and reload the page

**Problem**: Can't see a specific satellite
**Solution**: Check your user permissions or contact your administrator

**Problem**: Data appears outdated
**Solution**: Check your internet connection and refresh the page

### Getting Help

For additional assistance:

- Check our [Knowledge Base](https://asttroshield.com/help)
- Contact support at support@asttroshield.com
- Call our help desk at +1-555-ASTROSHIELD 