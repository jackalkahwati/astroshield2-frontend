# DnD (Dungeons and Dragons) Integration Guide for AstroShield

## Overview

AstroShield now includes comprehensive integration with the Space Security and Defense Program's (SSDP) Dungeons and Dragons (DnD) counter-CCD (Camouflage, Concealment, and Deception) capability. This integration enables AstroShield to process BOGEY objects, detect CCD tactics, and provide enhanced space domain awareness.

## What is DnD?

DnD is a prototype capability that focuses on discovering "unknown" objects such as:
- Potential clandestine payloads on rocket bodies
- Undiscovered microsats
- Signature-managed objects
- Objects using CCD tactics to avoid detection

DnD operates on the principle of "Avoid Operational Surprise" - one of the Chief of Space Operations' core tenets.

## Key Capabilities

### 1. BOGEY Object Processing
- **BOGEY Definition**: Unidentified objects without an established origin (per joint brevity definitions)
- **Threat Assessment**: Automated classification into LOW, MEDIUM, HIGH, and CRITICAL threat levels
- **Real-time Updates**: Continuous processing of DnD catalog data from UDL

### 2. CCD Tactic Detection
AstroShield can detect the following CCD tactics:
- **Payload Concealment**: Hidden payloads on legitimate objects
- **Debris Event Cover**: Using debris fields to mask activities
- **Signature Management**: Techniques to reduce detectability
- **Deceptive Maneuvers**: Unusual orbital behaviors
- **RF Masking**: Radio frequency signature manipulation
- **Eclipse Exploitation**: Activities during eclipse periods
- **Temporal Correlation**: Coordinated activities with events

### 3. Protect List Monitoring
- Monitors conjunctions between BOGEY objects and high-value assets
- Includes US and allied military/government satellites
- Automated risk assessment and alerting

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UDL (SSDP)    │    │   AstroShield   │    │  Kafka Topics   │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ DnD Catalog │ │───▶│ │ DnD Service │ │───▶│ │ SS4 CCDM    │ │
│ │   Elsets    │ │    │ │             │ │    │ │   Topics    │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ State Vecs  │ │───▶│ │ Conjunction │ │───▶│ │ UI Events   │ │
│ │ (Protect)   │ │    │ │  Analysis   │ │    │ │             │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Data Flow

### 1. DnD Catalog Ingestion
```python
# Query DnD elsets from UDL
query_params = {
    "source": "SSDP",
    "origin": "DnD",
    "epoch": f">{(datetime.utcnow() - timedelta(days=7)).isoformat()}Z",
    "maxResults": 1000
}
elsets = await udl_client.get_elsets(**query_params)
```

### 2. BOGEY Object Processing
```python
# Convert DnD elset to BOGEY object
bogey = BOGEYObject(
    dnd_id=elset['origObjectId'],
    visual_magnitude=elset['algorithm']['mv'],
    rms_accuracy=elset['algorithm']['rms'],
    threat_level=assess_threat_level(mv, rms, position),
    suspected_tactics=detect_ccd_tactics(elset, position, velocity)
)
```

### 3. Kafka Publishing
```python
# Publish BOGEY updates to standard topics
await kafka_client.publish(StandardKafkaTopics.SS4_CCDM_CCDM_DB, message)
if threat_level >= HIGH:
    await kafka_client.publish(StandardKafkaTopics.SS4_CCDM_OOI, message)
```

## API Endpoints

### BOGEY Operations

#### Get All BOGEY Objects
```http
GET /api/v1/dnd/bogeys?threat_level=HIGH&limit=100
```

#### Get Specific BOGEY
```http
GET /api/v1/dnd/bogeys/{dnd_id}
```

#### Get BOGEY Summary
```http
GET /api/v1/dnd/bogeys/summary
```

### Threat Assessment

#### Detailed Threat Assessment
```http
GET /api/v1/dnd/threat-assessment/{dnd_id}
```

Response includes:
- Threat level and confidence score
- Orbital characteristics
- Risk factors
- Recommended actions

### Operational Control

#### Manual Catalog Update
```http
POST /api/v1/dnd/catalog/update
```

#### Conjunction Check
```http
POST /api/v1/dnd/conjunctions/check
```

#### Statistics
```http
GET /api/v1/dnd/statistics
```

## Configuration

### Environment Variables
```bash
# UDL Configuration
UDL_BASE_URL=https://unifieddatalibrary.com
UDL_API_KEY=your_udl_api_key

# DnD Specific Settings
DND_CATALOG_UPDATE_INTERVAL=3600  # seconds
DND_CONJUNCTION_CHECK_INTERVAL=1800  # seconds
DND_THREAT_REASSESSMENT_INTERVAL=3600  # seconds
```

### Protect List Configuration
```python
# High-value asset patterns for conjunction monitoring
PROTECT_LIST = [
    "GPS",
    "MILSTAR", 
    "DSCS",
    "WGS",
    "SBIRS",
    "NROL",
    "USA-",
    "COSMOS",
    "GLONASS"
]
```

## Threat Assessment Logic

### Threat Levels
- **CRITICAL**: Bright objects (mv < 15) in GEO with multiple CCD tactics
- **HIGH**: Well-tracked objects (rms < 1km) in critical orbits
- **MEDIUM**: Moderately tracked objects (rms < 5km) with some CCD indicators
- **LOW**: Dim or poorly tracked objects with minimal CCD indicators

### Risk Factors
1. **Orbital Regime**: GEO objects pose higher risk to critical infrastructure
2. **Tracking Quality**: Poor tracking (high RMS) indicates potential evasion
3. **Visual Magnitude**: Dim objects may use signature management
4. **CCD Tactics**: Multiple tactics indicate sophisticated adversary

## Known Debris Events

AstroShield tracks the following major debris events from the DnD SITREP:

| Parent Object | Event Date | Total Objects | Uncatalogued |
|---------------|------------|---------------|--------------|
| INTELSAT 33E  | 2024-10-19 | 587          | 146          |
| ATLAS 5 CENTAUR | 2024-09-06 | 1327        | 279          |
| COSMOS 2473   | 2024-05-27 | 1            | 1            |

## Integration Examples

### 1. Real-time BOGEY Monitoring
```python
# Subscribe to BOGEY updates
async def handle_bogey_update(message):
    bogey_data = message.data
    if bogey_data['threatLevel'] in ['HIGH', 'CRITICAL']:
        await send_operator_alert(bogey_data)
        
kafka_client.subscribe(StandardKafkaTopics.SS4_CCDM_OOI, handle_bogey_update)
```

### 2. Conjunction Alerting
```python
# Process conjunction alerts
async def handle_conjunction_alert(message):
    alert = message.data
    if alert['riskLevel'] > 0.5:
        await escalate_to_command(alert)
        await notify_asset_operator(alert['protectObjectId'])
```

### 3. CCD Tactic Analysis
```python
# Analyze CCD tactic trends
def analyze_ccd_trends(bogey_catalog):
    tactic_counts = {}
    for bogey in bogey_catalog.values():
        for tactic in bogey.suspected_tactics:
            tactic_counts[tactic.value] = tactic_counts.get(tactic.value, 0) + 1
    
    return sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True)
```

## Operational Procedures

### 1. Daily Operations
- Review BOGEY summary dashboard
- Check high-threat objects for updates
- Monitor conjunction alerts
- Validate protect list coverage

### 2. Incident Response
- **CRITICAL Threat Detected**: Immediate escalation to command
- **High-Risk Conjunction**: Notify asset operator within 15 minutes
- **New CCD Tactic**: Update detection algorithms and brief analysts

### 3. Weekly Reviews
- Analyze CCD tactic trends
- Review debris event correlations
- Update protect list as needed
- Validate threat assessment accuracy

## Security Considerations

### 1. Data Classification
- DnD data is UNCLASSIFIED but not publicly available
- Requires UDL access permissions via SSDP
- BOGEY analysis may reveal sensitive patterns

### 2. Access Control
- API endpoints require authentication
- Role-based access to threat assessments
- Audit logging for all DnD operations

### 3. Information Sharing
- Coordinate with 18 SDS for catalog correlation
- Share threat assessments with JCO as appropriate
- Follow OPSEC guidelines for CCD discussions

## Troubleshooting

### Common Issues

#### 1. UDL Connection Failures
```bash
# Check UDL connectivity
curl -H "Authorization: Bearer $UDL_API_KEY" \
     "https://unifieddatalibrary.com/udl/elset/history?source=SSDP&origin=DnD&maxResults=1"
```

#### 2. Missing BOGEY Objects
- Verify UDL permissions for DnD data
- Check query parameters and time ranges
- Validate SGP4-XP TLE processing

#### 3. Kafka Publishing Errors
- Confirm topic access permissions
- Check message format compliance
- Verify Kafka broker connectivity

### Monitoring

#### Key Metrics
- BOGEY catalog size and growth rate
- Threat level distribution
- Conjunction alert frequency
- CCD tactic detection rates

#### Health Checks
```python
# DnD service health check
async def health_check():
    return {
        "catalog_size": len(dnd_service.catalog_processor.bogey_catalog),
        "last_update": dnd_service.last_catalog_update,
        "high_threat_count": len([b for b in catalog.values() 
                                 if b.threat_level >= BOGEYThreatLevel.HIGH]),
        "status": "OPERATIONAL"
    }
```

## Future Enhancements

### Planned Features
1. **Machine Learning Integration**: Automated CCD tactic classification
2. **Multi-Sensor Fusion**: Correlation with optical and RF signatures
3. **Predictive Analytics**: Forecast BOGEY behavior patterns
4. **Enhanced Visualization**: 3D orbital displays with threat overlays

### Research Areas
1. **Advanced Orbit Determination**: Improved SGP4-XP processing
2. **Behavioral Analysis**: Pattern recognition for CCD activities
3. **Sensor Tasking**: Automated collection requests for BOGEYs
4. **International Cooperation**: Data sharing with allied partners

## References

1. DnD Program SITREP (14 April 2025)
2. DnD UDL Data User's Guide (Version 1.1)
3. Multi-service Brevity Codes (March 2023)
4. Chief of Space Operations C-NOTE #15: "Competitive Endurance"
5. UDL API Documentation
6. AstroShield Kafka Standards Compliance Guide 