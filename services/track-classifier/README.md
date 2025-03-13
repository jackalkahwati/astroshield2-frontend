# Track Classification API

This microservice provides a mechanism for classifying tracks to determine the type of space object based on electro-optical (EO) observations.

## Features

- Single track classification
- Batch track classification
- OpenAPI 3.1 compliant API

## API Endpoints

### Electro-Optical Object Type Classification

#### Single Track Classification
- **Endpoint**: `POST /classify-eo-track/object-type`
- **Description**: Classifies a single EO track to determine if it's a payload, rocket body, or debris
- **Input**: JSON object containing EO observation records
- **Output**: Classification probabilities for different object types

#### Batch Track Classification
- **Endpoint**: `POST /classify-eo-track/object-type/batch`
- **Description**: Classifies multiple EO tracks in a single request
- **Input**: Array of JSON objects containing EO observation records
- **Output**: Collection of classification results

## Data Models

### Input Data

An Electro-Optical Observation (EOO) contains:
- `OB_TIME`: Observation time (ISO format)
- `RA`: Right Ascension (degrees)
- `DEC`: Declination (degrees)
- `MAG`: Magnitude
- `SOLAR_PHASE_ANGLE`: Solar Phase Angle (degrees)
- `SOURCE`: Source of the observation (string)

### Output Data

Classification results include probabilities for:
- `PAYLOAD`: Probability of being a payload (0-1)
- `ROCKET_BODY`: Probability of being a rocket body (0-1)
- `DEBRIS`: Probability of being debris (0-1)

## Examples

### Single Track Classification

#### Request:

```json
{
  "RECORDS": [
    {
      "OB_TIME": "2021-01-01T00:00:00.000Z",
      "RA": 0,
      "DEC": 0,
      "MAG": 5.5,
      "SOLAR_PHASE_ANGLE": 0,
      "SOURCE": "PPEC"
    }
  ]
}
```

#### Response:

```json
{
  "PAYLOAD": 0.32,
  "ROCKET_BODY": 0.58,
  "DEBRIS": 0.10
}
```

### Batch Track Classification

#### Request:

```json
[
  {
    "RECORDS": [
      {
        "OB_TIME": "2021-01-01T00:00:00.000Z",
        "RA": 0,
        "DEC": 0,
        "MAG": 5.5,
        "SOLAR_PHASE_ANGLE": 0,
        "SOURCE": "PPEC"
      }
    ]
  },
  {
    "RECORDS": [
      {
        "OB_TIME": "2021-01-01T01:00:00.000Z",
        "RA": 45,
        "DEC": 30,
        "MAG": 3.0,
        "SOLAR_PHASE_ANGLE": 20,
        "SOURCE": "PPEC"
      }
    ]
  }
]
```

#### Response:

```json
{
  "RECORDS": [
    {
      "PAYLOAD": 0.32,
      "ROCKET_BODY": 0.58,
      "DEBRIS": 0.10
    },
    {
      "PAYLOAD": 0.72,
      "ROCKET_BODY": 0.18,
      "DEBRIS": 0.10
    }
  ]
}
```

## Setup and Deployment

### Requirements

- Python 3.7+
- FastAPI
- Uvicorn

### Installation

1. Install required packages:
   ```
   pip install fastapi uvicorn
   ```

2. Run the service:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### API Documentation

When the service is running, you can access:
- Swagger UI: `/services/track-classifier/docs`
- ReDoc: `/services/track-classifier/redoc`
- OpenAPI Schema: `/services/track-classifier/openapi.json`

## Implementation Notes

The current implementation uses a simple rule-based approach for demonstration purposes:
- Brighter objects (low magnitude) are classified as likely payloads
- Medium brightness objects tend to be classified as rocket bodies
- Dimmer objects (high magnitude) are more likely to be debris

In a production environment, this would be replaced with a trained machine learning model that considers all observation parameters and temporal patterns. 