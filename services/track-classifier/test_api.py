import requests
import json
from datetime import datetime, timezone

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_single_track_classification():
    """Test the single track classification endpoint."""
    endpoint = f"{BASE_URL}/classify-eo-track/object-type"
    
    # Test data
    data = {
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
    
    # Send request
    response = requests.post(endpoint, json=data)
    
    # Check response
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    
    result = response.json()
    print("Single track classification result:")
    print(json.dumps(result, indent=2))
    
    # Verify structure
    assert "PAYLOAD" in result, "PAYLOAD field missing from response"
    assert "ROCKET_BODY" in result, "ROCKET_BODY field missing from response"
    assert "DEBRIS" in result, "DEBRIS field missing from response"
    
    # Verify probabilities sum to approximately 1
    total_prob = result["PAYLOAD"] + result["ROCKET_BODY"] + result["DEBRIS"]
    assert 0.99 <= total_prob <= 1.01, f"Probabilities should sum to 1, got {total_prob}"
    
    print("Single track classification test passed!")

def test_batch_track_classification():
    """Test the batch track classification endpoint."""
    endpoint = f"{BASE_URL}/classify-eo-track/object-type/batch"
    
    # Test data
    data = [
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
    
    # Send request
    response = requests.post(endpoint, json=data)
    
    # Check response
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    
    result = response.json()
    print("\nBatch track classification result:")
    print(json.dumps(result, indent=2))
    
    # Verify structure
    assert "RECORDS" in result, "RECORDS field missing from response"
    assert len(result["RECORDS"]) == 2, f"Expected 2 records, got {len(result['RECORDS'])}"
    
    # Check each record
    for i, record in enumerate(result["RECORDS"]):
        assert "PAYLOAD" in record, f"PAYLOAD field missing from record {i}"
        assert "ROCKET_BODY" in record, f"ROCKET_BODY field missing from record {i}"
        assert "DEBRIS" in record, f"DEBRIS field missing from record {i}"
        
        # Verify probabilities sum to approximately 1
        total_prob = record["PAYLOAD"] + record["ROCKET_BODY"] + record["DEBRIS"]
        assert 0.99 <= total_prob <= 1.01, f"Probabilities for record {i} should sum to 1, got {total_prob}"
    
    print("Batch track classification test passed!")

if __name__ == "__main__":
    print("Running Track Classification API tests...")
    try:
        test_single_track_classification()
        test_batch_track_classification()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise 