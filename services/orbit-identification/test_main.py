import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_identify_leo_orbit():
    """Test identification of a Low Earth Orbit"""
    response = client.post(
        "/identify",
        json={
            "SEMI_MAJOR_AXIS": 7000,
            "ECCENTRICITY": 0.001,
            "INCLINATION": 98
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "TAGS" in data
    assert "LEO" in data["TAGS"]
    assert "POLAR" in data["TAGS"]
    assert "PROGRADE" in data["TAGS"]
    assert "CIRCULAR" in data["TAGS"]

def test_identify_geo_orbit():
    """Test identification of a Geostationary Orbit"""
    response = client.post(
        "/identify",
        json={
            "SEMI_MAJOR_AXIS": 42164,
            "ECCENTRICITY": 0.0001,
            "INCLINATION": 0.05
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "TAGS" in data
    assert "GEO" in data["TAGS"]
    assert "GSO" in data["TAGS"]
    assert "EQUATORIAL" in data["TAGS"] or "NEAR_EQUATORIAL" in data["TAGS"]
    assert "CIRCULAR" in data["TAGS"]

def test_identify_meo_orbit():
    """Test identification of a Medium Earth Orbit"""
    response = client.post(
        "/identify",
        json={
            "SEMI_MAJOR_AXIS": 26000,
            "ECCENTRICITY": 0.005,
            "INCLINATION": 55
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "TAGS" in data
    assert "MEO" in data["TAGS"]
    assert "PROGRADE" in data["TAGS"]
    assert "CIRCULAR" in data["TAGS"] or "NEAR_CIRCULAR" in data["TAGS"]

def test_identify_heo_orbit():
    """Test identification of a Highly Elliptical Orbit"""
    response = client.post(
        "/identify",
        json={
            "SEMI_MAJOR_AXIS": 24000,
            "ECCENTRICITY": 0.7,
            "INCLINATION": 63.4
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "TAGS" in data
    assert "HIGHLY_ELLIPTICAL_ORBIT" in data["TAGS"]
    assert "ELLIPTIC" in data["TAGS"]
    assert "PROGRADE" in data["TAGS"]

def test_identify_retrograde_orbit():
    """Test identification of a Retrograde Orbit"""
    response = client.post(
        "/identify",
        json={
            "SEMI_MAJOR_AXIS": 8000,
            "ECCENTRICITY": 0.01,
            "INCLINATION": 98.7
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "TAGS" in data
    assert "RETROGRADE" in data["TAGS"]
    assert "LEO" in data["TAGS"]
    assert "POLAR" in data["TAGS"]

def test_identify_with_mean_motion():
    """Test identification using mean motion instead of semi-major axis"""
    response = client.post(
        "/identify",
        json={
            "MEAN_MOTION": 14.8,  # Approximately 7000 km semi-major axis
            "ECCENTRICITY": 0.001,
            "INCLINATION": 98
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "TAGS" in data
    assert "LEO" in data["TAGS"]
    assert "POLAR" in data["TAGS"]

def test_identify_invalid_input():
    """Test validation error when required fields are missing"""
    response = client.post(
        "/identify",
        json={
            "INCLINATION": 98
        }
    )
    assert response.status_code == 422  # Unprocessable Entity
    data = response.json()
    assert "detail" in data

def test_identify_batch():
    """Test batch identification of multiple orbits"""
    response = client.post(
        "/identify-batch",
        json={
            "REF_FRAME": "GCRF",
            "TIME_SYSTEM": "UTC",
            "MEAN_ELEMENT_THEORY": "SGP4",
            "RECORDS": [
                {
                    "SEMI_MAJOR_AXIS": 7000,
                    "ECCENTRICITY": 0.001,
                    "INCLINATION": 98
                },
                {
                    "SEMI_MAJOR_AXIS": 42164,
                    "ECCENTRICITY": 0.0001,
                    "INCLINATION": 0.05
                }
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "RECORDS" in data
    assert len(data["RECORDS"]) == 2
    
    # Check first record (LEO)
    assert "LEO" in data["RECORDS"][0]["TAGS"]
    assert "POLAR" in data["RECORDS"][0]["TAGS"]
    
    # Check second record (GEO)
    assert "GEO" in data["RECORDS"][1]["TAGS"]
    assert "GSO" in data["RECORDS"][1]["TAGS"]

def test_protected_regions():
    """Test identification of IADC protected regions"""
    # LEO protected region
    response = client.post(
        "/identify",
        json={
            "SEMI_MAJOR_AXIS": 7200,
            "ECCENTRICITY": 0.001,
            "INCLINATION": 51.6
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "IADC_LEO_PROTECTED_REGION" in data["TAGS"]
    
    # GEO protected region
    response = client.post(
        "/identify",
        json={
            "SEMI_MAJOR_AXIS": 42164,
            "ECCENTRICITY": 0.0001,
            "INCLINATION": 0.05
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "IADC_GEO_PROTECTED_REGION" in data["TAGS"]

if __name__ == "__main__":
    pytest.main(["-v", "test_main.py"]) 