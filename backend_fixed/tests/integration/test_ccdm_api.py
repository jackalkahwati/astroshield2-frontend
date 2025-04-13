import pytest
from fastapi.testclient import TestClient
import datetime
import json
from app.main import app

client = TestClient(app)

class TestCCDMAPI:
    """Integration tests for CCDM API endpoints"""

    def test_analyze_endpoint(self):
        """Test the /api/v1/ccdm/analyze endpoint"""
        # Arrange
        payload = {
            "norad_id": 25544,
            "analysis_type": "FULL",
            "options": {"include_trajectory": True}
        }
        
        # Act
        response = client.post("/api/v1/ccdm/analyze", json=payload)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["norad_id"] == 25544
        assert "analysis_results" in data
        assert "timestamp" in data
        assert "summary" in data
        
    def test_threat_assessment_endpoint(self):
        """Test the /api/v1/ccdm/threat-assessment endpoint"""
        # Arrange
        payload = {
            "norad_id": 33591,
            "assessment_factors": ["COLLISION", "MANEUVER", "DEBRIS"]
        }
        
        # Act
        response = client.post("/api/v1/ccdm/threat-assessment", json=payload)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["norad_id"] == 33591
        assert "overall_threat" in data
        assert "confidence" in data
        assert "threat_components" in data
        assert "recommendations" in data
        
    def test_historical_analysis_endpoint(self):
        """Test the /api/v1/ccdm/historical endpoint"""
        # Arrange
        now = datetime.datetime.utcnow()
        week_ago = now - datetime.timedelta(days=7)
        payload = {
            "norad_id": 43013,
            "start_date": week_ago.isoformat(),
            "end_date": now.isoformat()
        }
        
        # Act
        response = client.post("/api/v1/ccdm/historical", json=payload)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["norad_id"] == 43013
        assert "analysis_points" in data
        assert "start_date" in data
        assert "end_date" in data
        assert "trend_summary" in data
        
    def test_shape_changes_endpoint(self):
        """Test the /api/v1/ccdm/shape-changes endpoint"""
        # Arrange
        now = datetime.datetime.utcnow()
        month_ago = now - datetime.timedelta(days=30)
        payload = {
            "norad_id": 48274,
            "start_date": month_ago.isoformat(),
            "end_date": now.isoformat(),
            "sensitivity": 0.75
        }
        
        # Act
        response = client.post("/api/v1/ccdm/shape-changes", json=payload)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["norad_id"] == 48274
        assert "detected_changes" in data
        assert "summary" in data
        
    def test_quick_assessment_endpoint(self):
        """Test the /api/v1/ccdm/quick-assessment/{norad_id} endpoint"""
        # Arrange
        norad_id = 25544
        
        # Act
        response = client.get(f"/api/v1/ccdm/quick-assessment/{norad_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["norad_id"] == norad_id
        assert "overall_threat" in data
        assert "confidence" in data
        assert "threat_components" in data
        assert "recommendations" in data
        
    def test_last_week_analysis_endpoint(self):
        """Test the /api/v1/ccdm/last-week-analysis/{norad_id} endpoint"""
        # Arrange
        norad_id = 27424
        
        # Act
        response = client.get(f"/api/v1/ccdm/last-week-analysis/{norad_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["norad_id"] == norad_id
        assert "analysis_points" in data
        assert "start_date" in data
        assert "end_date" in data
        assert "trend_summary" in data
        
    def test_invalid_norad_id_returns_404(self):
        """Test that invalid NORAD ID returns 404"""
        # Arrange
        invalid_norad_id = 999999  # Non-existent NORAD ID
        
        # Act
        response = client.get(f"/api/v1/ccdm/quick-assessment/{invalid_norad_id}")
        
        # Assert
        # This could be 404 or 400 depending on implementation
        assert response.status_code in [404, 400]
        
    def test_invalid_date_range_returns_400(self):
        """Test that invalid date range returns 400"""
        # Arrange
        now = datetime.datetime.utcnow()
        future = now + datetime.timedelta(days=7)
        payload = {
            "norad_id": 43013,
            "start_date": future.isoformat(),  # Start date in the future
            "end_date": now.isoformat()  # End date before start date
        }
        
        # Act
        response = client.post("/api/v1/ccdm/historical", json=payload)
        
        # Assert
        assert response.status_code == 400 