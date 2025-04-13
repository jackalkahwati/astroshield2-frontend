#!/usr/bin/env python3
"""
Simple test for the CCDM service - no dependencies on other modules
"""

import unittest
from unittest.mock import MagicMock
import datetime
import sys
import os

# Add the backend_fixed directory to the path
sys.path.insert(0, os.path.abspath("backend_fixed"))

# Import the modules directly
from backend_fixed.app.services.ccdm import CCDMService
from backend_fixed.app.models.ccdm import (
    ObjectAnalysisRequest, 
    ThreatAssessmentRequest,
    HistoricalAnalysisRequest,
    ShapeChangeRequest
)

class TestCCDMService(unittest.TestCase):
    """Test suite for CCDMService class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_db = MagicMock()
        self.ccdm_service = CCDMService(self.mock_db)
    
    def test_analyze_object(self):
        """Test analyze_object method returns expected response"""
        # Arrange
        norad_id = 25544  # ISS
        request = ObjectAnalysisRequest(
            norad_id=norad_id,
            analysis_type="FULL",
            options={"include_trajectory": True}
        )
        
        # Act
        response = self.ccdm_service.analyze_object(request)
        
        # Assert
        self.assertEqual(response.norad_id, norad_id)
        self.assertGreater(len(response.analysis_results), 0)
        self.assertIsInstance(response.timestamp, datetime.datetime)
        
    def test_assess_threat(self):
        """Test assess_threat method returns expected threat assessment"""
        # Arrange
        norad_id = 33591  # Hubble
        request = ThreatAssessmentRequest(
            norad_id=norad_id,
            assessment_factors=["COLLISION", "MANEUVER", "DEBRIS"]
        )
        
        # Act
        response = self.ccdm_service.assess_threat(request)
        
        # Assert
        self.assertEqual(response.norad_id, norad_id)
        self.assertGreater(response.confidence, 0.0)
        self.assertLessEqual(response.confidence, 1.0)
        self.assertIsInstance(response.threat_components, dict)
        self.assertGreater(len(response.recommendations), 0)
        
    def test_get_historical_analysis(self):
        """Test get_historical_analysis method returns historical data"""
        # Arrange
        norad_id = 43013  # NOAA-20
        now = datetime.datetime.utcnow()
        week_ago = now - datetime.timedelta(days=7)
        request = HistoricalAnalysisRequest(
            norad_id=norad_id,
            start_date=week_ago,
            end_date=now
        )
        
        # Act
        response = self.ccdm_service.get_historical_analysis(request)
        
        # Assert
        self.assertEqual(response.norad_id, norad_id)
        self.assertGreater(len(response.analysis_points), 0)
        self.assertGreaterEqual(response.start_date, week_ago)
        self.assertLessEqual(response.end_date, now)
        
    def test_detect_shape_changes(self):
        """Test detect_shape_changes method returns shape change data"""
        # Arrange
        norad_id = 48274  # Starlink
        now = datetime.datetime.utcnow()
        month_ago = now - datetime.timedelta(days=30)
        request = ShapeChangeRequest(
            norad_id=norad_id,
            start_date=month_ago,
            end_date=now,
            sensitivity=0.75
        )
        
        # Act
        response = self.ccdm_service.detect_shape_changes(request)
        
        # Assert
        self.assertEqual(response.norad_id, norad_id)
        self.assertIsInstance(response.detected_changes, list)
        
    def test_quick_assess_norad_id(self):
        """Test quick_assess_norad_id method returns assessment"""
        # Arrange
        norad_id = 25544  # ISS
        
        # Act
        response = self.ccdm_service.quick_assess_norad_id(norad_id)
        
        # Assert
        self.assertEqual(response.norad_id, norad_id)
        self.assertGreater(response.confidence, 0.0)
        self.assertLessEqual(response.confidence, 1.0)
        
    def test_get_last_week_analysis(self):
        """Test get_last_week_analysis method returns last week's data"""
        # Arrange
        norad_id = 27424  # XMM-Newton
        
        # Act
        response = self.ccdm_service.get_last_week_analysis(norad_id)
        
        # Assert
        self.assertEqual(response.norad_id, norad_id)
        self.assertGreater(len(response.analysis_points), 0)
        
        # Verify dates are within last week
        now = datetime.datetime.utcnow()
        week_ago = now - datetime.timedelta(days=7)
        self.assertGreaterEqual(response.start_date, week_ago)
        self.assertLessEqual(response.end_date, now)


if __name__ == "__main__":
    print("Running CCDM service tests...")
    unittest.main() 