import pytest
from unittest.mock import MagicMock, patch
import datetime
from app.services.ccdm import CCDMService
from app.models.ccdm import (
    ObjectAnalysisRequest, 
    ObjectAnalysisResponse,
    ThreatAssessmentRequest, 
    ObjectThreatAssessment,
    HistoricalAnalysisRequest, 
    HistoricalAnalysisResponse,
    ShapeChangeRequest,
    ShapeChangeResponse,
    ThreatLevel,
    PropulsionType
)

class TestCCDMService:
    """Test suite for CCDMService class"""

    @pytest.fixture
    def mock_db_session(self):
        """Creates a mock database session for testing"""
        session = MagicMock()
        return session

    @pytest.fixture
    def ccdm_service(self, mock_db_session):
        """Creates a CCDMService instance with mocked dependencies"""
        return CCDMService(db=mock_db_session)
    
    def test_analyze_object(self, ccdm_service):
        """Test analyze_object method returns expected response"""
        # Arrange
        norad_id = 25544  # ISS
        request = ObjectAnalysisRequest(
            norad_id=norad_id,
            analysis_type="FULL",
            options={"include_trajectory": True}
        )
        
        # Act
        response = ccdm_service.analyze_object(request)
        
        # Assert
        assert isinstance(response, ObjectAnalysisResponse)
        assert response.norad_id == norad_id
        assert len(response.analysis_results) > 0
        assert isinstance(response.timestamp, datetime.datetime)
        
    def test_assess_threat(self, ccdm_service):
        """Test assess_threat method returns expected threat assessment"""
        # Arrange
        norad_id = 33591  # Hubble
        request = ThreatAssessmentRequest(
            norad_id=norad_id,
            assessment_factors=["COLLISION", "MANEUVER", "DEBRIS"]
        )
        
        # Act
        response = ccdm_service.assess_threat(request)
        
        # Assert
        assert isinstance(response, ObjectThreatAssessment)
        assert response.norad_id == norad_id
        assert isinstance(response.overall_threat, ThreatLevel)
        assert response.confidence >= 0.0 and response.confidence <= 1.0
        assert isinstance(response.threat_components, dict)
        assert len(response.recommendations) > 0
        
    def test_get_historical_analysis(self, ccdm_service):
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
        response = ccdm_service.get_historical_analysis(request)
        
        # Assert
        assert isinstance(response, HistoricalAnalysisResponse)
        assert response.norad_id == norad_id
        assert len(response.analysis_points) > 0
        assert response.start_date >= week_ago
        assert response.end_date <= now
        
    def test_detect_shape_changes(self, ccdm_service):
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
        response = ccdm_service.detect_shape_changes(request)
        
        # Assert
        assert isinstance(response, ShapeChangeResponse)
        assert response.norad_id == norad_id
        assert isinstance(response.detected_changes, list)
        
    @patch('app.services.ccdm.CCDMService._get_satellite_data')
    def test_quick_assess_norad_id(self, mock_get_satellite_data, ccdm_service):
        """Test quick_assess_norad_id method with mocked satellite data"""
        # Arrange
        norad_id = 25544
        mock_get_satellite_data.return_value = {
            "name": "ISS",
            "orbit_type": "LEO",
            "country": "International"
        }
        
        # Act
        response = ccdm_service.quick_assess_norad_id(norad_id)
        
        # Assert
        assert isinstance(response, ObjectThreatAssessment)
        assert response.norad_id == norad_id
        mock_get_satellite_data.assert_called_once_with(norad_id)
        
    def test_get_last_week_analysis(self, ccdm_service):
        """Test get_last_week_analysis method returns last week's data"""
        # Arrange
        norad_id = 27424  # XMM-Newton
        
        # Act
        response = ccdm_service.get_last_week_analysis(norad_id)
        
        # Assert
        assert isinstance(response, HistoricalAnalysisResponse)
        assert response.norad_id == norad_id
        assert len(response.analysis_points) > 0
        
        # Verify dates are within last week
        now = datetime.datetime.utcnow()
        week_ago = now - datetime.timedelta(days=7)
        assert response.start_date >= week_ago
        assert response.end_date <= now 