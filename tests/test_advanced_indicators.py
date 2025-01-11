import pytest
from datetime import datetime, timedelta
from models.indicator_models import (
    SystemInteraction, EclipsePeriod, TrackingData,
    UNRegistryEntry, OrbitOccupancyData, StimulationEvent,
    LaunchTrackingData
)
from analysis.advanced_indicators import (
    StimulationEvaluator, LaunchTrackingEvaluator,
    EclipseTrackingEvaluator, OrbitOccupancyEvaluator,
    UNRegistryEvaluator
)

@pytest.fixture
def sample_stimulation_event():
    return StimulationEvent(
        event_id="TEST-001",
        timestamp=datetime.now(),
        spacecraft_id="SAT-001",
        stimulation_type="RF",
        source_system="TEST-SYS",
        response_characteristics={"power": 100, "duration": 5},
        confidence=0.95,
        evidence={"sensor_data": {"snr": 20}}
    )

@pytest.fixture
def sample_eclipse_period():
    return EclipsePeriod(
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(minutes=30),
        eclipse_type="UMBRA",
        spacecraft_id="SAT-001",
        orbit_position={"lat": 0, "lon": 0, "alt": 500}
    )

@pytest.fixture
def sample_tracking_data():
    return TrackingData(
        object_id="OBJ-001",
        timestamp=datetime.now(),
        position={"x": 100, "y": 200, "z": 300},
        velocity={"vx": 1, "vy": 2, "vz": 3},
        tracking_source="RADAR-1",
        confidence=0.9,
        uncorrelated_tracks=[]
    )

class TestStimulationEvaluator:
    def test_analyze_stimulation(self, sample_stimulation_event):
        evaluator = StimulationEvaluator()
        result = evaluator.analyze_stimulation(
            sample_stimulation_event.dict(),
            {"interactions": []}
        )
        assert len(result) > 0
        assert all(hasattr(indicator, 'confidence') for indicator in result)

    def test_invalid_stimulation(self):
        evaluator = StimulationEvaluator()
        with pytest.raises(ValueError):
            evaluator.analyze_stimulation({}, {})

class TestEclipseTrackingEvaluator:
    def test_analyze_eclipse_tracking(self, sample_eclipse_period, sample_tracking_data):
        evaluator = EclipseTrackingEvaluator()
        result = evaluator.analyze_eclipse_tracking(
            sample_tracking_data.dict(),
            {"eclipse_periods": [sample_eclipse_period.dict()]}
        )
        assert len(result) > 0
        assert all(hasattr(indicator, 'confidence') for indicator in result)

    def test_no_eclipse_data(self, sample_tracking_data):
        evaluator = EclipseTrackingEvaluator()
        with pytest.raises(ValueError):
            evaluator.analyze_eclipse_tracking(
                sample_tracking_data.dict(),
                {"eclipse_periods": []}
            )

class TestOrbitOccupancyEvaluator:
    def test_analyze_orbit_occupancy(self):
        evaluator = OrbitOccupancyEvaluator()
        data = OrbitOccupancyData(
            region_id="REG-001",
            timestamp=datetime.now(),
            total_objects=10,
            object_density=0.001,
            typical_density=0.0005,
            orbital_band={"min_alt": 500, "max_alt": 550},
            neighboring_objects=["SAT-001", "SAT-002"]
        )
        result = evaluator.analyze_orbit_occupancy(
            data.dict(),
            {"catalog": []}
        )
        assert len(result) > 0
        assert all(hasattr(indicator, 'confidence') for indicator in result)

class TestUNRegistryEvaluator:
    def test_analyze_un_registry(self):
        evaluator = UNRegistryEvaluator()
        data = UNRegistryEntry(
            international_designator="2023-001A",
            registration_date=datetime.now(),
            state_of_registry="USA",
            launch_date=datetime.now() - timedelta(days=30),
            orbital_parameters={"perigee": 500, "apogee": 550},
            function="COMMUNICATIONS",
            status="ACTIVE"
        )
        result = evaluator.analyze_un_registry(
            {"spacecraft_id": "SAT-001"},
            {"entries": [data.dict()]}
        )
        assert len(result) > 0
        assert all(hasattr(indicator, 'confidence') for indicator in result)

class TestLaunchTrackingEvaluator:
    def test_analyze_launch_tracking(self):
        evaluator = LaunchTrackingEvaluator()
        data = LaunchTrackingData(
            launch_id="LAUNCH-001",
            timestamp=datetime.now(),
            expected_objects=3,
            tracked_objects=["OBJ-001", "OBJ-002", "OBJ-003"],
            tracking_status="NORMAL",
            confidence_metrics={"position": 0.9, "velocity": 0.85},
            anomalies=[]
        )
        result = evaluator.analyze_launch_tracking(
            data.dict(),
            {"current": {}}
        )
        assert len(result) > 0
        assert all(hasattr(indicator, 'confidence') for indicator in result)

    def test_anomalous_tracking(self):
        evaluator = LaunchTrackingEvaluator()
        data = LaunchTrackingData(
            launch_id="LAUNCH-001",
            timestamp=datetime.now(),
            expected_objects=3,
            tracked_objects=["OBJ-001", "OBJ-002"],  # Missing one object
            tracking_status="ANOMALY",
            confidence_metrics={"position": 0.9, "velocity": 0.85},
            anomalies=[{"type": "MISSING_OBJECT", "description": "Expected 3, found 2"}]
        )
        result = evaluator.analyze_launch_tracking(
            data.dict(),
            {"current": {}}
        )
        assert len(result) > 0
        assert any(indicator.type == "ANOMALOUS_OBJECT_COUNT" for indicator in result)
