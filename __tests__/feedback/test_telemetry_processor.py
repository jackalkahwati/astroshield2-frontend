import pytest
import asyncio
import numpy as np
from datetime import datetime
from feedback.telemetry_processor import TelemetryProcessor, RawTelemetry, ProcessedTelemetry

@pytest.fixture
def telemetry_processor():
    return TelemetryProcessor()

@pytest.mark.asyncio
async def test_process_telemetry_data(telemetry_processor):
    # Create mock raw telemetry data
    raw_telemetry = RawTelemetry(
        spacecraft_id='test-spacecraft',
        timestamp=datetime.fromisoformat('2024-01-01T00:00:00'),
        sensor_data={
            'position': [1000, 2000, 3000],
            'velocity': [1, 2, 3],
            'attitude': [0.1, 0.2, 0.3]
        },
        system_status={
            'power': 'NOMINAL',
            'fuel': 85.5,
            'communications': 'OPTIMAL'
        },
        environmental_data={
            'radiation': 0.15,
            'temperature': 293,
            'magnetic_field': [0.1, -0.2, 0.3]
        }
    )

    # Process the telemetry data
    processed = await telemetry_processor.process_telemetry(raw_telemetry)

    # Verify the processed data
    assert processed.spacecraft_id == raw_telemetry.spacecraft_id
    assert isinstance(processed.sensor_readings, dict)
    assert isinstance(processed.resource_status, dict)
    assert isinstance(processed.threat_indicators, dict)
    assert isinstance(processed.quality_metrics, dict)

    # Verify sensor readings
    assert len(processed.sensor_readings) > 0
    for value in processed.sensor_readings.values():
        assert isinstance(value, float)

    # Verify resource status
    assert 'power' in processed.resource_status
    assert processed.resource_status['power'] == 0.8  # NOMINAL status
    assert processed.resource_status['fuel'] == 85.5
    assert processed.resource_status['communications'] == 1.0  # OPTIMAL status

    # Verify threat indicators
    assert len(processed.threat_indicators) > 0
    for value in processed.threat_indicators.values():
        assert 0 <= value <= 1

    # Verify quality metrics
    assert 'completeness' in processed.quality_metrics
    assert 'reliability' in processed.quality_metrics
    assert 'accuracy' in processed.quality_metrics
    for metric in processed.quality_metrics.values():
        assert 0 <= metric <= 1

@pytest.mark.asyncio
async def test_handle_missing_data(telemetry_processor):
    # Create mock raw telemetry with missing data
    raw_telemetry = RawTelemetry(
        spacecraft_id='test-spacecraft',
        timestamp=datetime.fromisoformat('2024-01-01T00:00:00'),
        sensor_data={
            'position': None,
            'velocity': [1, 2, 3]
        },
        system_status={
            'power': 'UNKNOWN'
        },
        environmental_data={}
    )

    # Process the telemetry data
    processed = await telemetry_processor.process_telemetry(raw_telemetry)

    # Verify handling of missing data
    assert processed.spacecraft_id == raw_telemetry.spacecraft_id
    assert 'position' not in processed.sensor_readings
    assert 'velocity' in processed.sensor_readings
    assert processed.quality_metrics['completeness'] < 1.0

@pytest.mark.asyncio
async def test_batch_processing(telemetry_processor):
    # Create multiple telemetry entries
    raw_telemetry_1 = RawTelemetry(
        spacecraft_id='test-spacecraft',
        timestamp=datetime.fromisoformat('2024-01-01T00:00:00'),
        sensor_data={'temperature': 293},
        system_status={'power': 'NOMINAL'},
        environmental_data={'radiation': 0.15}
    )

    raw_telemetry_2 = RawTelemetry(
        spacecraft_id='test-spacecraft',
        timestamp=datetime.fromisoformat('2024-01-01T00:00:01'),
        sensor_data={'temperature': 294},
        system_status={'power': 'NOMINAL'},
        environmental_data={'radiation': 0.16}
    )

    # Process telemetry entries
    await telemetry_processor.process_telemetry(raw_telemetry_1)
    await telemetry_processor.process_telemetry(raw_telemetry_2)

    # Process the batch
    batch_stats = await telemetry_processor._process_telemetry_batch()

    # Verify batch statistics
    assert batch_stats is not None
    assert 'sensor_readings' in batch_stats
    assert 'resource_status' in batch_stats
    assert 'threat_indicators' in batch_stats

    # Verify statistics calculations
    for category in batch_stats:
        for key in batch_stats[category]:
            stats = batch_stats[category][key]
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats

@pytest.mark.asyncio
async def test_environmental_data_processing(telemetry_processor):
    # Create raw telemetry with environmental data
    raw_telemetry = RawTelemetry(
        spacecraft_id='test-spacecraft',
        timestamp=datetime.fromisoformat('2024-01-01T00:00:00'),
        sensor_data={},
        system_status={},
        environmental_data={
            'radiation': [0.1, 0.15, 0.12],
            'temperature': 293,
            'magnetic_field': [0.1, -0.2, 0.3]
        }
    )

    # Process the telemetry data
    processed = await telemetry_processor.process_telemetry(raw_telemetry)

    # Verify threat indicators
    assert 'radiation_volatility' in processed.threat_indicators
    assert 'temperature' in processed.threat_indicators
    assert 'magnetic_field_volatility' in processed.threat_indicators

    # Verify all threat indicators are normalized
    for value in processed.threat_indicators.values():
        assert 0 <= value <= 1

@pytest.mark.asyncio
async def test_quality_metrics_calculation(telemetry_processor):
    # Create raw telemetry with various quality levels
    raw_telemetry = RawTelemetry(
        spacecraft_id='test-spacecraft',
        timestamp=datetime.fromisoformat('2024-01-01T00:00:00'),
        sensor_data={
            'temperature': 293,
            'pressure': 1.0,
            'humidity': 0.5
        },
        system_status={
            'power': 'NOMINAL',
            'fuel': 85.5
        },
        environmental_data={
            'radiation': 0.15,
            'magnetic_field': [0.1, -0.2, 0.3]
        }
    )

    # Process the telemetry data
    processed = await telemetry_processor.process_telemetry(raw_telemetry)

    # Verify quality metrics
    assert 'completeness' in processed.quality_metrics
    assert 'reliability' in processed.quality_metrics
    assert 'accuracy' in processed.quality_metrics

    # Verify metric ranges
    for metric in processed.quality_metrics.values():
        assert 0 <= metric <= 1

    # Calculate expected completeness
    raw_field_count = (
        len(raw_telemetry.sensor_data) +
        len(raw_telemetry.system_status) +
        len(raw_telemetry.environmental_data)
    )
    processed_field_count = (
        len(processed.sensor_readings) +
        len(processed.resource_status) +
        len(processed.threat_indicators)
    )
    expected_completeness = processed_field_count / raw_field_count

    # Verify completeness calculation
    assert processed.quality_metrics['completeness'] == expected_completeness
