"""Test suite for memory optimization verification."""

import pytest
import psutil
import time
import numpy as np
from datetime import datetime, timedelta
import gc
import asyncio
from typing import Dict, List
from collections import deque

from models.monitoring.real_time_monitor import RealTimeMonitor, SystemMetrics
from models.trajectory_predictor import TrajectoryPredictor
from feedback.telemetry_processor import TelemetryProcessor, RawTelemetry
from api.trajectory_api import generate_trajectory_points, process_trajectory_batch

def get_process_memory() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

@pytest.fixture
def real_time_monitor():
    """Create a RealTimeMonitor instance for testing."""
    monitor = RealTimeMonitor(max_metrics=1000)
    yield monitor
    monitor.stop()

@pytest.fixture
def telemetry_processor():
    """Create a TelemetryProcessor instance for testing."""
    processor = TelemetryProcessor(batch_size=100)
    yield processor
    processor.cleanup()

@pytest.fixture
def trajectory_predictor():
    """Create a TrajectoryPredictor instance for testing."""
    config = {
        'atmospheric_model': 'exponential',
        'wind_model': 'constant',
        'monte_carlo_samples': 100
    }
    return TrajectoryPredictor(config)

def generate_test_metrics(count):
    """Generate test metrics data."""
    base_time = datetime.now()
    for i in range(count):
        yield {
            'timestamp': (base_time + timedelta(seconds=i)).isoformat(),
            'cpu_usage': np.random.uniform(0, 100),
            'memory_usage': np.random.uniform(0, 1000),
            'network_io': np.random.uniform(0, 500)
        }

def generate_test_telemetry(count):
    """Generate test telemetry data."""
    base_time = datetime.now()
    for i in range(count):
        yield {
            'timestamp': (base_time + timedelta(seconds=i)).isoformat(),
            'satellite_id': f'SAT-{i % 10}',
            'position': {
                'x': np.random.uniform(-1000, 1000),
                'y': np.random.uniform(-1000, 1000),
                'z': np.random.uniform(-1000, 1000)
            },
            'velocity': {
                'vx': np.random.uniform(-10, 10),
                'vy': np.random.uniform(-10, 10),
                'vz': np.random.uniform(-10, 10)
            }
        }

def test_real_time_monitor_memory(real_time_monitor):
    """Test memory usage of RealTimeMonitor during metric collection."""
    initial_memory = get_process_memory()
    
    # Add metrics in batches
    for batch in range(5):
        metrics = list(generate_test_metrics(200))
        for metric in metrics:
            real_time_monitor.add_metric(metric)
        time.sleep(0.1)  # Allow time for processing
    
    final_memory = get_process_memory()
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be bounded due to fixed-size deque
    assert memory_increase < 50, f"Memory increase ({memory_increase:.2f}MB) exceeds threshold"
    assert len(real_time_monitor.current_metrics) <= 1000, "Metrics queue exceeded max size"

def test_trajectory_analysis_memory():
    """Test memory usage of trajectory analysis functions."""
    initial_memory = get_process_memory()
    
    # Generate and process trajectory points in batches
    total_points = 10000
    batch_size = 100
    breakup_events = []
    
    generator = generate_trajectory_points(
        start_position=(0, 0, 0),
        start_velocity=(1, 1, 1),
        time_steps=total_points,
        dt=0.1
    )
    
    current_batch = []
    for point in generator:
        current_batch.append(point)
        if len(current_batch) >= batch_size:
            events = process_trajectory_batch(current_batch)
            if events:
                breakup_events.extend(events)
            current_batch = []
    
    final_memory = get_process_memory()
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be bounded due to batch processing
    assert memory_increase < 50, f"Memory increase ({memory_increase:.2f}MB) exceeds threshold"

def test_telemetry_processor_memory(telemetry_processor):
    """Test memory usage of TelemetryProcessor during telemetry processing."""
    initial_memory = get_process_memory()
    
    # Process telemetry data in batches
    for batch in range(5):
        telemetry_data = list(generate_test_telemetry(200))
        for data in telemetry_data:
            telemetry_processor.add_telemetry(data)
        time.sleep(0.1)  # Allow time for batch processing
    
    final_memory = get_process_memory()
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be bounded due to batch processing and weak references
    assert memory_increase < 50, f"Memory increase ({memory_increase:.2f}MB) exceeds threshold"

def test_combined_system_memory(real_time_monitor, telemetry_processor):
    """Test memory usage when multiple components are working together."""
    initial_memory = get_process_memory()
    
    # Run all components simultaneously
    for _ in range(3):
        # Add metrics
        metrics = list(generate_test_metrics(100))
        for metric in metrics:
            real_time_monitor.add_metric(metric)
        
        # Process telemetry
        telemetry_data = list(generate_test_telemetry(100))
        for data in telemetry_data:
            telemetry_processor.add_telemetry(data)
        
        # Process trajectories
        generator = generate_trajectory_points(
            start_position=(0, 0, 0),
            start_velocity=(1, 1, 1),
            time_steps=1000,
            dt=0.1
        )
        batch = []
        for point in generator:
            batch.append(point)
            if len(batch) >= 100:
                process_trajectory_batch(batch)
                batch = []
        
        time.sleep(0.2)  # Allow time for processing
    
    final_memory = get_process_memory()
    memory_increase = final_memory - initial_memory
    
    # Combined memory increase should still be bounded
    assert memory_increase < 150, f"Combined memory increase ({memory_increase:.2f}MB) exceeds threshold"

def test_memory_cleanup(real_time_monitor, telemetry_processor):
    """Test proper memory cleanup after component shutdown."""
    # Generate some load
    metrics = list(generate_test_metrics(500))
    telemetry_data = list(generate_test_telemetry(500))
    
    for metric in metrics:
        real_time_monitor.add_metric(metric)
    
    for data in telemetry_data:
        telemetry_processor.add_telemetry(data)
    
    time.sleep(0.5)  # Allow time for processing
    
    initial_memory = get_process_memory()
    
    # Cleanup
    real_time_monitor.stop()
    telemetry_processor.cleanup()
    
    time.sleep(0.5)  # Allow time for cleanup
    
    final_memory = get_process_memory()
    memory_difference = final_memory - initial_memory
    
    # Memory should decrease or stay roughly the same after cleanup
    assert memory_difference < 10, f"Memory not properly cleaned up (difference: {memory_difference:.2f}MB)"

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 