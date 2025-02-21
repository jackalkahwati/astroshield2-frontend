"""Test suite for memory monitoring functionality."""

import pytest
import time
from pathlib import Path
import json
import psutil
from datetime import datetime, timedelta
from collections import deque

from src.models.monitoring.memory_monitor import MemoryMonitor

@pytest.fixture
def memory_monitor():
    """Create a memory monitor instance for testing."""
    # Ensure test log directory exists and is empty
    log_dir = Path("test_logs/memory")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    monitor = MemoryMonitor(
        max_samples=100,
        gc_threshold=90.0,
        sample_interval=0.1,
        log_dir="test_logs/memory"
    )
    yield monitor
    monitor.stop()
    # Cleanup test logs
    if Path("test_logs").exists():
        for file in Path("test_logs/memory").glob("*"):
            file.unlink()
        Path("test_logs/memory").rmdir()
        Path("test_logs").rmdir()

def test_monitor_initialization(memory_monitor):
    """Test memory monitor initialization."""
    assert memory_monitor.max_samples == 100
    assert memory_monitor.gc_threshold == 90.0
    assert memory_monitor.sample_interval == 0.1
    assert isinstance(memory_monitor.memory_samples, deque)
    assert memory_monitor.memory_samples.maxlen == 100
    assert memory_monitor.peak_memory == 0.0

def test_memory_monitoring(memory_monitor):
    """Test basic memory monitoring functionality."""
    memory_monitor.start()
    
    # Allocate some memory to test monitoring
    data = []
    for _ in range(1000):
        data.append("x" * 1000)
    
    # Let the monitor collect some samples
    time.sleep(0.5)
    
    # Get memory stats
    stats = memory_monitor.get_memory_stats()
    assert stats['current'] is not None
    assert stats['peak']['value'] > 0
    assert stats['peak']['unit'] == 'MB'
    assert stats['average'] > 0
    
    # Clear data to free memory
    data.clear()

def test_memory_logging(memory_monitor):
    """Test memory statistics logging."""
    memory_monitor.start()
    time.sleep(0.3)  # Allow time for some samples
    memory_monitor.stop()
    
    # Check log files
    log_dir = Path("test_logs/memory")
    assert (log_dir / "memory.log").exists()
    assert (log_dir / "memory_stats.jsonl").exists()
    assert (log_dir / "memory_summary.json").exists()
    
    # Verify summary file content
    with open(log_dir / "memory_summary.json") as f:
        summary = json.load(f)
        assert 'peak_memory_mb' in summary
        assert 'samples_count' in summary
        assert 'final_memory_mb' in summary
        assert 'average_memory_mb' in summary

def test_garbage_collection_trigger(memory_monitor):
    """Test garbage collection triggering."""
    # Lower the GC threshold temporarily for testing
    memory_monitor.gc_threshold = 1.0  # This will trigger GC more easily
    memory_monitor.start()
    
    # Allocate and release memory to trigger GC
    data = []
    for _ in range(100000):  # Increase allocation size
        data.append("x" * 10000)  # Increase string size
    
    # Force memory pressure
    more_data = ["y" * 10000 for _ in range(50000)]
    data.extend(more_data)
    
    time.sleep(0.5)  # Allow more time for GC to trigger
    data.clear()
    more_data.clear()
    
    # Check if GC was logged
    log_file = Path("test_logs/memory/memory.log")
    assert log_file.exists()
    
    # Wait a bit to ensure log is written
    time.sleep(0.2)
    
    with open(log_file) as f:
        log_content = f.read()
        assert "Garbage collection triggered" in log_content, "GC was not triggered as expected"

def test_memory_stats_accuracy(memory_monitor):
    """Test accuracy of memory statistics."""
    memory_monitor.start()
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)
    
    # Allocate known amount of memory
    data = ["x" * 1000000 for _ in range(10)]  # Allocate ~10MB
    time.sleep(0.3)  # Allow time for monitoring
    
    # Get stats
    stats = memory_monitor.get_memory_stats()
    current_memory = stats['current']['rss']
    
    # Memory usage should have increased
    assert current_memory > initial_memory
    
    # Clear data
    data.clear()
    time.sleep(0.3)  # Allow time for memory to be freed
    
    # Get final stats
    final_stats = memory_monitor.get_memory_stats()
    assert final_stats['peak']['value'] >= current_memory

def test_monitor_cleanup(memory_monitor):
    """Test proper cleanup of monitor resources."""
    memory_monitor.start()
    time.sleep(0.2)
    
    # Stop monitoring
    memory_monitor.stop()
    assert not memory_monitor._running
    assert not hasattr(memory_monitor, 'monitor_thread') or not memory_monitor.monitor_thread.is_alive()
    
    # Check if files are properly saved
    summary_file = Path("test_logs/memory/memory_summary.json")
    assert summary_file.exists()
    
    # Verify we can still get stats after stopping
    stats = memory_monitor.get_memory_stats()
    assert stats['current'] is not None
    assert stats['peak'] is not None
    assert stats['average'] is not None 