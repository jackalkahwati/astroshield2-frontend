"""Script to run memory optimization tests and generate reports."""

import os
import sys
import pytest
import json
import psutil
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from models.monitoring.memory_monitor import MemoryMonitor

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class MemoryTracker:
    def __init__(self):
        self.memory_samples = []
        self.test_peaks = {}
        self.current_test = None
        self.monitor = MemoryMonitor(
            max_samples=10000,
            gc_threshold=90.0,
            sample_interval=0.1,
            log_dir="reports/memory/monitor"
        )
        
    def start(self):
        """Start memory tracking."""
        self.monitor.start()
        
    def stop(self):
        """Stop memory tracking."""
        self.monitor.stop()
        
    def sample_memory(self):
        """Sample current memory usage."""
        memory_info = self.monitor._get_memory_info()
        self.memory_samples.append(memory_info['rss'])
        if self.current_test:
            self.test_peaks[self.current_test] = max(
                self.test_peaks.get(self.current_test, 0),
                memory_info['rss']
            )
        return memory_info['rss']

    def set_current_test(self, test_name):
        """Set the current test being monitored."""
        self.current_test = test_name

class MemoryPlugin:
    def __init__(self):
        self.memory_tracker = MemoryTracker()

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item):
        self.memory_tracker.set_current_test(item.name)
        return None

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_teardown(self):
        self.memory_tracker.set_current_test(None)
        
    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionstart(self, session):
        self.memory_tracker.start()
        
    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        self.memory_tracker.stop()

def create_memory_report(memory_tracker):
    """Generate comprehensive memory usage report."""
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path("reports/memory")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Get monitored stats
    monitor_stats = memory_tracker.monitor.get_memory_stats()
    
    # Calculate statistics
    memory_samples = np.array(memory_tracker.memory_samples)
    initial_memory = memory_samples[0] if len(memory_samples) > 0 else 0
    final_memory = memory_samples[-1] if len(memory_samples) > 0 else 0
    peak_memory = np.max(memory_samples) if len(memory_samples) > 0 else 0
    avg_memory = np.mean(memory_samples) if len(memory_samples) > 0 else 0
    
    # Create memory usage graph
    plt.figure(figsize=(12, 8))
    
    # Plot overall memory usage
    plt.subplot(2, 1, 1)
    plt.plot(memory_samples, label='Memory Usage')
    plt.title('Overall Memory Usage')
    plt.xlabel('Sample')
    plt.ylabel('Memory Usage (MB)')
    plt.grid(True)
    plt.legend()
    
    # Plot per-test peaks
    plt.subplot(2, 1, 2)
    tests = list(memory_tracker.test_peaks.keys())
    peaks = [memory_tracker.test_peaks[test] for test in tests]
    plt.bar(tests, peaks)
    plt.title('Peak Memory Usage by Test')
    plt.xlabel('Test Name')
    plt.ylabel('Peak Memory (MB)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(reports_dir / f'memory_graph_{report_time}.png')
    plt.close()
    
    # Generate report
    report = {
        'timestamp': report_time,
        'statistics': {
            'initial_memory_mb': round(initial_memory, 2),
            'final_memory_mb': round(final_memory, 2),
            'peak_memory_mb': round(peak_memory, 2),
            'average_memory_mb': round(avg_memory, 2),
            'gc_collections': monitor_stats['current']['gc_count'] if monitor_stats['current'] else None
        },
        'test_peaks': {
            test: round(peak, 2)
            for test, peak in memory_tracker.test_peaks.items()
        },
        'monitor_stats': {
            'peak': monitor_stats['peak'],
            'average': round(monitor_stats['average'], 2) if monitor_stats['average'] else None
        }
    }
    
    with open(reports_dir / f'memory_report_{report_time}.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nMemory Test Summary:")
    print("=" * 50)
    print(f"Initial Memory: {round(initial_memory, 2)} MB")
    print(f"Final Memory: {round(final_memory, 2)} MB")
    print(f"Peak Memory: {round(peak_memory, 2)} MB")
    print(f"Average Memory: {round(avg_memory, 2)} MB")
    print("\nPer-Test Memory Peaks:")
    for test, peak in memory_tracker.test_peaks.items():
        print(f"{test}: {round(peak, 2)} MB")
    print(f"\nTest reports saved to {reports_dir.absolute()}\n")

def main():
    """Run memory tests and generate reports."""
    # Configure pytest arguments
    test_dir = Path(__file__).parent
    pytest_args = [
        str(test_dir / 'test_memory_optimizations.py'),
        str(test_dir / 'test_memory_monitor.py'),
        '-v',
        '--no-header',
        '--no-summary',
        '--disable-warnings'
    ]
    
    # Create and register the memory plugin
    memory_plugin = MemoryPlugin()
    
    # Run tests with memory tracking
    exit_code = pytest.main(pytest_args, plugins=[memory_plugin])
    
    # Generate memory report
    create_memory_report(memory_plugin.memory_tracker)
    
    return exit_code

if __name__ == '__main__':
    sys.exit(main()) 