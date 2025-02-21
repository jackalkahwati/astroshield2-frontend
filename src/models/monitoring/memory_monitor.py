"""Memory monitoring system for tracking and optimizing memory usage."""

import psutil
import gc
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging
from collections import deque
from threading import Thread

class MemoryMonitor:
    """Specialized monitor for tracking memory usage and triggering optimizations."""
    
    def __init__(self, 
                 max_samples: int = 3600,
                 gc_threshold: float = 80.0,  # percent
                 sample_interval: float = 1.0,  # seconds
                 log_dir: str = "logs/memory"):
        self.max_samples = max_samples
        self.gc_threshold = gc_threshold
        self.sample_interval = sample_interval
        self.memory_samples = deque(maxlen=max_samples)
        self._running = False
        self.peak_memory = 0.0
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger("memory_monitor")
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add file handler
        file_handler = logging.FileHandler(self.log_dir / "memory.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
        
        # Create empty stats files
        (self.log_dir / "memory_stats.jsonl").touch()
        (self.log_dir / "memory_summary.json").touch()
    
    def start(self):
        """Start memory monitoring."""
        self._running = True
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Memory monitoring started")
    
    def stop(self):
        """Stop memory monitoring."""
        self._running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        self._save_memory_stats()
        self.logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                memory_info = self._get_memory_info()
                self._process_memory_info(memory_info)
                time.sleep(self.sample_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'rss': memory_info.rss / (1024 * 1024),  # MB
            'vms': memory_info.vms / (1024 * 1024),  # MB
            'percent': process.memory_percent(memtype='rss'),  # Use RSS for more accurate percentage
            'gc_count': gc.get_count()
        }
    
    def _process_memory_info(self, memory_info: Dict[str, Any]):
        """Process memory information and take action if needed."""
        self.memory_samples.append(memory_info)
        self.peak_memory = max(self.peak_memory, memory_info['rss'])
        
        # Check if we need to trigger garbage collection
        if memory_info['percent'] > self.gc_threshold:
            self._trigger_gc()
        
        # Log memory stats
        self._log_memory_stats(memory_info)
    
    def _trigger_gc(self):
        """Trigger garbage collection if memory usage is high."""
        before_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        gc.collect()  # Run a full collection
        gc.collect()  # Run a second pass to ensure thorough collection
        after_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        
        freed_mem = before_mem - after_mem
        self.logger.info(
            f"Garbage collection triggered. "
            f"Freed memory: {freed_mem:.2f}MB"
        )
    
    def _log_memory_stats(self, memory_info: Dict[str, Any]):
        """Log memory statistics."""
        # Write to JSONL file
        with open(self.log_dir / "memory_stats.jsonl", "a") as f:
            json.dump(memory_info, f)
            f.write('\n')
        
        # Log significant changes or high usage
        if memory_info['percent'] > self.gc_threshold:
            self.logger.warning(
                f"High memory usage: {memory_info['percent']:.1f}% "
                f"(RSS: {memory_info['rss']:.1f}MB)"
            )
    
    def _save_memory_stats(self):
        """Save final memory statistics."""
        if not self.memory_samples:
            return
        
        stats = {
            'peak_memory_mb': self.peak_memory,
            'samples_count': len(self.memory_samples),
            'final_memory_mb': self.memory_samples[-1]['rss'],
            'average_memory_mb': sum(s['rss'] for s in self.memory_samples) / len(self.memory_samples)
        }
        
        with open(self.log_dir / "memory_summary.json", "w") as f:
            json.dump(stats, f, indent=2)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        if not self.memory_samples:
            return {
                'current': None,
                'peak': None,
                'average': None
            }
        
        samples = list(self.memory_samples)
        return {
            'current': samples[-1],
            'peak': {
                'value': self.peak_memory,
                'unit': 'MB'
            },
            'average': sum(s['rss'] for s in samples) / len(samples)
        }
    
    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self._running:
            self.stop() 