"""Real-time monitoring system for trajectory predictions."""

import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from threading import Thread, Lock
from queue import Queue
import psutil
from collections import deque
from weakref import WeakSet

class SystemMetrics:
    def __init__(self, **kwargs):
        self.timestamp = kwargs.get('timestamp', datetime.now())
        self.cpu_usage = kwargs.get('cpu_usage', 0.0)
        self.memory_usage = kwargs.get('memory_usage', 0.0)
        self.network_io = kwargs.get('network_io', 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'network_io': self.network_io
        }

class RealTimeMonitor:
    """Monitors prediction system performance in real-time."""
    
    def __init__(self, max_metrics: int = 3600):
        self.max_metrics = max_metrics
        self.current_metrics = deque(maxlen=max_metrics)
        self._metric_buffer = []
        self._buffer_size = 100
        self._last_flush_time = time.time()
        self._flush_interval = 60  # seconds
        self._running = False
        self._setup_logging()
        self.alert_subscribers = WeakSet()  # Use WeakSet for subscribers
        
        # Configure logging
        self.log_dir = Path("logs/real_time")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("real_time_monitor")
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(self.log_dir / "real_time_metrics.log")
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(fh)
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': 80.0,  # percent
            'memory_usage': 80.0,  # percent
            'prediction_latency': 5.0,  # seconds
            'queue_size': 100  # maximum predictions in queue
        }
    
    def _setup_logging(self):
        """Set up logging for the monitor."""
        log_dir = Path("logs/monitor")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("real_time_monitor")
        if not self.logger.handlers:
            handler = logging.FileHandler(log_dir / "monitor.log")
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def start(self):
        """Start the monitor."""
        self._running = True
        self.logger.info("RealTimeMonitor started")
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop the monitor and clean up resources."""
        self._running = False
        self._flush_metric_buffer()
        self.logger.info("RealTimeMonitor stopped")
        self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            metrics = self._collect_metrics()
            self._process_metrics(metrics)
            time.sleep(1.0)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            metrics = SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                network_io=psutil.net_io_counters().bytes_recv,
                timestamp=datetime.utcnow()
            )
            
            # Add to buffer for batch processing
            self._metric_buffer.append(metrics)
            
            # Check if we should flush the buffer
            if len(self._metric_buffer) >= self._buffer_size or \
               time.time() - self._last_flush_time >= self._flush_interval:
                self._flush_metric_buffer()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, datetime.utcnow())
    
    def _flush_metric_buffer(self):
        """Flush the metric buffer to storage."""
        if not self._metric_buffer:
            return

        try:
            # Write metrics to storage
            storage_dir = Path("data/metrics")
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Use current date for file organization
            date_str = datetime.now().strftime("%Y%m%d")
            file_path = storage_dir / f"metrics_{date_str}.jsonl"
            
            # Write metrics in JSONL format
            with open(file_path, 'a') as f:
                for metric in self._metric_buffer:
                    f.write(json.dumps(metric.to_dict()) + '\n')
            
            # Clear buffer and update flush time
            self._metric_buffer = []
            self._last_flush_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Error flushing metric buffer: {str(e)}")
    
    def _process_metrics(self, metrics: SystemMetrics):
        """Process and log metrics, trigger alerts if needed."""
        # Log metrics
        self._log_metrics(metrics)
        
        # Check thresholds and trigger alerts
        alerts = []
        
        if metrics.cpu_usage > self.thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.thresholds['memory_usage']:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.network_io > self.thresholds['network_io']:
            alerts.append(f"High network IO usage: {metrics.network_io:.2f} bytes")
        
        if alerts:
            self._trigger_alerts(alerts)
    
    def _log_metrics(self, metrics: SystemMetrics):
        """Log current metrics."""
        # Log to file
        with open(self.log_dir / "metrics.jsonl", "a") as f:
            json.dump({
                'timestamp': metrics.timestamp.isoformat(),
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'network_io': metrics.network_io
            }, f)
            f.write('\n')
        
        # Log significant changes
        self.logger.info(
            f"Metrics - CPU: {metrics.cpu_usage:.1f}%, "
            f"Memory: {metrics.memory_usage:.1f}%, "
            f"Network IO: {metrics.network_io:.2f} bytes"
        )
    
    def _trigger_alerts(self, alerts: List[str]):
        """Trigger alerts for threshold violations."""
        alert_msg = "ALERT: " + "; ".join(alerts)
        self.logger.warning(alert_msg)
        
        # Save alert to separate file
        with open(self.log_dir / "alerts.log", "a") as f:
            f.write(f"{datetime.utcnow().isoformat()}: {alert_msg}\n")
    
    def get_recent_metrics(self, count: Optional[int] = None) -> list:
        """Get the most recent metrics."""
        if count is None or count > len(self.current_metrics):
            return list(self.current_metrics)
        return list(self.current_metrics)[-count:]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Generate a summary of current metrics."""
        if not self.current_metrics:
            return {
                'count': 0,
                'timestamp_range': None,
                'averages': None
            }

        metrics = list(self.current_metrics)
        return {
            'count': len(metrics),
            'timestamp_range': {
                'start': metrics[0].timestamp.isoformat(),
                'end': metrics[-1].timestamp.isoformat()
            },
            'averages': {
                'cpu_usage': sum(m.cpu_usage for m in metrics) / len(metrics),
                'memory_usage': sum(m.memory_usage for m in metrics) / len(metrics),
                'network_io': sum(m.network_io for m in metrics) / len(metrics)
            }
        }

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self._running:
            self.stop()

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update monitoring thresholds."""
        self.thresholds.update(new_thresholds)
        self.logger.info(f"Updated monitoring thresholds: {self.thresholds}") 