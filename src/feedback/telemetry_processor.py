import time
import json
import logging
import weakref
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Dict, List, Any, Optional

class TelemetryProcessor:
    def __init__(self, batch_size: int = 100, flush_interval: int = 60):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.telemetry_buffer = deque(maxlen=1000)  # Fixed-size buffer
        self._subscribers = weakref.WeakSet()  # Use weak references for subscribers
        self._metric_buffer = []
        self._last_flush_time = time.time()
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging for telemetry processing."""
        log_dir = Path("logs/telemetry")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("telemetry_processor")
        if not self.logger.handlers:
            handler = logging.FileHandler(log_dir / "telemetry.log")
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def add_telemetry(self, telemetry_data: Dict[str, Any]) -> None:
        """Add telemetry data to the processing queue."""
        try:
            # Add timestamp if not present
            if 'timestamp' not in telemetry_data:
                telemetry_data['timestamp'] = datetime.now().isoformat()
            
            self.telemetry_buffer.append(telemetry_data)
            self._metric_buffer.append(telemetry_data)
            
            # Check if we should flush the buffer
            if len(self._metric_buffer) >= self.batch_size or \
               time.time() - self._last_flush_time >= self.flush_interval:
                self._flush_metric_buffer()
                
        except Exception as e:
            self.logger.error(f"Error adding telemetry data: {str(e)}")

    def _flush_metric_buffer(self) -> None:
        """Flush the metric buffer to storage and notify subscribers."""
        if not self._metric_buffer:
            return

        try:
            # Process metrics in batch
            metrics = self._process_metrics(self._metric_buffer)
            
            # Notify subscribers
            for subscriber in self._subscribers:
                try:
                    subscriber(metrics)
                except Exception as e:
                    self.logger.error(f"Error notifying subscriber: {str(e)}")
            
            # Write to storage
            self._write_to_storage(self._metric_buffer)
            
            # Clear buffer and update flush time
            self._metric_buffer = []
            self._last_flush_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Error flushing metric buffer: {str(e)}")

    def _process_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of metrics to generate aggregated statistics."""
        try:
            # Initialize aggregates
            aggregates = {
                'count': len(metrics),
                'timestamp_range': {
                    'start': metrics[0]['timestamp'],
                    'end': metrics[-1]['timestamp']
                },
                'satellites': set(),
                'position_stats': {
                    'x': {'min': float('inf'), 'max': float('-inf'), 'sum': 0},
                    'y': {'min': float('inf'), 'max': float('-inf'), 'sum': 0},
                    'z': {'min': float('inf'), 'max': float('-inf'), 'sum': 0}
                }
            }
            
            # Process each metric
            for metric in metrics:
                # Track unique satellites
                if 'satellite_id' in metric:
                    aggregates['satellites'].add(metric['satellite_id'])
                
                # Update position stats
                if 'position' in metric:
                    pos = metric['position']
                    for axis in ['x', 'y', 'z']:
                        if axis in pos:
                            val = float(pos[axis])
                            stats = aggregates['position_stats'][axis]
                            stats['min'] = min(stats['min'], val)
                            stats['max'] = max(stats['max'], val)
                            stats['sum'] += val
            
            # Convert sets to lists for JSON serialization
            aggregates['satellites'] = list(aggregates['satellites'])
            
            # Calculate averages
            for axis in ['x', 'y', 'z']:
                stats = aggregates['position_stats'][axis]
                stats['avg'] = stats['sum'] / len(metrics)
                del stats['sum']  # Remove sum to save memory
                
            return aggregates
            
        except Exception as e:
            self.logger.error(f"Error processing metrics: {str(e)}")
            return {'error': str(e)}

    def _write_to_storage(self, metrics: List[Dict[str, Any]]) -> None:
        """Write metrics to persistent storage."""
        try:
            storage_dir = Path("data/telemetry")
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Use current date for file organization
            date_str = datetime.now().strftime("%Y%m%d")
            file_path = storage_dir / f"telemetry_{date_str}.jsonl"
            
            # Append metrics to file in JSONL format
            with open(file_path, 'a') as f:
                for metric in metrics:
                    f.write(json.dumps(metric) + '\n')
                    
        except Exception as e:
            self.logger.error(f"Error writing to storage: {str(e)}")

    def subscribe(self, callback) -> None:
        """Subscribe to telemetry updates."""
        self._subscribers.add(callback)

    def unsubscribe(self, callback) -> None:
        """Unsubscribe from telemetry updates."""
        self._subscribers.discard(callback)

    def cleanup(self) -> None:
        """Clean up resources and flush remaining data."""
        try:
            if self._metric_buffer:
                self._flush_metric_buffer()
            self._subscribers.clear()
            self.telemetry_buffer.clear()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Ensure cleanup is called when the object is destroyed."""
        self.cleanup() 