from datetime import datetime
from typing import Dict, List
import threading
import time

class AlertConfig:
    THRESHOLDS = {
        'csp_violations': {'warning': 5, 'critical': 10},
        'blocked_requests': {'warning': 10, 'critical': 20},
        'rate_limited': {'warning': 20, 'critical': 50},
        'potential_leaks': {'warning': 1, 'critical': 5},
        'sanitized_errors': {'warning': 50, 'critical': 100}
    }

class MetricsCollector:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.metrics = {
            'https_requests': 0,
            'total_requests': 0,
            'csp_violations': 0,
            'blocked_requests': 0,
            'rate_limited': 0,
            'sanitized_errors': 0,
            'potential_leaks': 0,
        }
        self.alerts = []
        self._start_collection()

    def _start_collection(self):
        def collect():
            while True:
                self._check_alerts()
                time.sleep(300)  # Check alerts every 5 minutes
                with self._lock:
                    # Reset counters every hour
                    if datetime.now().minute == 0:
                        self.metrics = {k: 0 for k in self.metrics.keys()}

        thread = threading.Thread(target=collect, daemon=True)
        thread.start()

    def _check_alerts(self) -> List[Dict]:
        with self._lock:
            new_alerts = []
            for metric, value in self.metrics.items():
                if metric in AlertConfig.THRESHOLDS:
                    thresholds = AlertConfig.THRESHOLDS[metric]
                    if value >= thresholds['critical']:
                        new_alerts.append({
                            'type': metric,
                            'severity': 'critical',
                            'value': value,
                            'threshold': thresholds['critical'],
                            'timestamp': datetime.now().isoformat()
                        })
                    elif value >= thresholds['warning']:
                        new_alerts.append({
                            'type': metric,
                            'severity': 'warning',
                            'value': value,
                            'threshold': thresholds['warning'],
                            'timestamp': datetime.now().isoformat()
                        })
            
            self.alerts = new_alerts
            return new_alerts

    def increment(self, metric: str, value: int = 1):
        with self._lock:
            if metric in self.metrics:
                self.metrics[metric] += value

    def get_current_metrics(self) -> Dict:
        with self._lock:
            https_percentage = (
                (self.metrics['https_requests'] / self.metrics['total_requests'] * 100)
                if self.metrics['total_requests'] > 0
                else 100.0
            )
            return {
                'httpsPercentage': round(https_percentage, 2),
                'cspViolations': self.metrics['csp_violations'],
                'blockedRequests': self.metrics['blocked_requests'],
                'rateLimited': self.metrics['rate_limited'],
                'sanitizedErrors': self.metrics['sanitized_errors'],
                'potentialLeaks': self.metrics['potential_leaks'],
                'timestamp': datetime.now().isoformat()
            }

    def get_active_alerts(self) -> List[Dict]:
        with self._lock:
            return self.alerts.copy()

# Global metrics collector instance
metrics_collector = MetricsCollector() 