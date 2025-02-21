"""Performance metrics tracking for trajectory predictions."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import json
import logging
from pathlib import Path

@dataclass
class PredictionMetrics:
    """Container for prediction performance metrics."""
    prediction_id: str
    timestamp: datetime
    actual_impact: Optional[Dict[str, float]]  # lat, lon, time
    predicted_impact: Dict[str, float]  # lat, lon, time
    initial_state: Dict[str, float]  # position, velocity
    confidence: float
    error_metrics: Dict[str, float]  # distance_error, time_error
    environmental_conditions: Dict[str, float]  # atmospheric density, wind, etc.
    computation_time: float  # seconds

class PerformanceTracker:
    """Tracks and logs prediction performance metrics."""
    
    def __init__(self, log_dir: str = "logs/predictions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("prediction_metrics")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        fh = logging.FileHandler(self.log_dir / "prediction_metrics.log")
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(fh)
        
        # Initialize metrics storage
        self.current_metrics: List[PredictionMetrics] = []
        self.validation_cases: Dict[str, Any] = self._load_validation_cases()
    
    def _load_validation_cases(self) -> Dict[str, Any]:
        """Load known validation cases from file."""
        try:
            with open("data/validation_cases.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning("No validation cases file found")
            return {}
    
    def record_prediction(self, 
                         prediction_id: str,
                         predicted_impact: Dict[str, float],
                         initial_state: Dict[str, float],
                         confidence: float,
                         environmental_conditions: Dict[str, float],
                         computation_time: float) -> None:
        """Record a new prediction."""
        metrics = PredictionMetrics(
            prediction_id=prediction_id,
            timestamp=datetime.utcnow(),
            actual_impact=None,  # Will be updated when available
            predicted_impact=predicted_impact,
            initial_state=initial_state,
            confidence=confidence,
            error_metrics={},  # Will be updated when actual impact is known
            environmental_conditions=environmental_conditions,
            computation_time=computation_time
        )
        
        self.current_metrics.append(metrics)
        self._log_prediction(metrics)
    
    def update_with_actual(self, 
                          prediction_id: str,
                          actual_impact: Dict[str, float]) -> None:
        """Update metrics with actual impact data."""
        for metrics in self.current_metrics:
            if metrics.prediction_id == prediction_id:
                metrics.actual_impact = actual_impact
                metrics.error_metrics = self._calculate_errors(
                    metrics.predicted_impact,
                    actual_impact
                )
                self._log_validation(metrics)
                break
    
    def _calculate_errors(self,
                         predicted: Dict[str, float],
                         actual: Dict[str, float]) -> Dict[str, float]:
        """Calculate prediction errors."""
        # Haversine distance for lat/lon
        distance_error = self._haversine_distance(
            predicted['lat'], predicted['lon'],
            actual['lat'], actual['lon']
        )
        
        # Time error in seconds
        time_error = abs(predicted['time'] - actual['time'])
        
        return {
            'distance_error_km': distance_error,
            'time_error_seconds': time_error
        }
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def _log_prediction(self, metrics: PredictionMetrics) -> None:
        """Log prediction metrics."""
        self.logger.info(f"New prediction recorded: {metrics.prediction_id}")
        
        # Save to JSON file
        with open(self.log_dir / f"prediction_{metrics.prediction_id}.json", "w") as f:
            json.dump({
                'prediction_id': metrics.prediction_id,
                'timestamp': metrics.timestamp.isoformat(),
                'predicted_impact': metrics.predicted_impact,
                'initial_state': metrics.initial_state,
                'confidence': metrics.confidence,
                'environmental_conditions': metrics.environmental_conditions,
                'computation_time': metrics.computation_time
            }, f, indent=2)
    
    def _log_validation(self, metrics: PredictionMetrics) -> None:
        """Log validation results."""
        self.logger.info(
            f"Validation update for prediction {metrics.prediction_id}: "
            f"Distance error: {metrics.error_metrics['distance_error_km']:.2f} km, "
            f"Time error: {metrics.error_metrics['time_error_seconds']:.2f} s"
        )
        
        # Update JSON file
        with open(self.log_dir / f"prediction_{metrics.prediction_id}.json", "r+") as f:
            data = json.load(f)
            data['actual_impact'] = metrics.actual_impact
            data['error_metrics'] = metrics.error_metrics
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Generate summary statistics for validated predictions."""
        validated_metrics = [m for m in self.current_metrics if m.actual_impact]
        
        if not validated_metrics:
            return {}
        
        distance_errors = [m.error_metrics['distance_error_km'] for m in validated_metrics]
        time_errors = [m.error_metrics['time_error_seconds'] for m in validated_metrics]
        confidences = [m.confidence for m in validated_metrics]
        
        return {
            'mean_distance_error_km': np.mean(distance_errors),
            'std_distance_error_km': np.std(distance_errors),
            'mean_time_error_seconds': np.mean(time_errors),
            'std_time_error_seconds': np.std(time_errors),
            'mean_confidence': np.mean(confidences),
            'total_predictions': len(validated_metrics)
        }
    
    def validate_against_known_cases(self) -> Dict[str, Any]:
        """Compare predictions against known validation cases."""
        results = {
            'total_cases': len(self.validation_cases),
            'passed_cases': 0,
            'failed_cases': 0,
            'details': []
        }
        
        for case_id, case_data in self.validation_cases.items():
            # Find matching prediction
            matching_metrics = [
                m for m in self.current_metrics
                if m.initial_state == case_data['initial_state']
            ]
            
            if not matching_metrics:
                continue
                
            metrics = matching_metrics[0]
            errors = self._calculate_errors(
                metrics.predicted_impact,
                case_data['actual_impact']
            )
            
            passed = (
                errors['distance_error_km'] <= case_data['max_distance_error']
                and errors['time_error_seconds'] <= case_data['max_time_error']
            )
            
            results['details'].append({
                'case_id': case_id,
                'passed': passed,
                'errors': errors
            })
            
            if passed:
                results['passed_cases'] += 1
            else:
                results['failed_cases'] += 1
        
        return results 