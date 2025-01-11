from typing import Dict, Any
import random
from datetime import datetime

class PhysicalEvaluator:
    def evaluate(self, satellite_id: str) -> Dict[str, Any]:
        """Generate mock physical evaluation data for demo purposes"""
        
        # Mock physical indicators
        indicators = {
            'size_change': True if random.random() > 0.9 else False,
            'brightness_variation': True if random.random() > 0.85 else False,
            'thermal_anomaly': True if random.random() > 0.95 else False,
            'debris_detection': True if random.random() > 0.98 else False,
            'spin_rate_change': True if random.random() > 0.8 else False
        }
        
        # Calculate metadata
        active_indicators = sum(1 for v in indicators.values() if v)
        total_indicators = len(indicators)
        physical_score = float(active_indicators / total_indicators) if total_indicators > 0 else 0.0
        
        return {
            'indicators': indicators,
            'metadata': {
                'active_indicators': int(active_indicators),
                'total_indicators': int(total_indicators),
                'physical_score': float(physical_score),
                'evaluation_timestamp': datetime.utcnow().isoformat(),
                'satellite_id': satellite_id
            }
        } 