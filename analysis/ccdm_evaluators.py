from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel
import numpy as np

class CCDMIndicator(BaseModel):
    indicator_name: str
    is_detected: bool
    confidence_level: float
    evidence: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'indicator_name': self.indicator_name,
            'is_detected': self.is_detected,
            'confidence_level': self.confidence_level,
            'evidence': self.evidence,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class CCDMEvaluator:
    def evaluate(self, data: np.ndarray) -> float:
        """
        Evaluate data using CCDM metrics
        Returns a score between 0 and 1
        """
        if data.size == 0:
            return 0.0
            
        # Implement CCDM evaluation logic
        mean_value = np.mean(data)
        std_value = np.std(data)
        
        # Normalize score between 0 and 1
        score = np.clip(mean_value / (std_value + 1e-6), 0, 1)
        
        return float(score)
