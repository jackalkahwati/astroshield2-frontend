from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.preprocessing import StandardScaler
from .base_evaluator import BaseEvaluator
from .ccdm_evaluators import CCDMIndicator

class AdvancedEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.ccdm = CCDMIndicator()

    def evaluate(self, data: Dict[str, Union[float, List[float]]]) -> Dict[str, float]:
        processed_data = self._preprocess_data(data)
        base_metrics = super().evaluate(data)
        advanced_metrics = self._calculate_advanced_metrics(processed_data)
        return {**base_metrics, **advanced_metrics}

    def _preprocess_data(self, data: Dict[str, Union[float, List[float]]]) -> np.ndarray:
        features = []
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, list):
                features.extend(value)
            else:
                features.append(value)
        return np.array(features).reshape(1, -1)

    def _calculate_advanced_metrics(self, data: np.ndarray) -> Dict[str, float]:
        scaled_data = self.scaler.fit_transform(data)
        ccdm_score = self.ccdm.evaluate(scaled_data)
        
        return {
            "complexity_score": float(np.mean(scaled_data)),
            "stability_index": float(np.std(scaled_data)),
            "ccdm_indicator": float(ccdm_score)
        }
