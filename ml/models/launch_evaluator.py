from typing import Dict, Any
import torch
import torch.nn as nn
from .base_model import BaseModel

class LaunchEvaluator(BaseModel):
    def __init__(self, input_size=32):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)
        
    def _preprocess_data(self, data: Dict[str, Any]) -> torch.Tensor:
        """Convert launch data to tensor"""
        try:
            features = torch.tensor(data['features'], dtype=torch.float32)
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            return features.to(self.device)
        except Exception as e:
            raise ValueError(f"Invalid input data format: {str(e)}")
            
    def _postprocess_output(self, output: torch.Tensor) -> Dict[str, Any]:
        """Convert model output to launch evaluation"""
        score = output.item()
        confidence = min(abs(score - 0.5) * 2, 1.0)
        
        return {
            'score': score,
            'confidence': confidence,
            'details': {
                'launch_site_risk': 'high' if score > 0.7 else 'moderate' if score > 0.4 else 'low',
                'object_count_anomaly': score > 0.8,
                'registry_status': 'missing' if score > 0.6 else 'registered'
            }
        } 