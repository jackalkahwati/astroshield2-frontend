from typing import Dict, Any
import torch
import torch.nn as nn
from .base_model import BaseModel

class TrackEvaluator(BaseModel):
    def __init__(self, input_size=64, hidden_size=128):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
        
    def _preprocess_data(self, data: Dict[str, Any]) -> torch.Tensor:
        """Convert tracking data to tensor"""
        try:
            # Extract features
            features = torch.tensor(data['features'], dtype=torch.float32)
            if len(features.shape) == 2:
                features = features.unsqueeze(0)
            return features.to(self.device)
        except Exception as e:
            raise ValueError(f"Invalid input data format: {str(e)}")
            
    def _postprocess_output(self, output: torch.Tensor) -> Dict[str, Any]:
        """Convert model output to tracking evaluation"""
        score = output.item()
        confidence = min(abs(score - 0.5) * 2, 1.0)
        
        return {
            'score': score,
            'confidence': confidence,
            'details': {
                'tracking_quality': 'good' if score > 0.7 else 'poor',
                'needs_adjustment': score < 0.5
            }
        } 