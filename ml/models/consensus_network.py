from typing import Dict, Any
import torch
import torch.nn as nn
from .base_model import BaseModel

class ConsensusNetwork(BaseModel):
    def __init__(self, input_size=128):
        super().__init__()
        
        # Feature extraction for each evaluation type
        self.feature_nets = nn.ModuleDict({
            'stability': nn.Linear(32, 64),
            'maneuvers': nn.Linear(32, 64),
            'rf': nn.Linear(32, 64),
            'physical': nn.Linear(32, 64),
            'orbital': nn.Linear(32, 64),
            'environmental': nn.Linear(32, 64),
            'launch': nn.Linear(32, 64),
            'compliance': nn.Linear(32, 64)
        })
        
        # Attention mechanism for feature weighting
        self.attention = nn.MultiheadAttention(64, num_heads=4)
        
        # Final consensus layers
        self.consensus_net = nn.Sequential(
            nn.Linear(64 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process each evaluation type
        features = []
        for eval_type, net in self.feature_nets.items():
            if eval_type in x:
                feat = net(x[eval_type])
                features.append(feat)
        
        # Stack features
        stacked = torch.stack(features, dim=0)
        
        # Apply attention
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Flatten and concatenate
        flat = attended.transpose(0, 1).reshape(-1, 64 * 8)
        
        # Get consensus
        return self.consensus_net(flat)
        
    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert evaluation data to tensors"""
        try:
            processed = {}
            for key, value in data.items():
                if isinstance(value, dict) and 'features' in value:
                    tensor = torch.tensor(value['features'], dtype=torch.float32)
                    if len(tensor.shape) == 1:
                        tensor = tensor.unsqueeze(0)
                    processed[key] = tensor.to(self.device)
            return processed
        except Exception as e:
            raise ValueError(f"Invalid input data format: {str(e)}")
            
    def _postprocess_output(self, output: torch.Tensor) -> Dict[str, Any]:
        """Convert model output to consensus evaluation"""
        score = output.item()
        confidence = min(abs(score - 0.5) * 2, 1.0)
        
        # Determine threat level
        if score > 0.8:
            threat_level = 'high'
            recommendation = 'immediate investigation required'
        elif score > 0.6:
            threat_level = 'moderate'
            recommendation = 'continued monitoring recommended'
        else:
            threat_level = 'low'
            recommendation = 'routine monitoring sufficient'
            
        return {
            'score': score,
            'confidence': confidence,
            'details': {
                'threat_level': threat_level,
                'recommendation': recommendation,
                'consensus_strength': 'strong' if confidence > 0.8 else 'moderate' if confidence > 0.5 else 'weak'
            }
        } 