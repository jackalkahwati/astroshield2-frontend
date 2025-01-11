import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ThreatDetectionResult:
    is_threat: bool
    confidence: float
    threat_type: Optional[str]
    severity: float
    features: List[str]
    timestamp: datetime

class AttentionLayer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        attended = torch.sum(x * attention_weights, dim=1)
        return attended, attention_weights

class ThreatDetectorNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.attention = AttentionLayer(hidden_dim, hidden_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 4)  # [is_threat, confidence, severity, threat_type]
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        attended, attention_weights = self.attention(features.unsqueeze(1))
        output = self.classifier(attended)
        return output, attention_weights

class ThreatDetector:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'confidence_threshold': 0.75,
            'feature_importance_threshold': 0.5,
            'input_dim': 32,
            'hidden_dim': 128
        }
        
        self.model = ThreatDetectorNN(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim']
        )
        
        self.threat_types = ['CYBER', 'KINETIC', 'ELECTRONIC', 'UNKNOWN']

    def _preprocess_data(self, sensor_data: Dict) -> torch.Tensor:
        """Convert sensor data to model input format"""
        features = []
        
        # Extract numerical features
        features.extend([
            sensor_data.get('signal_strength', 0),
            sensor_data.get('frequency', 0),
            sensor_data.get('duration', 0),
            sensor_data.get('power_level', 0)
        ])
        
        # Position features
        pos = sensor_data.get('position', {'x': 0, 'y': 0, 'z': 0})
        features.extend([pos['x'], pos['y'], pos['z']])
        
        # Velocity features
        vel = sensor_data.get('velocity', {'vx': 0, 'vy': 0, 'vz': 0})
        features.extend([vel['vx'], vel['vy'], vel['vz']])
        
        # Pad or truncate to match input_dim
        while len(features) < self.config['input_dim']:
            features.append(0)
        features = features[:self.config['input_dim']]
        
        return torch.FloatTensor(features)

    def _postprocess_output(self, output: torch.Tensor, attention_weights: torch.Tensor) -> Dict:
        """Convert model output to threat detection result"""
        probs = torch.sigmoid(output[0])
        is_threat = bool(probs[0] > self.config['confidence_threshold'])
        confidence = float(probs[1])
        severity = float(probs[2])
        threat_type_idx = torch.argmax(output[0][3:]).item()
        
        return ThreatDetectionResult(
            is_threat=is_threat,
            confidence=confidence,
            threat_type=self.threat_types[threat_type_idx],
            severity=severity,
            features=[f"feature_{i}" for i in range(len(attention_weights))],
            timestamp=datetime.now()
        )

    async def detect_threats(self, sensor_data: Dict) -> Dict:
        """
        Detect threats using the ML model
        """
        try:
            # Preprocess input data
            x = self._preprocess_data(sensor_data)
            
            # Model inference
            with torch.no_grad():
                output, attention_weights = self.model(x.unsqueeze(0))
                result = self._postprocess_output(output, attention_weights)
            
            # Convert to dictionary format
            return {
                'id': sensor_data.get('id', 1),
                'type': result.threat_type,
                'confidence': result.confidence,
                'severity': result.severity,
                'is_threat': result.is_threat,
                'timestamp': result.timestamp.isoformat()
            }

        except Exception as e:
            print(f"Error in threat detection: {str(e)}")
            return []

    def export_to_onnx(self, path: str):
        """Export model to ONNX format"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, self.config['input_dim'])
            
            # Export the model
            torch.onnx.export(
                self.model,
                dummy_input,
                path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output', 'attention_weights'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                    'attention_weights': {0: 'batch_size'}
                }
            )
            
            print(f"Model exported to: {path}")
            
            # Verify the exported model
            import onnx
            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model check passed")
            
        except Exception as e:
            print(f"Error exporting model: {str(e)}")
