import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class LaunchResult:
    is_nominal: bool
    confidence: float
    anomalies: List[Dict]
    object_count: int
    launch_characteristics: Dict
    threat_assessment: Dict
    timestamp: datetime

class LaunchNet(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, sequence_length: int = 24):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # Temporal feature extraction (LSTM)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention for temporal features
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Launch normality assessment head
        self.normality_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Object count prediction head
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)  # Direct count prediction
        )
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 6)  # Number of anomaly types
        )
        
        # Threat assessment head
        self.threat_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 4)  # Threat characteristics
        )

    def forward(self, x):
        # Process sequence through LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim*2]
        
        # Apply attention
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        context = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden_dim*2]
        
        # Generate outputs
        normality_score = self.normality_head(context)
        object_count = self.count_head(context)
        anomaly_logits = self.anomaly_head(context)
        threat_features = self.threat_head(context)
        
        return normality_score, object_count, anomaly_logits, threat_features, attention_weights

class LaunchEvaluator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'input_dim': 128,
            'hidden_dim': 256,
            'sequence_length': 24,
            'confidence_threshold': 0.75
        }
        
        self.model = LaunchNet(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            sequence_length=self.config['sequence_length']
        )
        
        self.anomaly_types = [
            'UNEXPECTED_OBJECT_COUNT',
            'UNUSUAL_TRAJECTORY',
            'DEBRIS_ANOMALY',
            'ENERGY_MISMATCH',
            'TIMING_IRREGULARITY',
            'SIGNATURE_MISMATCH'
        ]
        
        self.threat_characteristics = [
            'MILITARY_INDICATOR',
            'TECHNOLOGY_LEVEL',
            'PAYLOAD_RISK',
            'TRAJECTORY_CONCERN'
        ]

    def _preprocess_data(self, launch_data: List[Dict]) -> torch.Tensor:
        """Convert launch tracking data to model input format"""
        sequence = []
        
        for tracking_point in launch_data[-self.config['sequence_length']:]:
            features = []
            
            # Object tracking features
            tracking = tracking_point.get('tracking', {})
            features.extend([
                tracking.get('object_count', 0),
                tracking.get('track_quality', 0),
                tracking.get('track_confidence', 0)
            ])
            
            # Trajectory features
            trajectory = tracking_point.get('trajectory', {})
            features.extend([
                trajectory.get('altitude', 0),
                trajectory.get('velocity', 0),
                trajectory.get('acceleration', 0),
                trajectory.get('inclination', 0)
            ])
            
            # Energy/signature features
            signature = tracking_point.get('signature', {})
            features.extend([
                signature.get('thermal', 0),
                signature.get('radar_cross_section', 0),
                signature.get('spectral', 0)
            ])
            
            # Pad individual tracking point features
            while len(features) < self.config['input_dim']:
                features.append(0)
            features = features[:self.config['input_dim']]
            
            sequence.append(features)
        
        # Pad sequence to fixed length
        while len(sequence) < self.config['sequence_length']:
            sequence.insert(0, [0] * self.config['input_dim'])
        
        return torch.FloatTensor(sequence)

    def _process_anomalies(self, anomaly_logits: torch.Tensor) -> List[Dict]:
        """Convert anomaly logits to structured anomaly information"""
        anomalies = []
        probs = torch.sigmoid(anomaly_logits)
        
        for i, prob in enumerate(probs):
            if prob > self.config['confidence_threshold']:
                anomalies.append({
                    'type': self.anomaly_types[i],
                    'confidence': float(prob),
                    'severity': float(prob * 0.8)  # Simplified severity calculation
                })
        
        return anomalies

    def _process_threat_assessment(self, threat_features: torch.Tensor) -> Dict:
        """Convert threat features to threat assessment"""
        assessment = {}
        for i, char in enumerate(self.threat_characteristics):
            assessment[char] = float(torch.sigmoid(threat_features[i]))
        
        # Calculate overall threat level
        threat_level = float(torch.mean(torch.sigmoid(threat_features)))
        assessment['overall_threat_level'] = threat_level
        
        return assessment

    async def evaluate_launch(self, launch_data: List[Dict]) -> LaunchResult:
        """
        Evaluate launch characteristics using the ML model
        """
        try:
            # Preprocess input data
            x = self._preprocess_data(launch_data)
            
            # Model inference
            with torch.no_grad():
                normality, count, anomaly_logits, threat_features, attention = self.model(x.unsqueeze(0))
                
                # Process outputs
                is_nominal = bool(normality[0] > self.config['confidence_threshold'])
                object_count = int(torch.round(count[0]).item())
                
                # Get anomalies and threat assessment
                anomalies = self._process_anomalies(anomaly_logits[0])
                threat_assessment = self._process_threat_assessment(threat_features[0])
                
                # Compile launch characteristics
                characteristics = {
                    'track_quality': float(torch.mean(attention[0])),
                    'trajectory_confidence': float(normality[0]),
                    'energy_profile': float(torch.mean(threat_features[0])),
                    'temporal_consistency': float(torch.std(attention[0]))
                }
                
                return LaunchResult(
                    is_nominal=is_nominal,
                    confidence=float(normality[0]),
                    anomalies=anomalies,
                    object_count=object_count,
                    launch_characteristics=characteristics,
                    threat_assessment=threat_assessment,
                    timestamp=datetime.now()
                )

        except Exception as e:
            print(f"Error in launch evaluation: {str(e)}")
            return None

    def export_to_onnx(self, path: str):
        """Export model to ONNX format"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, self.config['sequence_length'], self.config['input_dim'])
            
            # Export the model
            torch.onnx.export(
                self.model,
                dummy_input,
                path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=[
                    'normality_score',
                    'object_count',
                    'anomaly_logits',
                    'threat_features',
                    'attention_weights'
                ],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'normality_score': {0: 'batch_size'},
                    'object_count': {0: 'batch_size'},
                    'anomaly_logits': {0: 'batch_size'},
                    'threat_features': {0: 'batch_size'},
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
