import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class EnvironmentalResult:
    eclipse_status: Dict
    orbit_occupancy: Dict
    radiation_assessment: Dict
    environmental_risks: List[Dict]
    confidence: float
    timestamp: datetime

class EnvironmentalNet(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, sequence_length: int = 48):
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
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Eclipse prediction head
        self.eclipse_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3)  # [is_eclipse, umbra_prob, penumbra_prob]
        )
        
        # Orbit occupancy analysis head
        self.occupancy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 4)  # [density, congestion, collision_risk, maneuver_space]
        )
        
        # Radiation environment head
        self.radiation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 5)  # [total_dose, particle_flux, SAA_proximity, belt_region, solar_activity]
        )

    def forward(self, x):
        # Process sequence through LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim*2]
        
        # Apply attention
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        context = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden_dim*2]
        
        # Generate outputs
        eclipse_features = self.eclipse_head(context)
        occupancy_features = self.occupancy_head(context)
        radiation_features = self.radiation_head(context)
        
        return eclipse_features, occupancy_features, radiation_features, attention_weights

class EnvironmentalEvaluator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'input_dim': 128,
            'hidden_dim': 256,
            'sequence_length': 48,
            'confidence_threshold': 0.75
        }
        
        self.model = EnvironmentalNet(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            sequence_length=self.config['sequence_length']
        )
        
        self.risk_types = [
            'RADIATION_EXPOSURE',
            'COLLISION_HAZARD',
            'THERMAL_STRESS',
            'DEBRIS_FIELD',
            'ELECTROMAGNETIC_INTERFERENCE'
        ]

    def _preprocess_data(self, environmental_data: List[Dict]) -> torch.Tensor:
        """Convert environmental data to model input format"""
        sequence = []
        
        for data_point in environmental_data[-self.config['sequence_length']:]:
            features = []
            
            # Orbital parameters
            orbit = data_point.get('orbit', {})
            features.extend([
                orbit.get('altitude', 0),
                orbit.get('local_time', 0),
                orbit.get('beta_angle', 0)
            ])
            
            # Space environment
            environment = data_point.get('environment', {})
            features.extend([
                environment.get('radiation_flux', 0),
                environment.get('particle_density', 0),
                environment.get('magnetic_field', 0),
                environment.get('solar_activity', 0)
            ])
            
            # Nearby objects
            proximity = data_point.get('proximity', {})
            features.extend([
                proximity.get('object_count', 0),
                proximity.get('closest_approach', 0),
                proximity.get('relative_velocity', 0)
            ])
            
            # Pad individual data point features
            while len(features) < self.config['input_dim']:
                features.append(0)
            features = features[:self.config['input_dim']]
            
            sequence.append(features)
        
        # Pad sequence to fixed length
        while len(sequence) < self.config['sequence_length']:
            sequence.insert(0, [0] * self.config['input_dim'])
        
        return torch.FloatTensor(sequence)

    def _process_eclipse_status(self, eclipse_features: torch.Tensor) -> Dict:
        """Process eclipse prediction features"""
        probs = torch.sigmoid(eclipse_features)
        return {
            'in_eclipse': bool(probs[0] > self.config['confidence_threshold']),
            'umbra_probability': float(probs[1]),
            'penumbra_probability': float(probs[2]),
            'duration_estimate': float(probs[1] * 2000 + probs[2] * 1000)  # Simplified duration estimate
        }

    def _process_occupancy_analysis(self, occupancy_features: torch.Tensor) -> Dict:
        """Process orbit occupancy features"""
        features = torch.sigmoid(occupancy_features)
        return {
            'density_level': float(features[0]),
            'congestion_factor': float(features[1]),
            'collision_risk': float(features[2]),
            'maneuver_space': float(features[3]),
            'occupancy_trend': float(torch.mean(features))
        }

    def _process_radiation_assessment(self, radiation_features: torch.Tensor) -> Dict:
        """Process radiation environment features"""
        features = torch.sigmoid(radiation_features)
        return {
            'total_dose_rate': float(features[0]),
            'particle_flux_level': float(features[1]),
            'saa_proximity': float(features[2]),
            'radiation_belt_region': float(features[3]),
            'solar_activity_impact': float(features[4])
        }

    def _identify_environmental_risks(
        self,
        eclipse: Dict,
        occupancy: Dict,
        radiation: Dict
    ) -> List[Dict]:
        """Identify environmental risks based on all assessments"""
        risks = []
        
        # Check radiation risks
        if radiation['total_dose_rate'] > self.config['confidence_threshold']:
            risks.append({
                'type': 'RADIATION_EXPOSURE',
                'severity': float(radiation['total_dose_rate']),
                'confidence': float(radiation['particle_flux_level'])
            })
        
        # Check collision risks
        if occupancy['collision_risk'] > self.config['confidence_threshold']:
            risks.append({
                'type': 'COLLISION_HAZARD',
                'severity': float(occupancy['collision_risk']),
                'confidence': float(occupancy['density_level'])
            })
        
        # Check thermal risks during eclipse
        if eclipse['in_eclipse']:
            risks.append({
                'type': 'THERMAL_STRESS',
                'severity': float(eclipse['umbra_probability']),
                'confidence': float(eclipse['umbra_probability'])
            })
        
        return risks

    async def evaluate_environment(self, environmental_data: List[Dict]) -> EnvironmentalResult:
        """
        Evaluate environmental conditions using the ML model
        """
        try:
            # Preprocess input data
            x = self._preprocess_data(environmental_data)
            
            # Model inference
            with torch.no_grad():
                eclipse_features, occupancy_features, radiation_features, attention = self.model(x.unsqueeze(0))
                
                # Process individual assessments
                eclipse_status = self._process_eclipse_status(eclipse_features[0])
                orbit_occupancy = self._process_occupancy_analysis(occupancy_features[0])
                radiation_assessment = self._process_radiation_assessment(radiation_features[0])
                
                # Identify environmental risks
                risks = self._identify_environmental_risks(
                    eclipse_status,
                    orbit_occupancy,
                    radiation_assessment
                )
                
                # Calculate overall confidence
                confidence = float(torch.mean(attention[0]))
                
                return EnvironmentalResult(
                    eclipse_status=eclipse_status,
                    orbit_occupancy=orbit_occupancy,
                    radiation_assessment=radiation_assessment,
                    environmental_risks=risks,
                    confidence=confidence,
                    timestamp=datetime.now()
                )

        except Exception as e:
            print(f"Error in environmental evaluation: {str(e)}")
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
                    'eclipse_features',
                    'occupancy_features',
                    'radiation_features',
                    'attention_weights'
                ],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'eclipse_features': {0: 'batch_size'},
                    'occupancy_features': {0: 'batch_size'},
                    'radiation_features': {0: 'batch_size'},
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
