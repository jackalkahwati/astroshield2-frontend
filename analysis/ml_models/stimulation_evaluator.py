import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class StimulationResult:
    is_stimulated: bool
    confidence: float
    stimulation_type: str
    response_characteristics: Dict
    interaction_metrics: Dict
    timestamp: datetime

class StimulationNet(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, sequence_length: int = 32):
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
        
        # Stimulation detection head
        self.stimulation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Stimulation type classification head
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 5)  # Number of stimulation types
        )
        
        # Response characteristics head
        self.response_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 6)  # Number of response characteristics
        )

    def forward(self, x):
        # Process sequence through LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim*2]
        
        # Apply attention
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        context = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden_dim*2]
        
        # Generate outputs
        stimulation_prob = self.stimulation_head(context)
        type_logits = self.type_head(context)
        response_features = self.response_head(context)
        
        return stimulation_prob, type_logits, response_features, attention_weights

class StimulationEvaluator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'input_dim': 128,
            'hidden_dim': 256,
            'sequence_length': 32,
            'confidence_threshold': 0.75
        }
        
        self.model = StimulationNet(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            sequence_length=self.config['sequence_length']
        )
        
        self.stimulation_types = [
            'RF_INTERROGATION',
            'OPTICAL_ILLUMINATION',
            'RADAR_TRACKING',
            'SIGNAL_INJECTION',
            'PROXIMITY_OPERATION'
        ]
        
        self.response_characteristics = [
            'FREQUENCY_SHIFT',
            'POWER_ADJUSTMENT',
            'ORBITAL_CHANGE',
            'SIGNAL_MODIFICATION',
            'EMISSION_PATTERN',
            'TEMPORAL_RESPONSE'
        ]

    def _preprocess_data(self, interaction_data: List[Dict]) -> torch.Tensor:
        """Convert interaction data to model input format"""
        sequence = []
        
        for interaction in interaction_data[-self.config['sequence_length']:]:
            features = []
            
            # Signal characteristics
            signal = interaction.get('signal', {})
            features.extend([
                signal.get('frequency', 0),
                signal.get('power', 0),
                signal.get('bandwidth', 0),
                signal.get('duration', 0)
            ])
            
            # Spatial characteristics
            spatial = interaction.get('spatial', {})
            features.extend([
                spatial.get('range', 0),
                spatial.get('elevation', 0),
                spatial.get('azimuth', 0)
            ])
            
            # Response metrics
            response = interaction.get('response', {})
            features.extend([
                response.get('delay', 0),
                response.get('strength', 0),
                response.get('coherence', 0)
            ])
            
            # Pad individual interaction features
            while len(features) < self.config['input_dim']:
                features.append(0)
            features = features[:self.config['input_dim']]
            
            sequence.append(features)
        
        # Pad sequence to fixed length
        while len(sequence) < self.config['sequence_length']:
            sequence.insert(0, [0] * self.config['input_dim'])
        
        return torch.FloatTensor(sequence)

    def _process_response_features(self, features: torch.Tensor) -> Dict:
        """Convert response features to characteristics dict"""
        characteristics = {}
        for i, char in enumerate(self.response_characteristics):
            characteristics[char] = float(torch.sigmoid(features[i]))
        return characteristics

    def _compute_interaction_metrics(
        self,
        features: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> Dict:
        """Compute interaction metrics from model outputs"""
        return {
            'temporal_coherence': float(torch.mean(attention_weights)),
            'response_strength': float(torch.max(features)),
            'pattern_consistency': float(torch.std(features)),
            'interaction_duration': float(torch.sum(attention_weights > 0.1))
        }

    async def evaluate_stimulation(self, interaction_data: List[Dict]) -> StimulationResult:
        """
        Evaluate system stimulation using the ML model
        """
        try:
            # Preprocess input data
            x = self._preprocess_data(interaction_data)
            
            # Model inference
            with torch.no_grad():
                stim_prob, type_logits, response_features, attention = self.model(x.unsqueeze(0))
                
                # Process outputs
                is_stimulated = bool(stim_prob[0] > self.config['confidence_threshold'])
                type_probs = torch.softmax(type_logits[0], dim=0)
                stim_type = self.stimulation_types[torch.argmax(type_probs)]
                
                # Get response characteristics and metrics
                response_chars = self._process_response_features(response_features[0])
                interaction_metrics = self._compute_interaction_metrics(
                    response_features[0],
                    attention[0]
                )
                
                return StimulationResult(
                    is_stimulated=is_stimulated,
                    confidence=float(stim_prob[0]),
                    stimulation_type=stim_type,
                    response_characteristics=response_chars,
                    interaction_metrics=interaction_metrics,
                    timestamp=datetime.now()
                )

        except Exception as e:
            print(f"Error in stimulation evaluation: {str(e)}")
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
                    'stimulation_prob',
                    'type_logits',
                    'response_features',
                    'attention_weights'
                ],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'stimulation_prob': {0: 'batch_size'},
                    'type_logits': {0: 'batch_size'},
                    'response_features': {0: 'batch_size'},
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
