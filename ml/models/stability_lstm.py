import torch
import torch.nn as nn
import torch.onnx
import numpy as np
from typing import Tuple, Dict, Any

class StabilityLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, sequence_length=60):
        """
        LSTM model for stability evaluation
        Args:
            input_size: Number of input features (position + velocity)
            hidden_size: Size of LSTM hidden layers
            num_layers: Number of LSTM layers
            sequence_length: Length of input sequences
        """
        super(StabilityLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Dense layers for stability prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 4, 3)  # [stability_score, family_deviation, anomaly_score]
        )
        
        # Thresholds
        self.stability_threshold = 0.8
        self.family_threshold = 2.0  # standard deviations
        self.anomaly_threshold = 0.9

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            predictions: Tensor of shape (batch_size, 3) containing [stability, family_dev, anomaly]
            attention_weights: Attention weights for interpretability
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_size)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # Shape: (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(lstm_out * attention_weights, dim=1)  # Shape: (batch_size, hidden_size)
        
        # Final predictions
        predictions = self.fc_layers(context)  # Shape: (batch_size, 3)
        
        return predictions, attention_weights

    def analyze_stability(self, orbit_data: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze orbital stability
        Args:
            orbit_data: Tensor of shape (sequence_length, input_size) containing orbital parameters
        Returns:
            Dictionary containing stability analysis results
        """
        self.eval()
        with torch.no_grad():
            # Prepare input
            x = orbit_data.unsqueeze(0)  # Add batch dimension
            
            # Get predictions
            predictions, attention = self.forward(x)
            stability_score, family_deviation, anomaly_score = predictions[0]
            
            # Analyze results
            is_stable = stability_score > self.stability_threshold
            is_in_family = abs(family_deviation) < self.family_threshold
            is_anomalous = anomaly_score > self.anomaly_threshold
            
            return {
                'is_stable': bool(is_stable),
                'stability_score': float(stability_score),
                'family_deviation': float(family_deviation),
                'anomaly_score': float(anomaly_score),
                'attention_weights': attention.squeeze().numpy(),
                'confidence': float(1.0 - anomaly_score) if is_stable else float(anomaly_score)
            }

    def export_to_onnx(self, save_path: str, input_shape: Tuple[int, int, int]):
        """
        Export model to ONNX format
        Args:
            save_path: Path to save the ONNX model
            input_shape: Shape of input tensor (batch_size, sequence_length, input_size)
        """
        dummy_input = torch.randn(input_shape)
        torch.onnx.export(
            self,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['predictions', 'attention'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'predictions': {0: 'batch_size'},
                'attention': {0: 'batch_size'}
            }
        )
