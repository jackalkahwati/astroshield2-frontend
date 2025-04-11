import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import BaseModel
from .base_model import BaseModel

class MultiModalEncoder(nn.Module):
    """Encoder for different types of space object data"""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_size: int = 128,
        latent_dim: int = 64
    ):
        super().__init__()
        
        # Create encoders for each data type
        self.encoders = nn.ModuleDict({
            'physical': nn.Sequential(
                nn.Linear(input_dims['physical'], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_dim)
            ),
            'orbital': nn.Sequential(
                nn.Linear(input_dims['orbital'], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_dim)
            ),
            'signature': nn.Sequential(
                nn.Linear(input_dims['signature'], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_dim)
            ),
            'temporal': nn.LSTM(
                input_size=input_dims['temporal'],
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True
            )
        })
        
        self.temporal_projection = nn.Linear(hidden_size, latent_dim)
        self.latent_dim = latent_dim
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Encode different types of inputs"""
        encodings = {}
        
        # Encode each available input type
        for data_type, data in inputs.items():
            if data_type == 'temporal':
                # Process temporal data through LSTM
                lstm_out, _ = self.encoders[data_type](data)
                encoding = self.temporal_projection(lstm_out[:, -1])
            else:
                # Process other data types through their respective encoders
                encoding = self.encoders[data_type](data)
            
            encodings[data_type] = encoding
        
        return encodings

class MultiModalDecoder(nn.Module):
    """Decoder for different types of space object data"""
    
    def __init__(
        self,
        output_dims: Dict[str, int],
        hidden_size: int = 128,
        latent_dim: int = 64
    ):
        super().__init__()
        
        # Create decoders for each data type
        self.decoders = nn.ModuleDict({
            'physical': nn.Sequential(
                nn.Linear(latent_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dims['physical'])
            ),
            'orbital': nn.Sequential(
                nn.Linear(latent_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dims['orbital'])
            ),
            'signature': nn.Sequential(
                nn.Linear(latent_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dims['signature'])
            ),
            'temporal': nn.Sequential(
                nn.Linear(latent_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dims['temporal'])
            )
        })
    
    def forward(
        self,
        latent: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Decode latent representations"""
        reconstructions = {}
        
        # Decode each encoded type
        for data_type, encoding in latent.items():
            reconstruction = self.decoders[data_type](encoding)
            reconstructions[data_type] = reconstruction
        
        return reconstructions

class AnomalyDetector(BaseModel):
    """Multi-modal anomaly detection model, inheriting from BaseModel."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_size: int = 128,
        latent_dim: int = 64,
        num_flows: int = 3
    ):
        # Call BaseModel init first
        super().__init__()
        
        self.input_dims = input_dims # Store input_dims for preprocessing
        self.encoder = MultiModalEncoder(input_dims, hidden_size, latent_dim)
        self.decoder = MultiModalDecoder(input_dims, hidden_size, latent_dim)
        
        # Normalizing flow for better density estimation
        self.flows = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.Tanh()
            ) for _ in range(num_flows)
        ])
        
        # Anomaly scoring networks
        self.score_nets = nn.ModuleDict({
            data_type: nn.Sequential(
                nn.Linear(latent_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            ) for data_type in input_dims.keys()
        })

        # Ensure model components are moved to the correct device
        self.to(self.device)
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass, processing inputs through encoder, flows, decoder, and scoring nets."""
        # Encode inputs
        encodings = self.encoder(inputs)
        
        # Apply normalizing flows
        transformed = {}
        log_det = {}
        
        for data_type, encoding in encodings.items():
            z = encoding
            log_det_sum = 0
            
            # Note: Jacobian calculation can be computationally expensive and numerically unstable.
            # A simpler approach might be needed in practice, or use a library that handles flows efficiently.
            for flow in self.flows:
                z_new = flow(z)
                # Simplification: Skip Jacobian determinant calculation for now 
                # as it requires careful implementation and might not be stable.
                # log_det_sum += torch.log(torch.abs(
                #     torch.det(torch.autograd.functional.jacobian(flow, z))
                # ))
                z = z_new
            
            transformed[data_type] = z
            log_det[data_type] = log_det_sum # Will be 0 with the simplification above
        
        # Decode transformed representations
        reconstructions = self.decoder(transformed)
        
        # Calculate anomaly scores
        scores = {}
        for data_type, z in transformed.items():
            scores[data_type] = self.score_nets[data_type](z)
        
        return {
            'encodings': encodings,
            'transformed': transformed,
            'reconstructions': reconstructions,
            'log_det': log_det,
            'anomaly_scores': scores
        }

    def _preprocess_data(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert input numpy arrays to tensors on the correct device."""
        # Assuming input data is a dictionary of numpy arrays
        processed_data = {}
        for key, numpy_array in data.items():
            if key in self.input_dims: # Process only keys defined in input_dims
                # Add batch dimension if missing
                if numpy_array.ndim == 1:
                     numpy_array = numpy_array.reshape(1, -1)
                elif numpy_array.ndim == 2 and key == 'temporal': # Temporal data might have sequence length
                     numpy_array = numpy_array.reshape(1, numpy_array.shape[0], numpy_array.shape[1])
                elif numpy_array.ndim == 0:
                     numpy_array = numpy_array.reshape(1,1)
                     
                processed_data[key] = torch.from_numpy(numpy_array).float().to(self.device)
        return processed_data

    def _postprocess_output(self, output: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Convert model output tensors to a dictionary with scores and confidence."""
        # Calculate an overall anomaly score (e.g., max score across modalities)
        anomaly_scores = output.get('anomaly_scores', {})
        if not anomaly_scores:
            overall_score = 0.0
        else:
            # Example: Use the maximum score from any modality
            overall_score = torch.max(torch.cat([s.view(-1) for s in anomaly_scores.values()])).item()
            
        # Confidence could be related to the score or a separate metric
        # For simplicity, let's use the score as confidence here
        confidence = overall_score

        # Add detailed scores per modality
        details = {
            f"{key}_score": score.mean().item()
            for key, score in anomaly_scores.items()
        }
        
        return {
            "score": overall_score, # The primary anomaly score
            "confidence": confidence, # Confidence in the score
            "details": details # Additional details like per-modality scores
        }
    
    def detect_anomalies(
        self,
        inputs: Dict[str, torch.Tensor],
        thresholds: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """Detect anomalies in the input data (using forward pass outputs)."""
        self.eval()
        with torch.no_grad():
            # Note: The input here is expected to be preprocessed tensors already
            outputs = self.forward(inputs) 
            
            anomalies = {}
            for data_type, scores in outputs['anomaly_scores'].items():
                if data_type in thresholds:
                     anomalies[data_type] = scores > thresholds[data_type]
            
            return anomalies 