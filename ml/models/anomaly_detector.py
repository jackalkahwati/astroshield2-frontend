import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

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

class AnomalyDetector(nn.Module):
    """Multi-modal anomaly detection model"""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_size: int = 128,
        latent_dim: int = 64,
        num_flows: int = 3
    ):
        super().__init__()
        
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
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Encode inputs
        encodings = self.encoder(inputs)
        
        # Apply normalizing flows
        transformed = {}
        log_det = {}
        
        for data_type, encoding in encodings.items():
            z = encoding
            log_det_sum = 0
            
            for flow in self.flows:
                z_new = flow(z)
                log_det_sum += torch.log(torch.abs(
                    torch.det(torch.autograd.functional.jacobian(flow, z))
                ))
                z = z_new
            
            transformed[data_type] = z
            log_det[data_type] = log_det_sum
        
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
    
    def detect_anomalies(
        self,
        inputs: Dict[str, torch.Tensor],
        thresholds: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """Detect anomalies in the input data"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            
            anomalies = {}
            for data_type, scores in outputs['anomaly_scores'].items():
                anomalies[data_type] = scores > thresholds[data_type]
            
            return anomalies

def train_anomaly_detector(
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    model_params: Dict = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[AnomalyDetector, Dict[str, List[float]]]:
    """Train the anomaly detection model"""
    # Initialize model
    if model_params is None:
        model_params = {
            'input_dims': {
                data_type: data.shape[-1]
                for data_type, data in train_data.items()
            },
            'hidden_size': 128,
            'latent_dim': 64,
            'num_flows': 3
        }
    
    model = AnomalyDetector(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    kld_loss = nn.KLDivLoss(reduction='batchmean')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_scores': [],
        'val_scores': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_scores = []
        
        # Training
        for i in range(0, len(next(iter(train_data.values()))), batch_size):
            # Prepare batch
            batch = {
                k: v[i:i+batch_size]
                for k, v in train_data.items()
            }
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate losses
            recon_loss = sum(
                mse_loss(outputs['reconstructions'][k], batch[k])
                for k in batch.keys()
            )
            
            # KL divergence loss for normalizing flows
            kl_loss = sum(
                -outputs['log_det'][k].mean()
                for k in batch.keys()
            )
            
            # Combined loss
            loss = recon_loss + 0.1 * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Calculate average anomaly scores
            avg_score = torch.mean(torch.cat([
                score.view(-1) for score in outputs['anomaly_scores'].values()
            ])).item()
            train_scores.append(avg_score)
        
        # Validation
        model.eval()
        val_losses = []
        val_scores = []
        
        with torch.no_grad():
            for i in range(0, len(next(iter(val_data.values()))), batch_size):
                # Prepare batch
                batch = {
                    k: v[i:i+batch_size]
                    for k, v in val_data.items()
                }
                
                # Forward pass
                outputs = model(batch)
                
                # Calculate losses
                recon_loss = sum(
                    mse_loss(outputs['reconstructions'][k], batch[k])
                    for k in batch.keys()
                )
                
                kl_loss = sum(
                    -outputs['log_det'][k].mean()
                    for k in batch.keys()
                )
                
                loss = recon_loss + 0.1 * kl_loss
                val_losses.append(loss.item())
                
                # Calculate average anomaly scores
                avg_score = torch.mean(torch.cat([
                    score.view(-1) for score in outputs['anomaly_scores'].values()
                ])).item()
                val_scores.append(avg_score)
        
        # Update history
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_scores'].append(np.mean(train_scores))
        history['val_scores'].append(np.mean(val_scores))
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"Val Loss: {history['val_loss'][-1]:.4f}")
            print(f"Train Scores: {history['train_scores'][-1]:.4f}")
            print(f"Val Scores: {history['val_scores'][-1]:.4f}")
    
    return model, history 