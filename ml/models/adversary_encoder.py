import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any

class AdversaryEncoder(nn.Module):
    def __init__(self, input_dim: int = 128, encoding_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, encoding_dim)
        )
        
        # Decoder network for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim)
        )
        
        # Anomaly detection thresholds
        self.reconstruction_threshold = None
        self.encoding_threshold = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through both encoder and decoder.
        During training, returns only the decoded output for computing reconstruction loss.
        Use encode_behavior() for getting encodings and detect_anomalies() for anomaly detection.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  # Return only decoded output for training
        
    def get_encoding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get both encoded and decoded representations."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
        
    def encode_behavior(self, telemetry_data: torch.Tensor) -> torch.Tensor:
        """Convert telemetry data to behavior encoding."""
        self.eval()
        with torch.no_grad():
            encoded, _ = self.get_encoding(telemetry_data)
        return encoded
        
    def detect_anomalies(self, telemetry_data: torch.Tensor) -> Dict[str, Any]:
        """Identify unusual patterns in the telemetry data."""
        self.eval()
        with torch.no_grad():
            encoded, decoded = self.get_encoding(telemetry_data)
            
            # Compute reconstruction error
            reconstruction_error = torch.mean((telemetry_data - decoded) ** 2, dim=1)
            
            # Compute encoding statistics
            encoding_norm = torch.norm(encoded, dim=1)
            
            # Detect anomalies based on thresholds
            is_reconstruction_anomaly = reconstruction_error > self.reconstruction_threshold if self.reconstruction_threshold else torch.zeros_like(reconstruction_error, dtype=torch.bool)
            is_encoding_anomaly = encoding_norm > self.encoding_threshold if self.encoding_threshold else torch.zeros_like(encoding_norm, dtype=torch.bool)
            
            return {
                "is_anomaly": is_reconstruction_anomaly | is_encoding_anomaly,
                "reconstruction_error": reconstruction_error,
                "encoding_norm": encoding_norm,
                "encoded_representation": encoded
            }
    
    def fit(self, training_data: torch.Tensor, epochs: int = 100, batch_size: int = 32, learning_rate: float = 1e-3):
        """Train the encoder-decoder and set anomaly detection thresholds."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(training_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                x = batch[0]
                decoded = self.forward(x)  # Use forward for training
                loss = criterion(decoded, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
        
        # Compute thresholds based on training data
        with torch.no_grad():
            encoded, decoded = self.get_encoding(training_data)
            reconstruction_errors = torch.mean((training_data - decoded) ** 2, dim=1)
            encoding_norms = torch.norm(encoded, dim=1)
            
            # Set thresholds at 95th percentile
            self.reconstruction_threshold = torch.quantile(reconstruction_errors, 0.95)
            self.encoding_threshold = torch.quantile(encoding_norms, 0.95) 