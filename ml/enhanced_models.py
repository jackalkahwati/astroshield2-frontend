"""Enhanced ML models for CCDM analysis using unsupervised deep learning."""
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Generate decoder input sequence
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input)
        reconstruction = self.output_layer(decoder_output)
        
        return reconstruction

class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, input_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_var = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

class ManeuverDetectionModel:
    """Unsupervised maneuver detection using LSTM Autoencoder."""
    
    def __init__(self, sequence_length: int = 50, feature_dim: int = 6):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # LSTM Autoencoder for trajectory reconstruction
        self.model = LSTMAutoencoder(
            input_size=feature_dim,
            hidden_size=32
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def prepare_sequence_data(self, trajectory_data: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare sequential trajectory data."""
        features = []
        for point in trajectory_data:
            pos = point.get('position', {})
            vel = point.get('velocity', {})
            features.append([
                pos.get('x', 0), pos.get('y', 0), pos.get('z', 0),
                vel.get('vx', 0), vel.get('vy', 0), vel.get('vz', 0)
            ])
        return np.array(features)
    
    def fit(self, trajectory_sequences: List[List[Dict[str, Any]]], epochs=10):
        """Train the autoencoder on normal behavior."""
        self.model.train()
        X = np.vstack([self.prepare_sequence_data(seq) for seq in trajectory_sequences])
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            reconstruction = self.model(X_tensor)
            loss = F.mse_loss(reconstruction, X_tensor)
            loss.backward()
            self.optimizer.step()
    
    def predict(self, trajectory_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Predict maneuver probabilities using reconstruction error."""
        self.model.eval()
        X = self.prepare_sequence_data([trajectory_data])
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            reconstruction = self.model(X_tensor)
            reconstruction_error = F.mse_loss(reconstruction, X_tensor, reduction='none')
            error_scores = reconstruction_error.mean(dim=-1).cpu().numpy()
        
        # Normalize error scores to probabilities
        max_error = np.max(error_scores)
        normalized_scores = error_scores / max_error if max_error > 0 else error_scores
        avg_score = float(np.mean(normalized_scores))
        
        return {
            'no_maneuver': 1 - avg_score,
            'subtle_maneuver': avg_score if avg_score < 0.7 else 0.0,
            'significant_maneuver': avg_score if avg_score >= 0.7 else 0.0,
            'anomaly_score': avg_score
        }

class SignatureAnalysisModel:
    """Unsupervised signature analysis using Convolutional Autoencoder."""
    
    def __init__(self, input_shape: Tuple[int, int] = (100, 6)):
        self.input_shape = input_shape
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convolutional Autoencoder for signature patterns
        self.model = ConvAutoencoder(input_channels=input_shape[1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def prepare_signature_data(self, signature_data: Dict[str, Any]) -> np.ndarray:
        """Prepare signature data."""
        optical = signature_data.get('optical', {})
        radar = signature_data.get('radar', {})
        
        features = []
        for t in range(self.input_shape[0]):
            features.append([
                optical.get(f'intensity_{t}', 0),
                optical.get(f'spectrum_{t}', 0),
                optical.get(f'polarization_{t}', 0),
                radar.get(f'rcs_{t}', 0),
                radar.get(f'doppler_{t}', 0),
                radar.get(f'phase_{t}', 0)
            ])
        return np.array(features)
    
    def fit(self, signature_data: List[Dict[str, Any]], epochs=10):
        """Train the autoencoder on normal signatures."""
        self.model.train()
        X = np.vstack([self.prepare_signature_data(sig) for sig in signature_data])
        X_scaled = self.scaler.fit_transform(X.reshape(-1, self.input_shape[1])).reshape(X.shape)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            reconstruction = self.model(X_tensor)
            loss = F.mse_loss(reconstruction, X_tensor)
            loss.backward()
            self.optimizer.step()
    
    def predict(self, signature_data: Dict[str, Any]) -> Dict[str, float]:
        """Predict signature classification using reconstruction error."""
        self.model.eval()
        X = self.prepare_signature_data(signature_data)
        X_scaled = self.scaler.transform(X.reshape(-1, self.input_shape[1])).reshape(X.shape)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            reconstruction = self.model(X_tensor)
            reconstruction_error = F.mse_loss(reconstruction, X_tensor, reduction='none')
            error_scores = reconstruction_error.mean(dim=-1).cpu().numpy()
        
        avg_error = float(np.mean(error_scores))
        max_error = float(np.max(error_scores))
        
        return {
            'normal': 1 - avg_error,
            'anomalous': avg_error,
            'mismatched': max_error if max_error > avg_error else 0.0,
            'unknown': float(np.std(error_scores)),
            'anomaly_score': avg_error
        }

class AMRAnalysisModel:
    """Unsupervised AMR analysis using Variational Autoencoder."""
    
    def __init__(self, input_dim: int = 9, latent_dim: int = 3):
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Variational Autoencoder for AMR patterns
        self.model = VariationalAutoencoder(input_dim, latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def prepare_features(self, amr_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from AMR data."""
        features = []
        # Basic AMR features
        features.extend([
            amr_data.get('amr_value', 0),
            amr_data.get('solar_pressure', 0),
            amr_data.get('drag_coefficient', 0)
        ])
        
        # Derived features
        if 'historical_values' in amr_data:
            hist = amr_data['historical_values']
            features.extend([
                np.mean(hist),
                np.std(hist),
                np.max(hist),
                np.min(hist)
            ])
        else:
            features.extend([0, 0, 0, 0])
            
        # Environmental features
        env = amr_data.get('environment', {})
        features.extend([
            env.get('solar_flux', 0),
            env.get('geomagnetic_activity', 0)
        ])
        
        return np.array(features).reshape(1, -1)
    
    def fit(self, amr_data: List[Dict[str, Any]], epochs=10):
        """Train the VAE on normal AMR patterns."""
        self.model.train()
        X = np.vstack([self.prepare_features(data) for data in amr_data])
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            reconstruction, mu, log_var = self.model(X_tensor)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstruction, X_tensor)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss
            loss = recon_loss + 0.1 * kl_loss
            loss.backward()
            self.optimizer.step()
    
    def predict(self, amr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict AMR characteristics using VAE reconstruction."""
        self.model.eval()
        X = self.prepare_features(amr_data)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            reconstruction, mu, log_var = self.model(X_tensor)
            reconstruction_error = F.mse_loss(reconstruction, X_tensor, reduction='none')
            error_score = float(reconstruction_error.mean().cpu().numpy())
            
            # Calculate KL divergence
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_score = float(kl_div.cpu().numpy())
        
        # Normalize scores
        total_score = (error_score + 0.1 * kl_score) / 1.1
        
        return {
            'classification': {
                'normal': 1 - total_score,
                'anomalous': total_score
            },
            'anomaly_score': total_score,
            'reconstruction_error': error_score,
            'kl_divergence': kl_score,
            'confidence': float(1 - np.sqrt(error_score))
        }
