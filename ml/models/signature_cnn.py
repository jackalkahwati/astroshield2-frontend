import torch
import torch.nn as nn
import torch.nn.functional as F

class SignatureCNNAutoencoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        """
        CNN Autoencoder for signature analysis
        Args:
            input_channels: Number of input channels (RF, optical, radar)
            latent_dim: Dimension of latent space
        """
        super(SignatureCNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        # Thresholds
        self.anomaly_threshold = 0.1
        self.signature_confidence_threshold = 0.9

    def encode(self, x):
        """
        Encode input into latent space
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for VAE
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent vector
        """
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def analyze_signature(self, signature_data):
        """
        Analyze signature for anomalies
        Args:
            signature_data: Multi-channel signature data (RF, optical, radar)
        Returns:
            anomaly_detected: Boolean
            confidence: Float between 0 and 1
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            reconstructed, mu, log_var = self.forward(signature_data)
            
            # Calculate reconstruction error
            reconstruction_loss = F.mse_loss(reconstructed, signature_data)
            
            # Calculate KL divergence
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss
            total_loss = reconstruction_loss + kl_loss
            
            # Calculate anomaly score
            anomaly_score = total_loss.item()
            
            # Calculate confidence
            confidence = 1.0 - torch.sigmoid(torch.tensor(anomaly_score)).item()
            
            # Detect anomaly
            anomaly_detected = anomaly_score > self.anomaly_threshold

        return anomaly_detected, confidence

    def cross_sensor_correlation(self, optical_data, radar_data):
        """
        Correlate optical and radar signatures
        Args:
            optical_data: Optical signature data
            radar_data: Radar signature data
        Returns:
            mismatch_detected: Boolean
            correlation_score: Float between 0 and 1
        """
        self.eval()
        with torch.no_grad():
            # Get latent representations
            optical_mu, _ = self.encode(optical_data)
            radar_mu, _ = self.encode(radar_data)
            
            # Calculate correlation score
            correlation_score = F.cosine_similarity(optical_mu, radar_mu)
            
            # Detect mismatch
            mismatch_detected = correlation_score < self.signature_confidence_threshold

        return mismatch_detected, correlation_score.item()

    def analyze_rf_pattern(self, rf_sequence):
        """
        Analyze RF emission patterns
        Args:
            rf_sequence: Sequence of RF signature data
        Returns:
            pattern_change: Boolean
            confidence: Float between 0 and 1
        """
        self.eval()
        with torch.no_grad():
            # Get sequence of latent representations
            latent_sequence = []
            for rf_data in rf_sequence:
                mu, _ = self.encode(rf_data)
                latent_sequence.append(mu)
            
            # Stack latent vectors
            latent_sequence = torch.stack(latent_sequence)
            
            # Calculate temporal consistency
            temporal_diff = torch.diff(latent_sequence, dim=0)
            consistency_score = torch.mean(torch.norm(temporal_diff, dim=1))
            
            # Calculate confidence
            confidence = torch.sigmoid(-consistency_score).item()
            
            # Detect pattern change
            pattern_change = confidence < self.signature_confidence_threshold

        return pattern_change, confidence
