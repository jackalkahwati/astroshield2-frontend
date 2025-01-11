import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicalVAE(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, latent_dim=32):
        """
        Variational Autoencoder for physical characteristics analysis
        Args:
            input_dim: Number of input features (AMR, size, etc.)
            hidden_dim: Size of hidden layers
            latent_dim: Dimension of latent space
        """
        super(PhysicalVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Thresholds
        self.amr_threshold = 0.15  # 15% deviation
        self.confidence_threshold = 0.95
        self.change_threshold = 0.2  # 20% change

    def encode(self, x):
        """
        Encode input into latent space
        """
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent vector
        """
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def analyze_amr(self, amr_data, historical_amr=None):
        """
        Analyze Area-to-Mass Ratio
        Args:
            amr_data: Current AMR measurements
            historical_amr: Optional historical AMR data
        Returns:
            anomaly_detected: Boolean
            confidence: Float between 0 and 1
            change_magnitude: Float (percentage change)
        """
        self.eval()
        with torch.no_grad():
            # Encode current AMR
            current_mu, current_log_var = self.encode(amr_data)
            
            # Calculate reconstruction
            reconstructed = self.decode(self.reparameterize(current_mu, current_log_var))
            
            # Calculate reconstruction error
            reconstruction_error = F.mse_loss(reconstructed, amr_data)
            
            if historical_amr is not None:
                # Calculate change from historical data
                historical_mu, _ = self.encode(historical_amr)
                change_magnitude = torch.norm(current_mu - historical_mu) / torch.norm(historical_mu)
                change_magnitude = change_magnitude.item()
            else:
                change_magnitude = 0.0
            
            # Calculate confidence
            confidence = 1.0 - torch.sigmoid(reconstruction_error).item()
            
            # Detect anomaly
            anomaly_detected = (confidence < self.confidence_threshold or 
                              change_magnitude > self.amr_threshold)

        return anomaly_detected, confidence, change_magnitude

    def detect_subsatellite(self, primary_data, secondary_data):
        """
        Detect sub-satellite deployment
        Args:
            primary_data: Physical characteristics of primary object
            secondary_data: Physical characteristics of potential sub-satellite
        Returns:
            deployment_detected: Boolean
            confidence: Float between 0 and 1
        """
        self.eval()
        with torch.no_grad():
            # Get latent representations
            primary_mu, _ = self.encode(primary_data)
            secondary_mu, _ = self.encode(secondary_data)
            
            # Calculate similarity in latent space
            similarity = F.cosine_similarity(primary_mu, secondary_mu)
            
            # Calculate relative size ratio
            size_ratio = torch.norm(secondary_data) / torch.norm(primary_data)
            
            # Combined confidence score
            confidence = similarity.item() * (1.0 - torch.abs(size_ratio - 0.5))
            
            # Detect deployment
            deployment_detected = confidence > self.confidence_threshold

        return deployment_detected, confidence

    def track_physical_changes(self, sequence_data):
        """
        Track physical characteristic changes over time
        Args:
            sequence_data: Sequence of physical measurements
        Returns:
            changes_detected: Boolean
            change_points: List of indices where changes occurred
            confidence: Float between 0 and 1
        """
        self.eval()
        with torch.no_grad():
            # Get sequence of latent representations
            latent_sequence = []
            for data in sequence_data:
                mu, _ = self.encode(data)
                latent_sequence.append(mu)
            
            # Stack latent vectors
            latent_sequence = torch.stack(latent_sequence)
            
            # Calculate changes between consecutive measurements
            changes = torch.diff(latent_sequence, dim=0)
            change_magnitudes = torch.norm(changes, dim=1)
            
            # Detect significant changes
            change_points = torch.where(change_magnitudes > self.change_threshold)[0]
            
            # Calculate overall confidence
            confidence = torch.mean(torch.sigmoid(-change_magnitudes)).item()
            
            # Detect if any significant changes occurred
            changes_detected = len(change_points) > 0

        return changes_detected, change_points.tolist(), confidence

    def validate_physics(self, physical_data):
        """
        Validate physical characteristics against known constraints
        Args:
            physical_data: Tensor of physical measurements [mass, density, dimensions, thermal_params]
        Returns:
            is_valid: Boolean indicating if physics are valid
            violations: List of specific violations found
            confidence: Float between 0 and 1 indicating confidence in validation
        """
        self.eval()
        with torch.no_grad():
            violations = []
            
            # Extract key parameters (assuming standardized input format)
            mass = physical_data[0].item()
            density = physical_data[1].item()
            dimensions = physical_data[2:5]  # x, y, z
            thermal_conductivity = physical_data[5].item()
            
            # Basic physics validation checks
            if mass <= 0 or mass > 1000:  # kg
                violations.append("Mass out of valid range")
                
            if density <= 0 or density > 8000:  # kg/m³
                violations.append("Density outside physical limits")
                
            # Volume from dimensions
            volume = torch.prod(dimensions).item()
            calculated_density = mass / volume if volume > 0 else float('inf')
            
            if abs(calculated_density - density) / density > 0.1:  # 10% tolerance
                violations.append("Mass-density-volume relationship violated")
                
            # Thermal parameter validation
            if thermal_conductivity < 0.1 or thermal_conductivity > 500:  # W/(m·K)
                violations.append("Thermal conductivity outside physical limits")
            
            # Calculate confidence based on number and severity of violations
            confidence = 1.0 - (len(violations) * 0.2)  # Each violation reduces confidence by 20%
            confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            
            is_valid = len(violations) == 0

        return is_valid, violations, confidence

    def generate_training_data(self, num_samples=1000, anomaly_ratio=0.1):
        """
        Generate synthetic training data with physical characteristics
        Args:
            num_samples: Number of samples to generate
            anomaly_ratio: Ratio of anomalous samples to generate
        Returns:
            data: Generated training data (num_samples x input_dim)
            labels: Binary labels (0: normal, 1: anomalous)
            metrics: Dictionary of data generation metrics
        """
        normal_samples = int(num_samples * (1 - anomaly_ratio))
        anomaly_samples = num_samples - normal_samples
        
        data = []
        labels = []
        metrics = {
            'valid_samples': 0,
            'invalid_samples': 0,
            'avg_confidence': 0.0
        }
        
        # Generate normal samples
        for _ in range(normal_samples):
            # Generate physically valid parameters
            mass = np.random.uniform(1, 500)  # kg
            density = np.random.uniform(1000, 5000)  # kg/m³
            
            # Calculate volume and dimensions
            volume = mass / density
            dimension = np.cbrt(volume)
            dimensions = np.random.uniform(0.8, 1.2, 3) * dimension
            
            # Generate thermal parameters
            thermal_conductivity = np.random.uniform(1, 400)  # W/(m·K)
            
            # Combine parameters
            sample = torch.tensor([
                mass,
                density,
                *dimensions,
                thermal_conductivity,
                *np.random.uniform(0, 1, 4)  # Additional features
            ], dtype=torch.float32)
            
            # Validate the generated sample
            is_valid, _, confidence = self.validate_physics(sample)
            
            if is_valid:
                metrics['valid_samples'] += 1
                metrics['avg_confidence'] += confidence
            else:
                metrics['invalid_samples'] += 1
            
            data.append(sample)
            labels.append(0)  # Normal sample
        
        # Generate anomalous samples
        for _ in range(anomaly_samples):
            # Generate physically invalid parameters
            mass = np.random.choice([-1, 0, 1500])  # Invalid mass
            density = np.random.choice([-100, 0, 10000])  # Invalid density
            dimensions = np.random.uniform(0.1, 5, 3)  # Unrealistic dimensions
            thermal_conductivity = np.random.choice([-10, 0, 1000])  # Invalid thermal conductivity
            
            # Combine parameters
            sample = torch.tensor([
                mass,
                density,
                *dimensions,
                thermal_conductivity,
                *np.random.uniform(0, 1, 4)  # Additional features
            ], dtype=torch.float32)
            
            data.append(sample)
            labels.append(1)  # Anomalous sample
        
        # Convert to tensors
        data = torch.stack(data)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Calculate final metrics
        metrics['avg_confidence'] /= metrics['valid_samples'] if metrics['valid_samples'] > 0 else 1
        metrics['total_samples'] = num_samples
        metrics['anomaly_ratio'] = anomaly_samples / num_samples
        
        return data, labels, metrics

if __name__ == "__main__":
    # Initialize model
    model = PhysicalVAE(input_dim=10)
    print("Model initialized")
    
    # Generate training data
    data, labels, metrics = model.generate_training_data(num_samples=100)
    print("\nData Generation Metrics:")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Valid Samples: {metrics['valid_samples']}")
    print(f"Invalid Samples: {metrics['invalid_samples']}")
    print(f"Average Confidence: {metrics['avg_confidence']:.3f}")
    
    # Test physics validation
    print("\nTesting Physics Validation:")
    
    # Test valid sample
    valid_sample = torch.tensor([
        100.0,  # mass (kg)
        2000.0,  # density (kg/m³)
        1.0, 1.0, 1.0,  # dimensions (m)
        200.0,  # thermal conductivity
        0.5, 0.5, 0.5, 0.5  # additional features
    ], dtype=torch.float32)
    
    is_valid, violations, confidence = model.validate_physics(valid_sample)
    print("\nValid Sample Test:")
    print(f"Is Valid: {is_valid}")
    print(f"Violations: {violations}")
    print(f"Confidence: {confidence:.3f}")
    
    # Test invalid sample
    invalid_sample = torch.tensor([
        -1.0,  # invalid mass
        -100.0,  # invalid density
        0.1, 0.1, 0.1,  # unrealistic dimensions
        -10.0,  # invalid thermal conductivity
        0.5, 0.5, 0.5, 0.5  # additional features
    ], dtype=torch.float32)
    
    is_valid, violations, confidence = model.validate_physics(invalid_sample)
    print("\nInvalid Sample Test:")
    print(f"Is Valid: {is_valid}")
    print(f"Violations: {violations}")
    print(f"Confidence: {confidence:.3f}")
    
    # Test anomaly detection
    print("\nTesting Anomaly Detection:")
    anomaly_detected, confidence, change = model.analyze_amr(invalid_sample)
    print(f"Anomaly Detected: {anomaly_detected}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Change Magnitude: {change:.3f}")
