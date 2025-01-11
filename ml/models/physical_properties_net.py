import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class PhysicalPropertiesNet(nn.Module):
    """Neural network for analyzing physical properties of space objects"""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """Initialize the physical properties network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer for temporal processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Object type classification head
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # [rocket_body, active_satellite, debris, cubesat]
        )
        
        # Mass and dimensions head
        self.mass_dim_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # [mass, length, width, height]
        )
        
        # Attitude dynamics head
        self.attitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7)  # [3 euler angles, 3 angular velocities, tumbling_prob]
        )
        
        # Material properties head
        self.material_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # [density, reflectivity, thermal_expansion, material_type]
        )
        
        # Thermal characteristics head
        self.thermal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [temperature, thermal_capacity, thermal_conductivity]
        )
        
        # Deployment state head
        self.deployment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # [folded, partial, deployed, deployment_anomaly]
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Dictionary containing physical property analysis outputs
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Process through transformer
        encoded = self.transformer(features, src_key_padding_mask=mask)
        
        # Global average pooling
        pooled = torch.mean(encoded, dim=1)
        
        # Get predictions from each head
        object_type = torch.softmax(self.type_head(pooled), dim=-1)
        mass_dim = self.mass_dim_head(pooled)
        attitude = self.attitude_head(pooled)
        material = self.material_head(pooled)
        thermal = self.thermal_head(pooled)
        deployment = torch.softmax(self.deployment_head(pooled), dim=-1)
        
        # Split predictions into components
        mass, length, width, height = torch.split(mass_dim, 1, dim=-1)
        euler_angles = attitude[:, :3]
        angular_velocities = attitude[:, 3:6]
        tumbling_prob = torch.sigmoid(attitude[:, 6])
        
        density, reflectivity, thermal_expansion, material_type = torch.split(material, 1, dim=-1)
        temperature, thermal_capacity, thermal_conductivity = torch.split(thermal, 1, dim=-1)
        
        return {
            "object_type": object_type,
            "mass": torch.relu(mass),
            "dimensions": torch.stack([torch.relu(length), torch.relu(width), torch.relu(height)], dim=-1),
            "euler_angles": euler_angles,
            "angular_velocities": angular_velocities,
            "tumbling_probability": tumbling_prob,
            "material_properties": {
                "density": torch.relu(density),
                "reflectivity": torch.sigmoid(reflectivity),
                "thermal_expansion": torch.relu(thermal_expansion),
                "material_type": torch.sigmoid(material_type)
            },
            "thermal_properties": {
                "temperature": temperature,
                "thermal_capacity": torch.relu(thermal_capacity),
                "thermal_conductivity": torch.relu(thermal_conductivity)
            },
            "deployment_state": deployment,
            "encoded_features": encoded
        }
    
    def predict(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Make predictions for input data.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Dictionary containing physical property predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, mask)
            return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Compute the weighted loss for training.
        
        Args:
            outputs: Dictionary of model outputs
            targets: Dictionary of target values
            weights: Optional dictionary of loss weights
            
        Returns:
            Total weighted loss
        """
        if weights is None:
            weights = {
                "type": 0.2,
                "physical": 0.2,
                "attitude": 0.2,
                "material": 0.2,
                "thermal": 0.1,
                "deployment": 0.1
            }
        
        # Object type classification loss
        type_loss = torch.nn.functional.cross_entropy(
            outputs["object_type"],
            targets["object_type"]
        )
        
        # Physical properties loss
        physical_loss = torch.mean(
            (outputs["mass"] - targets["mass"]) ** 2 +
            torch.mean((outputs["dimensions"] - targets["dimensions"]) ** 2, dim=-1)
        )
        
        # Attitude dynamics loss
        attitude_loss = torch.mean(
            torch.mean((outputs["euler_angles"] - targets["euler_angles"]) ** 2, dim=-1) +
            torch.mean((outputs["angular_velocities"] - targets["angular_velocities"]) ** 2, dim=-1) +
            (outputs["tumbling_probability"] - targets["tumbling_probability"]) ** 2
        )
        
        # Material properties loss
        material_loss = torch.mean(
            (outputs["material_properties"]["density"] - targets["material_properties"]["density"]) ** 2 +
            (outputs["material_properties"]["reflectivity"] - targets["material_properties"]["reflectivity"]) ** 2 +
            (outputs["material_properties"]["thermal_expansion"] - targets["material_properties"]["thermal_expansion"]) ** 2 +
            (outputs["material_properties"]["material_type"] - targets["material_properties"]["material_type"]) ** 2
        )
        
        # Thermal properties loss
        thermal_loss = torch.mean(
            (outputs["thermal_properties"]["temperature"] - targets["thermal_properties"]["temperature"]) ** 2 +
            (outputs["thermal_properties"]["thermal_capacity"] - targets["thermal_properties"]["thermal_capacity"]) ** 2 +
            (outputs["thermal_properties"]["thermal_conductivity"] - targets["thermal_properties"]["thermal_conductivity"]) ** 2
        )
        
        # Deployment state loss
        deployment_loss = torch.nn.functional.cross_entropy(
            outputs["deployment_state"],
            targets["deployment_state"]
        )
        
        # Combine losses
        total_loss = (
            weights["type"] * type_loss +
            weights["physical"] * physical_loss +
            weights["attitude"] * attitude_loss +
            weights["material"] * material_loss +
            weights["thermal"] * thermal_loss +
            weights["deployment"] * deployment_loss
        )
        
        return total_loss
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model state."""
        self.load_state_dict(torch.load(path)) 