import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class ConsensusNet(nn.Module):
    """Neural network for combining outputs from multiple evaluator models"""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """Initialize the consensus network.
        
        Args:
            input_dims: Dictionary mapping model names to their output dimensions
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_models = len(input_dims)
        
        # Feature projection for each model's output
        self.feature_projections = nn.ModuleDict({
            model_name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for model_name, dim in input_dims.items()
        })
        
        # Importance weighting for each model
        self.importance_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Cross-attention between model outputs
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Consensus processor
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.consensus_processor = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Decision head
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [normal, anomaly, threat]
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(
        self,
        model_features: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            model_features: List of feature tensors from each model
                Each tensor shape: (batch_size, seq_len, feature_dim)
            
        Returns:
            Dictionary containing consensus outputs
        """
        batch_size = model_features[0].shape[0]
        
        # Project features from each model
        projected_features = []
        for i, features in enumerate(model_features):
            projected = self.feature_projections[i](features)
            projected_features.append(projected)
        
        # Stack features
        stacked_features = torch.stack(projected_features, dim=1)  # [B, M, S, H]
        
        # Calculate importance weights
        importance_logits = self.importance_weights(
            torch.mean(stacked_features, dim=2)
        )  # [B, M, 1]
        importance_weights = torch.softmax(importance_logits, dim=1)
        
        # Apply cross-attention between models
        flattened_features = stacked_features.view(
            batch_size * self.num_models,
            -1,
            self.hidden_dim
        )
        attention_output, attention_weights = self.cross_attention(
            flattened_features,
            flattened_features,
            flattened_features
        )
        attention_output = attention_output.view(
            batch_size,
            self.num_models,
            -1,
            self.hidden_dim
        )
        
        # Weight features by importance
        weighted_features = torch.sum(
            attention_output * importance_weights.unsqueeze(-1).unsqueeze(-1),
            dim=1
        )
        
        # Process consensus features
        consensus_features = self.consensus_processor(weighted_features)
        
        # Get outputs
        confidence = torch.sigmoid(self.confidence_head(consensus_features.mean(dim=1)))
        decision = torch.softmax(self.decision_head(consensus_features.mean(dim=1)), dim=-1)
        
        return {
            "confidence": confidence,
            "decision": decision,
            "importance_weights": importance_weights.squeeze(-1),
            "attention_weights": attention_weights,
            "consensus_features": consensus_features
        }
    
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
                "decision": 0.4,
                "confidence": 0.3,
                "diversity": 0.3
            }
        
        # Decision loss (cross entropy)
        decision_loss = torch.nn.functional.cross_entropy(
            outputs["decision"],
            targets["decision"]
        )
        
        # Confidence loss (binary cross entropy)
        confidence_loss = torch.nn.functional.binary_cross_entropy(
            outputs["confidence"],
            targets["confidence"]
        )
        
        # Diversity loss (encourage diverse model usage)
        diversity_loss = -torch.mean(
            torch.sum(
                outputs["importance_weights"] * torch.log(outputs["importance_weights"] + 1e-10),
                dim=1
            )
        )
        
        # Combine losses
        total_loss = (
            weights["decision"] * decision_loss +
            weights["confidence"] * confidence_loss +
            weights["diversity"] * diversity_loss
        )
        
        return total_loss
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model state."""
        self.load_state_dict(torch.load(path)) 