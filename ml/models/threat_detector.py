import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple

class ThreatDetectorNN(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_classes: int = 5):
        """Initialize the threat detector model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_classes: Number of threat classes
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, num_classes) containing class logits
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions for input data.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
            
            return {
                "predictions": predictions,
                "probabilities": probs,
                "confidence": confidence
            }
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model state."""
        self.load_state_dict(torch.load(path)) 