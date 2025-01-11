import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class StimulationNet(nn.Module):
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """Initialize the stimulation network model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional GRU for sequence processing
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Response prediction heads
        self.immediate_response_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # x, y, z acceleration response
        )
        
        self.delayed_response_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # x, y, z delayed response
        )
        
        self.stability_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Stability score
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
    def apply_attention(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention mechanism to sequence.
        
        Args:
            sequence: Input sequence of shape (batch_size, seq_len, hidden_dim * 2)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Calculate attention scores
        attention_weights = self.attention(sequence)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention to sequence
        context = torch.sum(attention_weights * sequence, dim=1)
        return context, attention_weights
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Optional initial hidden state
            
        Returns:
            Dictionary containing stimulation response predictions
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_proj(x)
        
        # Process sequence
        gru_output, hidden_state = self.gru(x, hidden)
        
        # Apply attention
        context, attention_weights = self.apply_attention(gru_output)
        
        # Get predictions
        immediate_response = self.immediate_response_head(context)
        delayed_response = self.delayed_response_head(context)
        stability_score = self.stability_head(context)
        
        return {
            "immediate_response": immediate_response,  # Acceleration response
            "delayed_response": delayed_response,      # Delayed motion response
            "stability_score": stability_score,        # Overall stability assessment
            "attention_weights": attention_weights,
            "hidden_state": hidden_state,
            "context_vector": context
        }
    
    def predict_response(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict stimulation response for input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing response predictions and analysis
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            predictions = {
                "acceleration_response": outputs["immediate_response"],
                "motion_response": outputs["delayed_response"],
                "is_stable": outputs["stability_score"] > 0.5,
                "stability_confidence": outputs["stability_score"],
                "critical_timepoints": outputs["attention_weights"].squeeze(-1),
                "response_encoding": outputs["context_vector"]
            }
            
            return predictions
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model state."""
        self.load_state_dict(torch.load(path)) 