import torch
import torch.nn as nn
from typing import Tuple, Dict

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 14, action_dim: int = 6, hidden_dim: int = 128):
        """Initialize the actor-critic model.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Actor network (outputs mean action)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bound actions between -1 and 1
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Tuple of (action, value) tensors
        """
        # Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Get mean action from actor
        action_mean = self.actor(state)
        
        # Add exploration noise during training
        if self.training:
            noise = torch.randn_like(action_mean) * 0.1
            action = torch.clamp(action_mean + noise, -1.0, 1.0)
        else:
            action = action_mean
        
        # Get state value from critic
        value = self.critic(state)
        
        return action, value
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model state."""
        self.load_state_dict(torch.load(path)) 