from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

@dataclass
class Experience:
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool

class StrategyGenerator(nn.Module):
    def __init__(self, state_dim: int = 14, action_dim: int = 6, hidden_dim: int = 128):
        """Initialize the strategy generator model.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Policy network (outputs mean action)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bound actions between -1 and 1
        )
        
        # Value network
        self.value_net = nn.Sequential(
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
            Tuple of (action_mean, value) tensors
        """
        # Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        action_mean = self.policy_net(state)
        value = self.value_net(state)
        return action_mean, value
    
    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Select action using the policy network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Tuple of (action, info_dict) where info_dict contains additional information
        """
        self.eval()
        with torch.no_grad():
            # Ensure state has batch dimension
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            action_mean, value = self.forward(state)
            # Add small noise for exploration
            noise = torch.randn_like(action_mean) * 0.1
            action = torch.clamp(action_mean + noise, -1.0, 1.0)
            
            return action, {
                "value": value,
                "action_mean": action_mean
            }
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model state."""
        self.load_state_dict(torch.load(path)) 