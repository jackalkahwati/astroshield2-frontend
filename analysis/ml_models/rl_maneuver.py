import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ManeuverAction:
    delta_v: List[float]  # [dx, dy, dz] velocity changes
    execution_time: datetime
    duration: float
    fuel_required: float

@dataclass
class ManeuverResult:
    success: bool
    action: ManeuverAction
    expected_reward: float
    state_value: float
    confidence: float
    metrics: Dict[str, float]

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Action standard deviation (learnable)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        features = self.feature_net(state)
        
        # Actor output
        action_mean = self.actor_mean(features)
        action_std = self.actor_logstd.exp()
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value

    def get_action(self, state, deterministic=False):
        action_mean, action_std, value = self(state)
        
        if deterministic:
            return action_mean
        
        # Sample action using reparameterization trick
        normal = torch.distributions.Normal(action_mean, action_std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value

class RLManeuverPlanner:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'max_delta_v': 2.0,
            'time_step': 1.0,
            'initial_fuel': 100,
            'fuel_consumption_rate': 0.1,
            'min_safe_distance': 100,
            'state_dim': 12,  # position, velocity, target, obstacles
            'action_dim': 3,  # delta_v in x,y,z
            'hidden_dim': 256,
            'clip_ratio': 0.2,
            'learning_rate': 3e-4
        }
        
        self.model = ActorCritic(
            state_dim=self.config['state_dim'],
            action_dim=self.config['action_dim'],
            hidden_dim=self.config['hidden_dim']
        )
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )

    def _preprocess_state(self, current_state: Dict, threats: List[Dict]) -> torch.Tensor:
        """Convert state information to model input format"""
        state = []
        
        # Current position and velocity
        pos = current_state.get('position', {'x': 0, 'y': 0, 'z': 0})
        vel = current_state.get('velocity', {'vx': 0, 'vy': 0, 'vz': 0})
        state.extend([pos['x'], pos['y'], pos['z']])
        state.extend([vel['vx'], vel['vy'], vel['vz']])
        
        # Target position (if available)
        target = current_state.get('target', {'x': 0, 'y': 0, 'z': 0})
        state.extend([target['x'], target['y'], target['z']])
        
        # Threat information (simplified to nearest threat)
        if threats:
            nearest = threats[0]
            threat_pos = nearest.get('position', {'x': 0, 'y': 0, 'z': 0})
            state.extend([threat_pos['x'], threat_pos['y'], threat_pos['z']])
        else:
            state.extend([0, 0, 0])
        
        return torch.FloatTensor(state)

    def _compute_metrics(self, action: torch.Tensor, state: torch.Tensor) -> Dict[str, float]:
        """Compute performance metrics for the maneuver"""
        delta_v = action.numpy()
        delta_v_magnitude = np.linalg.norm(delta_v)
        
        # Calculate fuel efficiency (simplified)
        fuel_efficiency = max(0, 1 - delta_v_magnitude / self.config['max_delta_v'])
        
        # Calculate safety margin (simplified)
        threat_pos = state[-3:].numpy()
        current_pos = state[:3].numpy()
        distance_to_threat = np.linalg.norm(threat_pos - current_pos)
        safety_margin = max(0, distance_to_threat - self.config['min_safe_distance'])
        
        return {
            'delta_v_magnitude': float(delta_v_magnitude),
            'fuel_efficiency': float(fuel_efficiency),
            'safety_margin': float(safety_margin),
            'execution_efficiency': float(0.9)  # placeholder
        }

    async def plan_maneuver(
        self,
        current_state: Dict,
        threats: List[Dict]
    ) -> ManeuverResult:
        """
        Plan maneuver using the trained PPO model
        """
        try:
            # Preprocess state
            state = self._preprocess_state(current_state, threats)
            
            # Get action from model
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(state)
            
            # Convert action to delta_v
            delta_v = action.numpy().tolist()
            
            # Create maneuver action
            maneuver = ManeuverAction(
                delta_v=delta_v,
                execution_time=datetime.now(),
                duration=10.0,  # placeholder
                fuel_required=np.linalg.norm(delta_v) * self.config['fuel_consumption_rate']
            )
            
            # Compute metrics
            metrics = self._compute_metrics(action, state)
            
            return ManeuverResult(
                success=True,
                action=maneuver,
                expected_reward=float(value.item()),
                state_value=float(value.item()),
                confidence=float(torch.sigmoid(log_prob).item()),
                metrics=metrics
            )

        except Exception as e:
            print(f"Error in maneuver planning: {str(e)}")
            return None

    def export_to_onnx(self, path: str):
        """Export model to ONNX format"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, self.config['state_dim'])
            
            # Export the model
            torch.onnx.export(
                self.model,
                dummy_input,
                path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['state'],
                output_names=['action_mean', 'action_std', 'value'],
                dynamic_axes={
                    'state': {0: 'batch_size'},
                    'action_mean': {0: 'batch_size'},
                    'action_std': {0: 'batch_size'},
                    'value': {0: 'batch_size'}
                }
            )
            
            print(f"Model exported to: {path}")
            
            # Verify the exported model
            import onnx
            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model check passed")
            
        except Exception as e:
            print(f"Error exporting model: {str(e)}")
