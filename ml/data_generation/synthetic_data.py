import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from dataclasses import dataclass
import random
from datetime import datetime, timedelta

@dataclass
class SpacecraftParameters:
    """Parameters for generating spacecraft telemetry data."""
    position_range: Tuple[float, float] = (-1000.0, 1000.0)
    velocity_range: Tuple[float, float] = (-10.0, 10.0)
    fuel_range: Tuple[float, float] = (0.0, 100.0)
    num_threat_classes: int = 5
    anomaly_probability: float = 0.1
    noise_std: float = 0.05

class SyntheticDataGenerator:
    def __init__(self):
        """Initialize the synthetic data generator."""
        self.threat_types = [
            "nominal",
            "collision_risk",
            "trajectory_anomaly",
            "system_failure",
            "unknown_threat"
        ]
    
    def generate_adversarial_data(self, num_samples: int) -> np.ndarray:
        """Generate synthetic adversarial data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Array of shape (num_samples, feature_dim) containing synthetic data
        """
        # Generate normal spacecraft behavior data
        normal_data = np.random.randn(num_samples // 2, 128)  # 128-dimensional feature space
        
        # Generate anomalous spacecraft behavior
        anomalous_data = np.random.randn(num_samples - num_samples // 2, 128) * 2 + 1
        
        # Combine and shuffle data
        data = np.vstack([normal_data, anomalous_data])
        np.random.shuffle(data)
        
        return data
    
    def generate_threat_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic threat detection data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (features, labels) arrays
        """
        # Generate feature vectors
        features = np.random.randn(num_samples, 64)  # 64-dimensional feature space
        
        # Generate threat labels (multi-class)
        labels = np.random.randint(0, len(self.threat_types), size=num_samples)
        
        return features, labels
    
    def generate_spacecraft_state(self) -> torch.Tensor:
        """Generate a synthetic spacecraft state.
        
        Returns:
            Tensor of shape (1, state_dim) containing the spacecraft state
        """
        # State includes position (3), velocity (3), attitude quaternion (4),
        # angular velocity (3), and battery level (1)
        state = np.zeros(14)
        
        # Position (x, y, z)
        state[0:3] = np.random.uniform(-100, 100, 3)
        
        # Velocity (vx, vy, vz)
        state[3:6] = np.random.uniform(-10, 10, 3)
        
        # Attitude quaternion (w, x, y, z)
        quat = self._generate_quaternions(1)[0]
        state[6:10] = quat
        
        # Angular velocity (wx, wy, wz)
        state[10:13] = np.random.uniform(-1, 1, 3)
        
        # Battery level (normalized)
        state[13] = np.random.uniform(0.2, 1.0)
        
        return torch.FloatTensor(state).unsqueeze(0)
    
    def step(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """Simulate a step in the environment.
        
        Args:
            state: Current state tensor
            action: Action tensor
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        # Convert tensors to numpy for easier manipulation
        state_np = state.detach().squeeze(0).numpy()
        action_np = action.detach().squeeze().numpy()
        
        # Update state based on action
        next_state = state_np.copy()
        
        # Update position based on velocity
        next_state[0:3] += next_state[3:6] * 0.1
        
        # Update velocity based on action
        next_state[3:6] += action_np[:3] * 0.1
        
        # Update attitude
        delta_quat = self._generate_quaternions(1)[0]
        next_state[6:10] = self._quaternion_multiply(next_state[6:10], delta_quat)
        
        # Update angular velocity based on action
        next_state[10:13] += action_np[3:6] * 0.01
        
        # Update battery level (slowly decreasing)
        next_state[13] = max(0.0, next_state[13] - 0.001)
        
        # Calculate reward based on state
        reward = self._compute_reward(next_state)
        
        # Check if episode is done
        done = next_state[13] < 0.2  # End if battery too low
        
        return torch.FloatTensor(next_state).unsqueeze(0), reward, done
    
    def _generate_quaternions(self, num_samples: int) -> np.ndarray:
        """Generate random quaternions.
        
        Args:
            num_samples: Number of quaternions to generate
            
        Returns:
            Array of shape (num_samples, 4) containing unit quaternions
        """
        quaternions = np.random.randn(num_samples, 4)
        return quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions.
        
        Args:
            q1: First quaternion (w, x, y, z)
            q2: Second quaternion (w, x, y, z)
            
        Returns:
            Result of quaternion multiplication
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([w, x, y, z])
    
    def _compute_reward(self, state: np.ndarray) -> float:
        """Compute reward based on state.
        
        Args:
            state: Current state array
            
        Returns:
            Reward value
        """
        # Penalize distance from origin
        position_penalty = -0.01 * np.linalg.norm(state[0:3])
        
        # Penalize high velocities
        velocity_penalty = -0.1 * np.linalg.norm(state[3:6])
        
        # Penalize high angular velocities
        angular_penalty = -0.1 * np.linalg.norm(state[10:13])
        
        # Reward high battery level
        battery_reward = state[13]
        
        return position_penalty + velocity_penalty + angular_penalty + battery_reward

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = SyntheticDataGenerator()
    
    # Generate spacecraft state data
    state_data = generator.generate_spacecraft_state(1000)
    print("\nSpacecraft State Data:")
    print(state_data.head())
    
    # Generate threat detection data
    threat_data, threat_labels = generator.generate_threat_data(1000)
    print("\nThreat Detection Data:")
    print(f"Data shape: {threat_data.shape}")
    print(f"Labels shape: {threat_labels.shape}")
    
    # Generate adversarial data
    adv_data = generator.generate_adversarial_data(1000)
    print("\nAdversarial Data:")
    print(f"Data shape: {adv_data.shape}")
    
    # Generate strategy data
    strategy_episodes = generator.generate_strategy_data(10)
    print("\nStrategy Generation Data:")
    print(f"Number of episodes: {len(strategy_episodes)}")
    print(f"First episode length: {len(strategy_episodes[0])}") 