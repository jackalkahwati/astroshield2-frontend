import numpy as np
import torch
from typing import Dict, List, Tuple

class ProximityDataGenerator:
    """Generator for synthetic conjunction data"""
    
    def __init__(
        self,
        min_range: float = 0.1,  # km
        max_range: float = 1000.0,  # km
        min_velocity: float = 1.0,  # km/s
        max_velocity: float = 10.0,  # km/s
        time_steps: int = 48,  # 48 hours
        feature_dim: int = 32
    ):
        self.min_range = min_range
        self.max_range = max_range
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.time_steps = time_steps
        self.feature_dim = feature_dim
    
    def generate_trajectory(self) -> np.ndarray:
        """Generate a single conjunction trajectory"""
        # Initial state
        initial_range = np.random.uniform(self.min_range, self.max_range)
        initial_velocity = np.random.uniform(self.min_velocity, self.max_velocity)
        
        # Generate time points
        time = np.linspace(0, self.time_steps, self.time_steps)
        
        # Add random perturbations to make it more realistic
        perturbations = np.random.normal(0, 0.1, self.time_steps)
        
        # Calculate ranges
        ranges = initial_range + initial_velocity * time + perturbations
        
        # Ensure ranges are positive
        ranges = np.maximum(ranges, self.min_range)
        
        return ranges
    
    def generate_features(self, trajectory: np.ndarray) -> np.ndarray:
        """Generate feature vector for a trajectory"""
        # Basic features from trajectory
        mean_range = np.mean(trajectory)
        min_range = np.min(trajectory)
        max_range = np.max(trajectory)
        std_range = np.std(trajectory)
        
        # Calculate velocities
        velocities = np.diff(trajectory)
        mean_velocity = np.mean(velocities)
        max_velocity = np.max(np.abs(velocities))
        
        # Calculate accelerations
        accelerations = np.diff(velocities)
        mean_acceleration = np.mean(accelerations)
        max_acceleration = np.max(np.abs(accelerations))
        
        # Basic feature vector
        basic_features = np.array([
            mean_range,
            min_range,
            max_range,
            std_range,
            mean_velocity,
            max_velocity,
            mean_acceleration,
            max_acceleration
        ])
        
        # Generate additional random features to simulate other parameters
        additional_features = np.random.normal(0, 1, self.feature_dim - len(basic_features))
        
        # Combine features
        features = np.concatenate([basic_features, additional_features])
        
        return features
    
    def generate_labels(self, trajectory: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate labels for a trajectory"""
        min_range = np.min(trajectory)
        
        # Calculate collision probability based on minimum range
        collision_prob = np.exp(-min_range / 10)  # Higher prob for smaller ranges
        
        # Determine severity level (0: Low, 1: Medium, 2: High, 3: Critical)
        if min_range > 100:
            severity = 0  # Low
        elif min_range > 50:
            severity = 1  # Medium
        elif min_range > 10:
            severity = 2  # High
        else:
            severity = 3  # Critical
        
        # One-hot encode severity
        severity_onehot = np.zeros(4)
        severity_onehot[severity] = 1
        
        return {
            'range': min_range,
            'probability': collision_prob,
            'severity': severity_onehot
        }
    
    def generate_data(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic conjunction data"""
        features_list = []
        labels_list = []
        
        for _ in range(num_samples):
            # Generate trajectory
            trajectory = self.generate_trajectory()
            
            # Generate features
            features = self.generate_features(trajectory)
            features_list.append(features)
            
            # Generate labels
            labels = self.generate_labels(trajectory)
            labels_list.append([
                labels['range'],
                labels['probability'],
                *labels['severity']
            ])
        
        # Convert to tensors
        features = torch.tensor(features_list, dtype=torch.float32)
        labels = torch.tensor(labels_list, dtype=torch.float32)
        
        return {
            'features': features,
            'labels': labels
        } 