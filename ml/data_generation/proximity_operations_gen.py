import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy.stats import multivariate_normal

class ProximityOperationsGenerator:
    """Generator for synthetic proximity operations data"""
    
    def __init__(
        self,
        sequence_length: int = 100,
        feature_dim: int = 32,
        time_step: float = 60.0  # seconds
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.time_step = time_step
        
        # Define operation types and their characteristics
        self.operation_types = {
            'approach': {
                'min_range': 0.1,  # km
                'max_range': 100.0,  # km
                'velocity_range': (0.001, 0.1),  # km/s
                'duration_range': (10, 30),  # timesteps
                'maneuver_probability': 0.8
            },
            'inspection': {
                'min_range': 0.5,  # km
                'max_range': 5.0,  # km
                'velocity_range': (0.005, 0.05),
                'duration_range': (20, 40),
                'maneuver_probability': 0.9
            },
            'docking': {
                'min_range': 0.0,  # km
                'max_range': 0.1,  # km
                'velocity_range': (0.001, 0.01),
                'duration_range': (30, 50),
                'maneuver_probability': 1.0
            },
            'avoidance': {
                'min_range': 1.0,  # km
                'max_range': 50.0,  # km
                'velocity_range': (0.01, 0.2),
                'duration_range': (5, 15),
                'maneuver_probability': 0.7
            }
        }
        
        # Define maneuver characteristics
        self.maneuver_types = {
            'hohmann': {
                'delta_v_range': (0.01, 0.1),  # km/s
                'duration_range': (5, 15)  # timesteps
            },
            'continuous_thrust': {
                'acceleration_range': (0.0001, 0.001),  # km/s²
                'duration_range': (10, 30)
            },
            'impulsive': {
                'delta_v_range': (0.005, 0.05),
                'duration_range': (1, 3)
            }
        }
    
    def generate_relative_motion(
        self,
        operation_type: str,
        duration: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Generate relative motion trajectory with maneuvers"""
        params = self.operation_types[operation_type]
        
        # Initialize state vector [x, y, z, vx, vy, vz]
        initial_range = np.random.uniform(
            params['min_range'],
            params['max_range']
        )
        initial_velocity = np.random.uniform(
            *params['velocity_range']
        )
        
        # Random initial position
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        position = direction * initial_range
        
        # Random initial velocity
        vel_direction = np.random.randn(3)
        vel_direction /= np.linalg.norm(vel_direction)
        velocity = vel_direction * initial_velocity
        
        state = np.concatenate([position, velocity])
        
        # Generate trajectory
        states = [state]
        maneuvers = []
        
        for t in range(1, duration):
            # Check for maneuver
            if np.random.random() < params['maneuver_probability'] / duration:
                # Select random maneuver type
                maneuver_type = np.random.choice(list(self.maneuver_types.keys()))
                maneuver_params = self.maneuver_types[maneuver_type]
                
                if maneuver_type == 'hohmann':
                    # Hohmann transfer maneuver
                    delta_v = np.random.uniform(*maneuver_params['delta_v_range'])
                    duration = np.random.randint(*maneuver_params['duration_range'])
                    
                    # Calculate transfer direction
                    target_pos = position * np.random.uniform(0.5, 2.0)
                    transfer_dir = target_pos - position
                    transfer_dir /= np.linalg.norm(transfer_dir)
                    
                    # Apply delta-v
                    velocity += transfer_dir * delta_v
                    
                    maneuvers.append({
                        'type': maneuver_type,
                        'time': t,
                        'delta_v': delta_v,
                        'duration': duration
                    })
                
                elif maneuver_type == 'continuous_thrust':
                    # Continuous thrust maneuver
                    acceleration = np.random.uniform(
                        *maneuver_params['acceleration_range']
                    )
                    duration = np.random.randint(*maneuver_params['duration_range'])
                    
                    # Random thrust direction
                    thrust_dir = np.random.randn(3)
                    thrust_dir /= np.linalg.norm(thrust_dir)
                    
                    # Apply acceleration
                    velocity += thrust_dir * acceleration * self.time_step
                    
                    maneuvers.append({
                        'type': maneuver_type,
                        'time': t,
                        'acceleration': acceleration,
                        'duration': duration
                    })
                
                else:  # impulsive
                    # Impulsive maneuver
                    delta_v = np.random.uniform(*maneuver_params['delta_v_range'])
                    
                    # Random impulse direction
                    impulse_dir = np.random.randn(3)
                    impulse_dir /= np.linalg.norm(impulse_dir)
                    
                    # Apply impulse
                    velocity += impulse_dir * delta_v
                    
                    maneuvers.append({
                        'type': maneuver_type,
                        'time': t,
                        'delta_v': delta_v
                    })
            
            # Update state with CW equations (simplified)
            position += velocity * self.time_step
            
            # Add process noise
            noise_pos = np.random.normal(0, 0.001, 3)  # 1m position noise
            noise_vel = np.random.normal(0, 0.0001, 3)  # 0.1m/s velocity noise
            
            position += noise_pos
            velocity += noise_vel
            
            # Store new state
            state = np.concatenate([position, velocity])
            states.append(state)
        
        return np.array(states), maneuvers
    
    def generate_features(
        self,
        states: np.ndarray,
        maneuvers: List[Dict]
    ) -> np.ndarray:
        """Generate feature vector from states and maneuvers"""
        num_timesteps = len(states)
        features = np.zeros((num_timesteps, self.feature_dim))
        
        for t in range(num_timesteps):
            feature_idx = 0
            
            # Basic kinematic features
            position = states[t, :3]
            velocity = states[t, 3:]
            
            # Range and range-rate
            range_val = np.linalg.norm(position)
            range_rate = np.dot(position, velocity) / range_val
            
            # Relative motion features
            features[t, feature_idx:feature_idx+2] = [range_val, range_rate]
            feature_idx += 2
            
            # Position and velocity
            features[t, feature_idx:feature_idx+6] = states[t]
            feature_idx += 6
            
            # Maneuver indicators
            maneuver_features = np.zeros(len(self.maneuver_types))
            for maneuver in maneuvers:
                if maneuver['time'] == t:
                    maneuver_idx = list(self.maneuver_types.keys()).index(
                        maneuver['type']
                    )
                    maneuver_features[maneuver_idx] = 1.0
            
            features[t, feature_idx:feature_idx+len(self.maneuver_types)] = maneuver_features
            feature_idx += len(self.maneuver_types)
            
            # Derived features
            if t > 0:
                # Acceleration
                accel = (velocity - states[t-1, 3:]) / self.time_step
                features[t, feature_idx:feature_idx+3] = accel
                feature_idx += 3
                
                # Jerk
                if t > 1:
                    prev_accel = (states[t-1, 3:] - states[t-2, 3:]) / self.time_step
                    jerk = (accel - prev_accel) / self.time_step
                    features[t, feature_idx:feature_idx+3] = jerk
                    feature_idx += 3
            
            # Fill remaining features with zeros
            if feature_idx < self.feature_dim:
                features[t, feature_idx:] = 0.0
        
        return features
    
    def generate_labels(
        self,
        states: np.ndarray,
        maneuvers: List[Dict],
        operation_type: str
    ) -> np.ndarray:
        """Generate labels for the sequence"""
        # Calculate minimum range
        min_range = min(np.linalg.norm(state[:3]) for state in states)
        
        # Calculate maximum velocity
        max_velocity = max(np.linalg.norm(state[3:]) for state in states)
        
        # Count maneuvers
        num_maneuvers = len(maneuvers)
        
        # One-hot encode operation type
        operation_types = list(self.operation_types.keys())
        operation_onehot = np.zeros(len(operation_types))
        operation_onehot[operation_types.index(operation_type)] = 1.0
        
        # Combine labels
        labels = np.concatenate([
            [min_range],
            [max_velocity],
            [num_maneuvers],
            operation_onehot
        ])
        
        return labels
    
    def generate_data(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic proximity operations data"""
        features_list = []
        labels_list = []
        
        for _ in range(num_samples):
            # Select random operation type
            operation_type = np.random.choice(list(self.operation_types.keys()))
            
            # Generate trajectory
            states, maneuvers = self.generate_relative_motion(
                operation_type,
                self.sequence_length
            )
            
            # Generate features
            features = self.generate_features(states, maneuvers)
            features_list.append(features)
            
            # Generate labels
            labels = self.generate_labels(states, maneuvers, operation_type)
            labels_list.append(labels)
        
        # Convert to tensors
        features = torch.tensor(np.stack(features_list), dtype=torch.float32)
        labels = torch.tensor(np.stack(labels_list), dtype=torch.float32)
        
        return {
            'features': features,
            'labels': labels
        } 