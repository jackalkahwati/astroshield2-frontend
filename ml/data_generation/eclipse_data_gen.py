import numpy as np
import torch
from typing import Dict, List, Tuple

class EclipseDataGenerator:
    """Generator for synthetic eclipse data"""
    
    def __init__(
        self,
        sequence_length: int = 48,  # 48 hours
        feature_dim: int = 32,
        time_step: float = 60.0  # seconds
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.time_step = time_step
        
        # Orbital parameters
        self.orbit_types = {
            'LEO': {
                'period': 90 * 60,  # seconds
                'eclipse_duration': 30 * 60,  # seconds
                'temperature_range': (200, 300),  # Kelvin
                'power_range': (0.5, 1.0)  # Normalized power
            },
            'MEO': {
                'period': 6 * 3600,
                'eclipse_duration': 2 * 3600,
                'temperature_range': (150, 350),
                'power_range': (0.4, 1.0)
            },
            'GEO': {
                'period': 24 * 3600,
                'eclipse_duration': 1.2 * 3600,
                'temperature_range': (100, 400),
                'power_range': (0.3, 1.0)
            }
        }
    
    def generate_eclipse_sequence(
        self,
        orbit_type: str
    ) -> np.ndarray:
        """Generate eclipse state sequence"""
        params = self.orbit_types[orbit_type]
        
        # Time points
        time = np.arange(self.sequence_length) * self.time_step
        
        # Calculate eclipse phase
        phase = 2 * np.pi * time / params['period']
        eclipse_phase = np.pi * params['eclipse_duration'] / params['period']
        
        # Generate eclipse states (0: sunlight, 1: penumbra, 2: umbra)
        states = np.zeros((self.sequence_length, 3))
        
        for t in range(self.sequence_length):
            # Normalize phase to [0, 2π]
            current_phase = phase[t] % (2 * np.pi)
            
            # Check eclipse condition
            if abs(current_phase - np.pi) < eclipse_phase:
                # In eclipse region
                eclipse_progress = abs(current_phase - np.pi) / eclipse_phase
                
                if eclipse_progress < 0.2:  # Penumbra entry
                    states[t, 1] = 1.0  # Penumbra
                elif eclipse_progress > 0.8:  # Penumbra exit
                    states[t, 1] = 1.0  # Penumbra
                else:  # Umbra
                    states[t, 2] = 1.0  # Umbra
            else:
                states[t, 0] = 1.0  # Sunlight
        
        return states
    
    def generate_thermal_sequence(
        self,
        eclipse_states: np.ndarray,
        orbit_type: str
    ) -> np.ndarray:
        """Generate thermal sequence based on eclipse states"""
        params = self.orbit_types[orbit_type]
        
        # Initialize temperature
        temperature = np.zeros(self.sequence_length)
        temperature[0] = np.random.uniform(*params['temperature_range'])
        
        # Thermal time constants
        cooling_rate = 0.1  # K/s in eclipse
        heating_rate = 0.2  # K/s in sunlight
        
        # Generate temperature profile
        for t in range(1, self.sequence_length):
            if eclipse_states[t, 0] > 0:  # In sunlight
                # Heat up
                target_temp = params['temperature_range'][1]
                delta_t = (target_temp - temperature[t-1]) * heating_rate
            else:  # In eclipse
                # Cool down
                target_temp = params['temperature_range'][0]
                delta_t = (target_temp - temperature[t-1]) * cooling_rate
            
            # Update temperature with noise
            temperature[t] = temperature[t-1] + delta_t * self.time_step
            temperature[t] += np.random.normal(0, 1)
        
        return temperature
    
    def generate_power_sequence(
        self,
        eclipse_states: np.ndarray,
        orbit_type: str
    ) -> np.ndarray:
        """Generate power state sequence based on eclipse states"""
        params = self.orbit_types[orbit_type]
        
        # Initialize power states
        power = np.zeros((self.sequence_length, 2))  # [battery_state, power_consumption]
        power[0, 0] = 1.0  # Start with full battery
        
        # Power parameters
        charge_rate = 0.02  # Per second in sunlight
        discharge_rate = 0.01  # Per second in eclipse
        
        # Generate power profile
        for t in range(1, self.sequence_length):
            # Update battery state
            if eclipse_states[t, 0] > 0:  # In sunlight
                # Charge battery
                power[t, 0] = min(
                    1.0,
                    power[t-1, 0] + charge_rate * self.time_step
                )
            else:  # In eclipse
                # Discharge battery
                power[t, 0] = max(
                    0.0,
                    power[t-1, 0] - discharge_rate * self.time_step
                )
            
            # Generate power consumption
            base_consumption = np.random.uniform(*params['power_range'])
            if eclipse_states[t, 2] > 0:  # In umbra
                # Higher power consumption for heating
                base_consumption *= 1.5
            
            power[t, 1] = base_consumption * (1 + np.random.normal(0, 0.05))
        
        return power
    
    def generate_features(
        self,
        eclipse_states: np.ndarray,
        temperature: np.ndarray,
        power: np.ndarray
    ) -> np.ndarray:
        """Generate feature vector"""
        # Combine basic features
        basic_features = np.concatenate([
            eclipse_states,
            temperature[:, np.newaxis],
            power
        ], axis=1)
        
        # Calculate derived features
        eclipse_duration = np.sum(eclipse_states[:, 1:], axis=1)
        temperature_rate = np.gradient(temperature, self.time_step)
        power_rate = np.gradient(power[:, 0], self.time_step)
        
        derived_features = np.stack([
            eclipse_duration,
            temperature_rate,
            power_rate
        ], axis=1)
        
        # Combine all features
        features = np.concatenate([basic_features, derived_features], axis=1)
        
        # Add random features to reach feature_dim
        if features.shape[1] < self.feature_dim:
            extra_features = np.random.normal(
                0, 0.1,
                (len(features), self.feature_dim - features.shape[1])
            )
            features = np.concatenate([features, extra_features], axis=1)
        
        return features
    
    def generate_data(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic eclipse data"""
        features_list = []
        labels_list = []
        
        for _ in range(num_samples):
            # Select random orbit type
            orbit_type = np.random.choice(list(self.orbit_types.keys()))
            
            # Generate sequences
            eclipse_states = self.generate_eclipse_sequence(orbit_type)
            temperature = self.generate_thermal_sequence(eclipse_states, orbit_type)
            power = self.generate_power_sequence(eclipse_states, orbit_type)
            
            # Generate features
            features = self.generate_features(eclipse_states, temperature, power)
            features_list.append(features)
            
            # Generate labels
            labels = {
                'eclipse_state': eclipse_states[-1],  # Final eclipse state
                'temperature': temperature[-1],  # Final temperature
                'power_state': power[-1]  # Final power state
            }
            labels_list.append(np.concatenate([
                labels['eclipse_state'],
                [labels['temperature']],
                labels['power_state']
            ]))
        
        # Convert to tensors
        features = torch.tensor(np.stack(features_list), dtype=torch.float32)
        labels = torch.tensor(np.stack(labels_list), dtype=torch.float32)
        
        return {
            'features': features,
            'labels': labels
        } 