import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy.stats import multivariate_normal

class TrackDataGenerator:
    """Generator for synthetic track data"""
    
    def __init__(
        self,
        sequence_length: int = 100,
        feature_dim: int = 32,
        time_step: float = 60.0  # seconds
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.time_step = time_step
        
        # Track types and their characteristics
        self.track_types = {
            'debris': {
                'velocity_range': (1, 10),  # km/s
                'size_range': (0.01, 0.5),  # meters
                'maneuver_prob': 0.0,
                'noise_std': 0.5
            },
            'active_satellite': {
                'velocity_range': (3, 8),
                'size_range': (1, 10),
                'maneuver_prob': 0.3,
                'noise_std': 0.2
            },
            'rocket_body': {
                'velocity_range': (2, 12),
                'size_range': (2, 20),
                'maneuver_prob': 0.1,
                'noise_std': 0.3
            },
            'unknown': {
                'velocity_range': (1, 15),
                'size_range': (0.1, 5),
                'maneuver_prob': 0.2,
                'noise_std': 0.8
            }
        }
        
        # Sensor characteristics
        self.sensor_types = {
            'radar': {
                'range_noise': 0.1,  # km
                'velocity_noise': 0.05,  # km/s
                'angle_noise': 0.02,  # radians
                'detection_prob': 0.8
            },
            'optical': {
                'angle_noise': 0.01,
                'magnitude_noise': 0.5,
                'detection_prob': 0.7
            }
        }
    
    def generate_initial_state(
        self,
        track_type: str
    ) -> np.ndarray:
        """Generate initial state vector [x, y, z, vx, vy, vz]"""
        params = self.track_types[track_type]
        
        # Random position in LEO (simplified model)
        r = np.random.uniform(6800, 7200)  # km from Earth center
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(-np.pi/2, np.pi/2)
        
        # Convert to Cartesian coordinates
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        
        # Random velocity
        v_mag = np.random.uniform(*params['velocity_range'])
        v_theta = np.random.uniform(0, 2*np.pi)
        v_phi = np.random.uniform(-np.pi/2, np.pi/2)
        
        vx = v_mag * np.cos(v_phi) * np.cos(v_theta)
        vy = v_mag * np.cos(v_phi) * np.sin(v_theta)
        vz = v_mag * np.sin(v_phi)
        
        return np.array([x, y, z, vx, vy, vz])
    
    def propagate_state(
        self,
        state: np.ndarray,
        track_type: str,
        dt: float
    ) -> np.ndarray:
        """Propagate state vector using simple orbital mechanics"""
        params = self.track_types[track_type]
        
        # Extract position and velocity
        r = state[:3]
        v = state[3:]
        
        # Calculate acceleration due to gravity (simplified)
        r_mag = np.linalg.norm(r)
        a_grav = -398600.0 * r / r_mag**3  # GM = 398600 km³/s²
        
        # Add maneuver if applicable
        if np.random.random() < params['maneuver_prob']:
            # Random delta-v maneuver
            dv_mag = np.random.uniform(0.01, 0.1)  # km/s
            dv_dir = np.random.randn(3)
            dv_dir /= np.linalg.norm(dv_dir)
            a_maneuver = dv_dir * dv_mag / dt
        else:
            a_maneuver = np.zeros(3)
        
        # Total acceleration
        a_total = a_grav + a_maneuver
        
        # Update state
        new_r = r + v * dt + 0.5 * a_total * dt**2
        new_v = v + a_total * dt
        
        # Add process noise
        noise_std = params['noise_std']
        noise_r = np.random.normal(0, noise_std, 3)
        noise_v = np.random.normal(0, noise_std/10, 3)
        
        new_state = np.concatenate([
            new_r + noise_r,
            new_v + noise_v
        ])
        
        return new_state
    
    def generate_measurement(
        self,
        state: np.ndarray,
        sensor_type: str
    ) -> Dict[str, np.ndarray]:
        """Generate sensor measurement from true state"""
        params = self.sensor_types[sensor_type]
        
        if np.random.random() > params['detection_prob']:
            return None
        
        if sensor_type == 'radar':
            # Extract position and velocity
            r = state[:3]
            v = state[3:]
            
            # Calculate range, azimuth, elevation
            range_val = np.linalg.norm(r)
            az = np.arctan2(r[1], r[0])
            el = np.arcsin(r[2] / range_val)
            
            # Calculate range rate
            range_rate = np.dot(r, v) / range_val
            
            # Add measurement noise
            range_val += np.random.normal(0, params['range_noise'])
            az += np.random.normal(0, params['angle_noise'])
            el += np.random.normal(0, params['angle_noise'])
            range_rate += np.random.normal(0, params['velocity_noise'])
            
            return {
                'range': range_val,
                'azimuth': az,
                'elevation': el,
                'range_rate': range_rate
            }
        else:  # optical
            # Calculate angular position
            r = state[:3]
            range_val = np.linalg.norm(r)
            az = np.arctan2(r[1], r[0])
            el = np.arcsin(r[2] / range_val)
            
            # Calculate visual magnitude (simplified model)
            base_magnitude = 10.0  # Base visual magnitude
            magnitude = base_magnitude + 5 * np.log10(range_val/1000)
            
            # Add measurement noise
            az += np.random.normal(0, params['angle_noise'])
            el += np.random.normal(0, params['angle_noise'])
            magnitude += np.random.normal(0, params['magnitude_noise'])
            
            return {
                'azimuth': az,
                'elevation': el,
                'magnitude': magnitude
            }
    
    def generate_track_sequence(
        self,
        track_type: str
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Generate sequence of states and measurements"""
        # Generate initial state
        state = self.generate_initial_state(track_type)
        states = [state]
        measurements = []
        
        # Generate sequence
        for t in range(1, self.sequence_length):
            # Propagate state
            state = self.propagate_state(state, track_type, self.time_step)
            states.append(state)
            
            # Generate measurements from both sensor types
            meas_t = {}
            for sensor in self.sensor_types:
                meas = self.generate_measurement(state, sensor)
                if meas is not None:
                    meas_t[sensor] = meas
            measurements.append(meas_t)
        
        return np.array(states), measurements
    
    def generate_features(
        self,
        states: np.ndarray,
        measurements: List[Dict]
    ) -> np.ndarray:
        """Generate feature vector from states and measurements"""
        num_timesteps = len(states)
        features = np.zeros((num_timesteps, self.feature_dim))
        
        for t in range(num_timesteps):
            # Basic kinematic features
            features[t, 0:6] = states[t]  # Position and velocity
            
            # Derived kinematic features
            r = states[t, :3]
            v = states[t, 3:]
            
            r_mag = np.linalg.norm(r)
            v_mag = np.linalg.norm(v)
            specific_angular_momentum = np.cross(r, v)
            h_mag = np.linalg.norm(specific_angular_momentum)
            
            features[t, 6:9] = [r_mag, v_mag, h_mag]
            
            # Measurement features
            if t > 0:
                meas = measurements[t-1]
                feature_idx = 9
                
                for sensor_type, sensor_meas in meas.items():
                    if sensor_type == 'radar':
                        features[t, feature_idx:feature_idx+4] = [
                            sensor_meas['range'],
                            sensor_meas['azimuth'],
                            sensor_meas['elevation'],
                            sensor_meas['range_rate']
                        ]
                        feature_idx += 4
                    else:  # optical
                        features[t, feature_idx:feature_idx+3] = [
                            sensor_meas['azimuth'],
                            sensor_meas['elevation'],
                            sensor_meas['magnitude']
                        ]
                        feature_idx += 3
        
        return features
    
    def generate_data(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic track data"""
        features_list = []
        labels_list = []
        
        for _ in range(num_samples):
            # Select random track type
            track_type = np.random.choice(list(self.track_types.keys()))
            
            # Generate track sequence
            states, measurements = self.generate_track_sequence(track_type)
            
            # Generate features
            features = self.generate_features(states, measurements)
            features_list.append(features)
            
            # Generate labels (track type one-hot encoding)
            track_types = list(self.track_types.keys())
            label = np.zeros(len(track_types))
            label[track_types.index(track_type)] = 1.0
            labels_list.append(label)
        
        # Convert to tensors
        features = torch.tensor(np.stack(features_list), dtype=torch.float32)
        labels = torch.tensor(np.stack(labels_list), dtype=torch.float32)
        
        return {
            'features': features,
            'labels': labels
        }
    
    def generate_training_data(
        self,
        num_samples: int = 10000,
        sequence_length: int = 48
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for track evaluation.
        
        Args:
            num_samples: Number of samples to generate
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (features, labels) where:
            - features shape: (num_samples, sequence_length, 64)
            - labels shape: (num_samples, 20) [state(6), radar(5), optical(4), class(4), conf(1)]
        """
        # Initialize arrays with correct dimensions
        X = np.zeros((num_samples, sequence_length, 64))  # Updated to match expected dimension
        y = np.zeros((num_samples, 20))  # [state(6), radar(5), optical(4), class(4), conf(1)]
        
        for i in range(num_samples):
            try:
                # Select random track type
                track_type = np.random.choice(list(self.track_types.keys()))
                
                # Generate initial state
                state = self.generate_initial_state(track_type)
                
                # Generate sequence
                for t in range(sequence_length):
                    # Propagate state
                    state = self.propagate_state(state, track_type, self.time_step)
                    
                    # Generate sensor measurements
                    radar_meas = self._generate_radar_measurement(state)  # Returns 5 values
                    optical_meas = self._generate_optical_measurement(state)  # Returns 4 values
                    
                    # Calculate additional features
                    r = state[:3]
                    v = state[3:]
                    r_mag = np.linalg.norm(r)
                    v_mag = np.linalg.norm(v)
                    h = np.cross(r, v)
                    h_mag = np.linalg.norm(h)
                    
                    # Combine into feature vector with padding to reach 64 dimensions
                    features = np.concatenate([
                        state,  # 6 dimensions
                        radar_meas,  # 5 dimensions
                        optical_meas,  # 4 dimensions
                        np.array([r_mag, v_mag, h_mag]),  # 3 dimensions
                        np.array([float(track_type == t) for t in self.track_types.keys()]),  # 4 dimensions
                        np.zeros(42)  # Padding to reach 64 dimensions
                    ])
                    
                    X[i, t] = features
                
                # Generate labels
                track_class = np.zeros(4)  # One-hot encoding
                track_class[list(self.track_types.keys()).index(track_type)] = 1
                
                # Final state and measurements for labels
                final_radar = self._generate_radar_measurement(state)
                final_optical = self._generate_optical_measurement(state)
                
                # Combine all labels
                y[i] = np.concatenate([
                    state,  # 6: position and velocity
                    final_radar,  # 5: radar measurements
                    final_optical,  # 4: optical measurements
                    track_class,  # 4: classification
                    np.array([0.9])  # 1: confidence
                ])
                
            except Exception as e:
                print(f"Error generating sample {i}: {str(e)}")
                continue
        
        return X, y
    
    def _generate_radar_measurement(self, state: np.ndarray) -> np.ndarray:
        """Generate radar measurements with noise"""
        params = self.sensor_types['radar']
        
        # Extract position and velocity
        pos = state[:3]
        vel = state[3:]
        
        # Add measurement noise
        noisy_pos = pos + np.random.normal(0, params['range_noise'], 3)
        
        # Convert to radar coordinates (range, azimuth, elevation, range_rate)
        r = np.linalg.norm(noisy_pos)
        az = np.arctan2(noisy_pos[1], noisy_pos[0])
        el = np.arcsin(noisy_pos[2] / r)
        
        # Calculate range rate
        range_rate = np.dot(pos, vel) / r
        range_rate += np.random.normal(0, params['velocity_noise'])
        
        # Add angle noise
        az += np.random.normal(0, params['angle_noise'])
        el += np.random.normal(0, params['angle_noise'])
        
        # Return 5 measurements: range, azimuth, elevation, range_rate, SNR
        snr = 20 * np.log10(1000 / r)  # Simple SNR model
        return np.array([r, az, el, range_rate, snr])
    
    def _generate_optical_measurement(self, state: np.ndarray) -> np.ndarray:
        """Generate optical measurements with noise"""
        params = self.sensor_types['optical']
        
        # Extract position
        pos = state[:3]
        
        # Convert to optical coordinates (right ascension, declination)
        r = np.linalg.norm(pos)
        ra = np.arctan2(pos[1], pos[0])
        dec = np.arcsin(pos[2] / r)
        
        # Add angle noise
        ra += np.random.normal(0, params['angle_noise'])
        dec += np.random.normal(0, params['angle_noise'])
        
        # Generate visual magnitude (simplified model)
        mag = 10 * np.log10(r) + np.random.normal(0, params['magnitude_noise'])
        
        # Calculate signal-to-noise ratio
        snr = 20 - mag  # Simple SNR model
        
        # Return 4 measurements: right ascension, declination, magnitude, SNR
        return np.array([ra, dec, mag, snr]) 