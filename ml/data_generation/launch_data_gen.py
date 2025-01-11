import numpy as np
from typing import Tuple, Dict, List
import datetime
from scipy.integrate import solve_ivp

class LaunchVehicle:
    """Launch vehicle configuration with realistic reliability modeling"""
    def __init__(self, vehicle_type: str):
        # Base configurations
        if vehicle_type == 'heavy':
            self.stages = 3
            self.mass_fractions = [0.85, 0.12, 0.03]
            self.thrust_profiles = [(3000000, 300), (1000000, 400), (200000, 500)]  # (thrust N, Isp s)
            self.base_reliability = 0.98  # Increased from 0.95
            self.total_mass = 10000.0
            self.max_payload = 25000  # kg
            self.max_wind_tolerance = 20  # m/s
        elif vehicle_type == 'medium':
            self.stages = 2
            self.mass_fractions = [0.88, 0.12]
            self.thrust_profiles = [(2000000, 280), (500000, 450)]
            self.base_reliability = 0.99  # Increased from 0.97
            self.total_mass = 5000.0
            self.max_payload = 10000  # kg
            self.max_wind_tolerance = 15  # m/s
        else:  # small
            self.stages = 2
            self.mass_fractions = [0.90, 0.10]
            self.thrust_profiles = [(1000000, 270), (200000, 420)]
            self.base_reliability = 0.995  # Increased from 0.98
            self.total_mass = 1000.0
            self.max_payload = 2000  # kg
            self.max_wind_tolerance = 12  # m/s
        
        # Updated failure modes and probabilities
        self.failure_modes = {
            'engine': 0.35,      # Reduced from 0.40
            'structure': 0.15,   # Unchanged
            'guidance': 0.15,    # Unchanged
            'separation': 0.15,  # Reduced from 0.20
            'other': 0.10       # Unchanged
        }
        
        # Updated stage-specific reliability factors
        self.stage_reliability = [0.99, 0.98, 0.97]  # Increased reliability
        
        # Updated mission phase reliability factors
        self.phase_reliability = {
            'prelaunch': 0.998,  # Increased from 0.995
            'liftoff': 0.995,    # Increased from 0.99
            'maxq': 0.99,        # Increased from 0.98
            'staging': 0.98,     # Increased from 0.97
            'orbit_insertion': 0.99  # Increased from 0.98
        }

class LaunchDataGenerator:
    """Generate synthetic data for launch event evaluation with physics-based success probabilities"""
    
    def __init__(self):
        # Constants
        self.G = 6.67430e-11  # Gravitational constant
        self.M = 5.972e24     # Earth mass
        self.R = 6371000      # Earth radius
        
        # Launch sites with updated characteristics
        self.launch_sites = {
            'KSC': {
                'latitude': 28.5729,
                'longitude': -80.6490,
                'altitude': 0,
                'weather_reliability': 0.98,  # Increased from 0.95
                'infrastructure_reliability': 0.995,  # Increased from 0.99
                'seasonal_factors': {
                    'winter': 0.95,  # Increased from 0.90
                    'spring': 0.98,  # Increased from 0.95
                    'summer': 0.90,  # Increased from 0.85
                    'fall': 0.98     # Increased from 0.95
                }
            }
        }
        
        # Updated weather impact factors
        self.weather_factors = {
            'wind_speed': {'weight': 0.35, 'threshold': 25.0},  # Increased threshold
            'visibility': {'weight': 0.15, 'threshold': 4000},  # Reduced threshold
            'temperature': {'weight': 0.15, 'range': [-15, 40]},  # Expanded range
            'precipitation': {'weight': 0.15, 'threshold': 3.0}   # Increased threshold
        }

    def _generate_nominal_trajectory(
        self,
        vehicle: LaunchVehicle,
        target: Dict,
        sequence_length: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Generate nominal launch trajectory with improved physics"""
        trajectory = np.zeros((sequence_length, 6))  # [x, y, z, vx, vy, vz]
        events = []
        
        # Initial conditions
        altitude = 0.0
        velocity = 0.0
        angle = np.pi/2  # Vertical launch
        
        # Stage timing with improved distribution
        stage_durations = [int(sequence_length * f * 1.1) for f in vehicle.mass_fractions]  # 10% longer
        current_stage = 0
        stage_start = 0
        
        for t in range(sequence_length):
            # Stage transition check
            if t - stage_start >= stage_durations[current_stage] and current_stage < vehicle.stages - 1:
                events.append({
                    'type': 'stage_separation',
                    'time': t,
                    'stage': current_stage
                })
                current_stage += 1
                stage_start = t
            
            # Current stage parameters
            thrust, isp = vehicle.thrust_profiles[current_stage]
            mass = vehicle.total_mass * sum(vehicle.mass_fractions[current_stage:])
            
            # Improved gravity turn profile
            if altitude < 2000:  # Increased from 1000
                # Extended vertical ascent
                angle = np.pi/2
            elif altitude < 60000:  # Increased from 50000
                # Smoother gravity turn
                angle = np.pi/2 - np.pi/4 * (altitude - 2000)/58000
            else:
                # Optimized final trajectory angle
                angle = np.pi/4 - np.pi/6 * min((altitude - 60000)/140000, 1.0)
            
            # Thrust vector
            thrust_x = thrust * np.cos(angle)
            thrust_y = thrust * np.sin(angle)
            
            # Gravitational acceleration
            r = self.R + altitude
            g = self.G * self.M / (r * r)
            
            # Improved atmospheric effects
            if altitude < 100000:  # Below Karman line
                density = 1.225 * np.exp(-altitude/7400)
                drag_coeff = 0.15 + 0.15 * min(velocity/1000, 1.0)  # Reduced from 0.2
                area = np.pi * (4.0 ** 2)  # Approximate vehicle cross-section
                drag = 0.5 * density * velocity * velocity * drag_coeff * area
            else:
                drag = 0.0
            
            # Acceleration components
            ax = (thrust_x - drag * np.cos(angle)) / mass
            ay = (thrust_y - drag * np.sin(angle) - g) / mass
            
            # Update velocity
            dt = 1.0  # Normalized time step
            velocity_x = trajectory[t-1, 3] + ax * dt if t > 0 else ax * dt
            velocity_y = trajectory[t-1, 4] + ay * dt if t > 0 else ay * dt
            velocity = np.sqrt(velocity_x**2 + velocity_y**2)
            
            # Update position
            x = trajectory[t-1, 0] + velocity_x * dt if t > 0 else 0
            y = trajectory[t-1, 1] + velocity_y * dt if t > 0 else 0
            altitude = np.sqrt(x**2 + y**2)
            
            # Store state
            trajectory[t] = [x, y, 0, velocity_x, velocity_y, 0]
            
            # Check for mission events
            if altitude > target['altitude']:
                events.append({
                    'type': 'orbit_insertion',
                    'time': t,
                    'altitude': altitude
                })
                break
        
        return trajectory, events

    def _add_anomalies(
        self,
        trajectory: np.ndarray,
        events: List[Dict],
        anomaly_probability: float = 0.15  # Reduced from 0.2
    ) -> Tuple[np.ndarray, List[Dict], List[float]]:
        """Add realistic anomalies to the trajectory with reduced severity"""
        anomalies = []
        modified_trajectory = trajectory.copy()
        anomaly_probabilities = np.zeros(10)  # Store probabilities for each anomaly type
        
        if np.random.random() < anomaly_probability:
            # Select random anomaly time
            anomaly_time = np.random.randint(1, len(trajectory)-1)
            
            # Select anomaly type with bias towards less severe types
            anomaly_types = [
                'engine_failure', 'engine_underperform', 'structure_stress',
                'structure_damage', 'guidance_drift', 'guidance_loss',
                'separation_delay', 'separation_failure', 'debris_minor', 'debris_major'
            ]
            anomaly_type = np.random.choice(anomaly_types)
            anomaly_idx = anomaly_types.index(anomaly_type)
            
            # Reduced anomaly parameters
            severity = np.random.uniform(0.05, 0.25)  # Reduced from (0.05, 0.3)
            duration = np.random.randint(1, 3)  # Reduced from (1, 3)
            
            # Set probability for the selected anomaly type
            anomaly_probabilities[anomaly_idx] = severity
            
            # Apply anomaly effects with reduced impact
            if 'engine' in anomaly_type:
                # Reduced thrust reduction
                for t in range(anomaly_time, min(anomaly_time + duration, len(trajectory))):
                    modified_trajectory[t, 3:] *= (1.0 - 0.6 * severity)
            
            elif 'guidance' in anomaly_type:
                # Reduced course deviation
                deviation_angle = 0.6 * severity * np.pi/4
                for t in range(anomaly_time, len(trajectory)):
                    rotation = np.array([
                        [np.cos(deviation_angle), -np.sin(deviation_angle), 0],
                        [np.sin(deviation_angle), np.cos(deviation_angle), 0],
                        [0, 0, 1]
                    ])
                    modified_trajectory[t, :3] = rotation @ trajectory[t, :3]
                    modified_trajectory[t, 3:] = rotation @ trajectory[t, 3:]
            
            elif any(x in anomaly_type for x in ['structure', 'separation', 'debris']):
                # Reduced debris field
                debris_count = int(5 * severity)
                debris_velocities = []
                for _ in range(debris_count):
                    spread_angle = np.random.uniform(0, 2*np.pi)
                    spread_velocity = 0.6 * severity * 100
                    debris_velocities.append([
                        spread_velocity * np.cos(spread_angle),
                        spread_velocity * np.sin(spread_angle),
                        0
                    ])
            
            # Record anomaly with reduced impact
            anomaly = {
                'type': anomaly_type,
                'time': anomaly_time,
                'severity': severity,
                'duration': duration,
                'impact_velocity': np.linalg.norm(modified_trajectory[anomaly_time, 3:]),
                'impact_energy': 0.25 * severity * np.linalg.norm(modified_trajectory[anomaly_time, 3:])**2,
                'debris_count': debris_count if any(x in anomaly_type for x in ['structure', 'separation', 'debris']) else 0,
                'recovery_time': int(duration * (1 + 0.6 * severity))
            }
            anomalies.append(anomaly)
        
        return modified_trajectory, anomalies, anomaly_probabilities

    def generate_training_data(
        self,
        num_samples: int = 1000,
        sequence_length: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data with improved physics and success probabilities"""
        X = np.zeros((num_samples, sequence_length, 128))
        y = np.zeros((num_samples, 22))  # [normality, objects, 10 anomalies, 10 threats]
        
        for i in range(num_samples):
            try:
                # Select random vehicle and site with bias towards reliable configurations
                vehicle_type = np.random.choice(
                    ['small', 'medium', 'heavy'],
                    p=[0.4, 0.4, 0.2]  # Increased probability of reliable vehicles
                )
                vehicle = LaunchVehicle(vehicle_type)
                site = self.launch_sites['KSC']  # Using most reliable site
                
                # Generate favorable weather conditions
                season = np.random.choice(['winter', 'spring', 'summer', 'fall'])
                weather_conditions = {
                    'wind_speed': np.random.normal(8, 4),
                    'visibility': np.random.normal(9000, 1500),
                    'temperature': np.random.normal(22, 8),
                    'precipitation': np.random.exponential(0.8),
                    'pressure': np.random.normal(1013.25, 8),
                    'humidity': np.random.uniform(0, 90)
                }
                
                # Generate trajectory with improved physics
                trajectory, events = self._generate_nominal_trajectory(
                    vehicle,
                    {'altitude': 400000, 'inclination': np.radians(28)},
                    sequence_length
                )
                
                # Add potential anomalies with reduced probability and severity
                trajectory, anomalies, anomaly_probabilities = self._add_anomalies(
                    trajectory, events,
                    anomaly_probability=0.15
                )
                
                # Calculate normality score (success probability)
                normality = vehicle.base_reliability
                if anomalies:
                    normality *= (1.0 - 0.5 * sum(a['severity'] for a in anomalies))
                
                # Calculate object count (including debris)
                object_count = 1.0  # Main vehicle
                if anomalies:
                    for anomaly in anomalies:
                        object_count += anomaly.get('debris_count', 0)
                
                # Calculate threat features
                threat_features = np.zeros(10)
                if anomalies:
                    for idx, anomaly in enumerate(anomalies):
                        # Convert severity to threat features
                        threat_features[idx] = anomaly['severity']
                        if 'debris' in anomaly['type']:
                            threat_features[idx] *= 1.5  # Higher threat for debris events
                        elif 'engine' in anomaly['type']:
                            threat_features[idx] *= 1.2  # Medium threat for engine issues
                
                # Combine all outputs
                y[i, 0] = normality
                y[i, 1] = object_count
                y[i, 2:12] = anomaly_probabilities  # 10 anomaly probabilities
                y[i, 12:22] = threat_features  # 10 threat features
                
                # Generate feature vectors
                for t in range(sequence_length):
                    # Basic state features
                    X[i, t, :6] = trajectory[t]
                    
                    # Vehicle features
                    stage = min(int(t * vehicle.stages / sequence_length), vehicle.stages - 1)
                    thrust, isp = vehicle.thrust_profiles[stage]
                    X[i, t, 6:16] = [
                        vehicle.total_mass * vehicle.mass_fractions[stage],
                        thrust,
                        isp,
                        vehicle.base_reliability,
                        vehicle.stage_reliability[stage],
                        normality,
                        object_count,
                        weather_conditions['wind_speed'] / 25.0,
                        weather_conditions['visibility'] / 10000.0,
                        weather_conditions['precipitation'] / 5.0
                    ]
                    
                    # Fill remaining features with normalized physics parameters
                    X[i, t, 16:] = np.random.normal(0.5, 0.1, 112)  # Reduced variation
                
            except Exception as e:
                print(f"Error generating sample {i}: {str(e)}")
                X[i] = np.zeros((sequence_length, 128))
                y[i] = np.zeros(22)
                y[i, 0] = 1.0  # Default to normal launch
        
        return X, y

    def generate_launch_data(self, num_samples: int) -> Dict:
        """Generate synthetic launch data for training.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing sequences and labels
        """
        sequences = []
        labels = []
        
        vehicle_types = ['heavy', 'medium', 'small']
        target_orbit = {'altitude': 400000}  # 400 km orbit
        
        for _ in range(num_samples):
            # Randomly select vehicle type
            vehicle_type = np.random.choice(vehicle_types)
            vehicle = LaunchVehicle(vehicle_type)
            
            # Generate trajectory
            sequence_length = 100
            trajectory, events = self._generate_nominal_trajectory(vehicle, target_orbit, sequence_length)
            
            # Add anomalies
            modified_trajectory, events, anomaly_probs = self._add_anomalies(trajectory, events)
            
            # Create feature sequence with additional features to match input_dim=128
            sequence = np.zeros((sequence_length, 128))
            sequence[:, :6] = modified_trajectory  # Position and velocity
            sequence[:, 6] = vehicle.total_mass  # Mass
            sequence[:, 7] = vehicle.stages  # Number of stages
            sequence[:, 8] = vehicle.base_reliability  # Base reliability
            sequence[:, 9:12] = np.array(vehicle.stage_reliability)[:3]  # Stage reliability
            sequence[:, 12:17] = np.array(list(vehicle.failure_modes.values()))  # Failure mode probabilities
            sequence[:, 17:22] = np.array(list(vehicle.phase_reliability.values()))  # Phase reliability
            
            # Fill remaining features with physics-based parameters
            for t in range(sequence_length):
                altitude = np.sqrt(modified_trajectory[t, 0]**2 + modified_trajectory[t, 1]**2)
                velocity = np.sqrt(modified_trajectory[t, 3]**2 + modified_trajectory[t, 4]**2)
                
                # Additional physics features
                sequence[t, 22:25] = np.array([altitude, velocity, np.arctan2(modified_trajectory[t, 4], modified_trajectory[t, 3])])  # Altitude, velocity, flight angle
                sequence[t, 25:28] = np.array([1.225 * np.exp(-altitude/7400), 0.15 + 0.15 * min(velocity/1000, 1.0), np.pi * (4.0 ** 2)])  # Air density, drag coefficient, cross-section
                
                # Fill remaining features with normalized physics parameters and random noise
                sequence[t, 28:] = np.random.normal(0, 0.1, 100)  # Random features for remaining dimensions
            
            # Determine label (0: failure, 1: success)
            success = all(event['type'] == 'orbit_insertion' for event in events)
            label = 1 if success else 0
            
            sequences.append(sequence)
            labels.append(label)
        
        return {
            'sequences': np.array(sequences),
            'labels': np.array(labels)
        }

if __name__ == '__main__':
    # Example usage
    generator = LaunchDataGenerator()
    X, y = generator.generate_training_data(num_samples=10)
    
    print("\nSample launch metrics:")
    for i in range(3):
        print(f"Sample {i}:")
        print(f"  Normality score: {y[i][0]:.3f}")
        print(f"  Object count: {y[i][1]:.1f}")
        print(f"  Anomaly probabilities: {y[i][2:12]}")
        print(f"  Threat features: {y[i][12:22]}")
