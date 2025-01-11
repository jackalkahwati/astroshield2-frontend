import numpy as np
from scipy.integrate import odeint
from typing import Tuple, List, Dict
import datetime

class ManeuverDataGenerator:
    """Generate synthetic data for spacecraft maneuver planning"""
    
    def __init__(self):
        # Physical constants
        self.G = 6.67430e-11  # Gravitational constant
        self.M = 5.972e24     # Earth mass
        self.R = 6371000      # Earth radius
        
        # Spacecraft parameters
        self.isp_range = (220, 320)  # Specific impulse range (s)
        self.thrust_range = (1, 10)   # Thrust range (N)
        self.min_safe_distance = 100000  # Minimum safe distance (m)
        
        # Environment parameters
        self.max_objects = 5  # Maximum number of nearby objects to consider

    def _fuel_consumption(self, thrust: float, isp: float, duration: float) -> float:
        """Calculate fuel consumption for a maneuver"""
        g0 = 9.81  # Standard gravity
        mdot = thrust / (isp * g0)  # Mass flow rate
        return mdot * duration

    def _relative_motion(
        self,
        state: np.ndarray,
        target_state: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Calculate relative position and velocity"""
        if len(state.shape) > 1 or len(target_state.shape) > 1:
            state = state.flatten()
            target_state = target_state.flatten()
        
        rel_pos = target_state[:3] - state[:3]
        rel_vel = target_state[3:] - state[3:]
        distance = np.linalg.norm(rel_pos)
        return np.concatenate([rel_pos, rel_vel]), distance

    def _gravitational_acceleration(self, position: np.ndarray) -> np.ndarray:
        """Calculate gravitational acceleration"""
        r = np.linalg.norm(position)
        if r < 1e-10:  # Avoid division by zero
            return np.zeros(3)
        return -self.G * self.M * position / r**3

    def _maneuver_dynamics(
        self,
        state: np.ndarray,
        t: float,
        thrust_vector: np.ndarray,
        mass: float
    ) -> np.ndarray:
        """State propagation with thrust"""
        position = state[:3]
        velocity = state[3:]
        
        # Gravitational acceleration
        a_gravity = self._gravitational_acceleration(position)
        
        # Thrust acceleration
        a_thrust = thrust_vector / mass
        
        # Total acceleration
        acceleration = a_gravity + a_thrust
        
        # Limit maximum acceleration to avoid numerical instability
        a_mag = np.linalg.norm(acceleration)
        if a_mag > 100:  # Limit to 10g
            acceleration = acceleration * 100 / a_mag
        
        return np.concatenate([velocity, acceleration])

    def generate_maneuver_sequence(
        self,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        obstacles: List[np.ndarray],
        duration: float,
        timesteps: int,
        mass: float,
        thrust_profile: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Generate a maneuver sequence with physics-based constraints"""
        try:
            # Time vector
            t = np.linspace(0, duration, timesteps)
            
            # Initialize arrays
            states = np.zeros((timesteps, 6))
            states[0] = initial_state
            
            # Specific impulse for this maneuver
            isp = np.random.uniform(*self.isp_range)
            
            # Track metrics
            fuel_used = 0
            min_obstacle_distance = float('inf')
            
            # Propagate trajectory
            for i in range(1, timesteps):
                dt = t[i] - t[i-1]
                
                # Current thrust vector
                thrust = thrust_profile[i-1]
                thrust_mag = np.linalg.norm(thrust)
                
                # Update fuel consumption
                fuel_used += self._fuel_consumption(thrust_mag, isp, dt)
                
                # Propagate state
                solution = odeint(
                    self._maneuver_dynamics,
                    states[i-1],
                    [t[i-1], t[i]],
                    args=(thrust, mass - fuel_used),
                    rtol=1e-8,
                    atol=1e-8
                )
                states[i] = solution[-1]
                
                # Check obstacle distances
                for obstacle in obstacles:
                    if isinstance(obstacle, np.ndarray):
                        if len(obstacle.shape) == 1:
                            obstacle_state = obstacle
                        else:
                            obstacle_state = obstacle[min(i, len(obstacle)-1)]
                    else:
                        obstacle_state = np.array(obstacle)
                    
                    _, distance = self._relative_motion(states[i], obstacle_state)
                    min_obstacle_distance = min(min_obstacle_distance, distance)
            
            # Calculate final metrics
            _, final_distance = self._relative_motion(states[-1], target_state)
            
            metrics = {
                'fuel_used': fuel_used,
                'min_obstacle_distance': min_obstacle_distance,
                'final_distance': final_distance,
                'success': final_distance < 1000 and min_obstacle_distance > self.min_safe_distance
            }
            
            return states, metrics
            
        except Exception as e:
            print(f"Error in maneuver sequence generation: {str(e)}")
            return np.zeros((timesteps, 6)), {
                'fuel_used': 0,
                'min_obstacle_distance': 0,
                'final_distance': float('inf'),
                'success': False
            }

    def generate_training_data(
        self,
        num_samples: int = 10000,
        sequence_length: int = 48
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for maneuver planning.
        
        Args:
            num_samples: Number of samples to generate
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (features, labels)
        """
        # Initialize arrays
        X = np.zeros((num_samples, sequence_length, 128))  # 128 features per timestep
        y = np.zeros((num_samples, 13))  # [4 maneuver types, 3 delta-v, 2 fuel metrics, 3 safety metrics, confidence]
        
        # Maneuver types
        maneuver_types = ['hohmann', 'continuous_thrust', 'impulsive', 'avoidance']
        
        for i in range(num_samples):
            try:
                # Select random maneuver type
                maneuver_type = np.random.choice(maneuver_types)
                
                # Generate initial state
                initial_state = self._generate_initial_state()
                target_state = self._generate_target_state(initial_state, maneuver_type)
                
                # Generate maneuver parameters
                if maneuver_type == 'hohmann':
                    params = self._generate_hohmann_parameters(initial_state, target_state)
                elif maneuver_type == 'continuous_thrust':
                    params = self._generate_continuous_thrust_parameters(initial_state, target_state)
                elif maneuver_type == 'impulsive':
                    params = self._generate_impulsive_parameters(initial_state, target_state)
                else:  # avoidance
                    params = self._generate_avoidance_parameters(initial_state)
                
                # Generate sequence
                sequence = []
                fuel_consumption = 0.0
                min_distance = float('inf')
                
                for t in range(sequence_length):
                    # Update state based on maneuver type
                    if maneuver_type == 'hohmann':
                        state, fuel = self._apply_hohmann_maneuver(initial_state, params, t/sequence_length)
                    elif maneuver_type == 'continuous_thrust':
                        state, fuel = self._apply_continuous_thrust(initial_state, params, t/sequence_length)
                    elif maneuver_type == 'impulsive':
                        state, fuel = self._apply_impulsive_maneuver(initial_state, params, t/sequence_length)
                    else:  # avoidance
                        state, fuel = self._apply_avoidance_maneuver(initial_state, params, t/sequence_length)
                    
                    fuel_consumption += fuel
                    
                    # Calculate minimum distance to obstacles
                    if maneuver_type == 'avoidance':
                        distance = np.linalg.norm(state[:3] - params['obstacle_state'][:3])
                        min_distance = min(min_distance, distance)
                    
                    sequence.append(state)
                
                sequence = np.array(sequence)
                
                # Calculate maneuver metrics
                delta_v = self._calculate_delta_v(sequence)
                fuel_efficiency = self._calculate_fuel_efficiency(fuel_consumption, delta_v)
                
                # Calculate safety metrics
                if maneuver_type == 'avoidance':
                    collision_risk = np.exp(-min_distance / self.min_safe_distance)
                    safety_confidence = 1.0 - collision_risk
                else:
                    collision_risk = 0.0
                    safety_confidence = 1.0
                
                # Generate feature vector for each timestep
                for t in range(sequence_length):
                    X[i, t] = np.concatenate([
                        sequence[t],  # State vector
                        self._calculate_orbital_elements(sequence[t]),
                        delta_v,
                        np.array([fuel_consumption, fuel_efficiency]),
                        np.array([collision_risk, min_distance, safety_confidence]),
                        np.array([float(mt == maneuver_type) for mt in maneuver_types])
                    ])
                
                # Generate labels
                y[i] = np.concatenate([
                    np.array([float(mt == maneuver_type) for mt in maneuver_types]),
                    delta_v,
                    np.array([fuel_consumption, fuel_efficiency]),
                    np.array([collision_risk, min_distance, safety_confidence]),
                    np.array([0.9])  # High confidence for synthetic data
                ])
                
            except Exception as e:
                print(f"Error generating sample {i}: {str(e)}")
                # Fill with zeros if generation fails
                continue
        
        return X, y
    
    def _generate_initial_state(self) -> np.ndarray:
        """Generate random initial state vector"""
        # Random position in LEO
        r = np.random.uniform(self.R + 200e3, self.R + 2000e3)  # 200-2000 km altitude
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(-np.pi/2, np.pi/2)
        
        # Convert to Cartesian coordinates
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        
        # Calculate circular orbit velocity
        v = np.sqrt(self.G * self.M / r)
        
        # Velocity components (circular orbit)
        vx = -v * np.sin(theta)
        vy = v * np.cos(theta)
        vz = 0.0
        
        return np.array([x, y, z, vx, vy, vz])
    
    def _generate_target_state(
        self,
        initial_state: np.ndarray,
        maneuver_type: str
    ) -> np.ndarray:
        """Generate target state based on maneuver type"""
        if maneuver_type == 'hohmann':
            # Target state in higher/lower orbit
            r_init = np.linalg.norm(initial_state[:3])
            scale = np.random.choice([0.8, 1.2])  # 20% change in altitude
            r_target = r_init * scale
            
            # Keep same orientation but adjust radius
            pos = initial_state[:3] * (r_target / r_init)
            vel = initial_state[3:] * np.sqrt(r_init / r_target)  # Adjust velocity for circular orbit
            
            return np.concatenate([pos, vel])
            
        elif maneuver_type == 'continuous_thrust':
            # Target state with plane change
            r = np.linalg.norm(initial_state[:3])
            angle = np.random.uniform(np.pi/12, np.pi/6)  # 15-30 degree change
            
            # Rotation matrix for plane change
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            
            pos = R @ initial_state[:3]
            vel = R @ initial_state[3:]
            
            return np.concatenate([pos, vel])
            
        elif maneuver_type == 'impulsive':
            # Target state with slight eccentricity change
            r = np.linalg.norm(initial_state[:3])
            v = np.linalg.norm(initial_state[3:])
            
            # Add small random velocity change
            dv = np.random.uniform(-0.1, 0.1, 3) * v
            new_vel = initial_state[3:] + dv
            
            return np.concatenate([initial_state[:3], new_vel])
            
        else:  # avoidance
            # Return original state (goal is to return to original orbit after avoidance)
            return initial_state.copy()
    
    def _calculate_orbital_elements(self, state: np.ndarray) -> np.ndarray:
        """Calculate orbital elements from state vector"""
        r = state[:3]
        v = state[3:]
        
        # Angular momentum
        h = np.cross(r, v)
        
        # Node vector
        n = np.cross([0, 0, 1], h)
        
        # Eccentricity vector
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        e = ((v_mag**2 - self.G * self.M / r_mag) * r - np.dot(r, v) * v) / (self.G * self.M)
        
        # Orbital elements
        a = -self.G * self.M / (2 * (v_mag**2 / 2 - self.G * self.M / r_mag))  # Semi-major axis
        e_mag = np.linalg.norm(e)  # Eccentricity
        i = np.arccos(h[2] / np.linalg.norm(h))  # Inclination
        
        return np.array([a, e_mag, i])
    
    def _calculate_delta_v(self, sequence: np.ndarray) -> np.ndarray:
        """Calculate total delta-v from sequence"""
        v_initial = sequence[0, 3:]
        v_final = sequence[-1, 3:]
        return v_final - v_initial
    
    def _calculate_fuel_efficiency(
        self,
        fuel_consumption: float,
        delta_v: np.ndarray
    ) -> float:
        """Calculate fuel efficiency metric"""
        dv_mag = np.linalg.norm(delta_v)
        if fuel_consumption > 0:
            return dv_mag / fuel_consumption
        return 0.0

    def validate_physics(self, sequence: np.ndarray, actions: np.ndarray) -> bool:
        """Validate physics constraints in generated data"""
        try:
            for i in range(len(sequence)):
                state = sequence[i, :6]  # Current state
                
                # Check position bounds
                r = np.linalg.norm(state[:3])
                if r < self.R or r > 100000000:  # Below Earth surface or beyond Moon
                    return False
                
                # Check velocity bounds
                v = np.linalg.norm(state[3:])
                if v > 12000:  # Exceeds escape velocity
                    return False
                
                # Check thrust bounds
                if i < len(actions):
                    thrust = np.linalg.norm(actions[i])
                    if thrust > max(self.thrust_range):
                        return False
            
            return True
            
        except Exception:
            return False

if __name__ == '__main__':
    # Example usage and validation
    generator = ManeuverDataGenerator()
    
    # Generate sample data
    X, A, R = generator.generate_training_data(num_samples=10)
    
    # Validate physics
    valid_sequences = 0
    for i in range(len(X)):
        if generator.validate_physics(X[i], A[i]):
            valid_sequences += 1
    
    print(f"Generated {len(X)} sequences")
    print(f"Physics validation passed: {valid_sequences}/{len(X)}")
    
    # Print sample rewards
    print("\nSample rewards:")
    for i in range(3):
        print(f"Sequence {i}: {R[i]:.3f}")
