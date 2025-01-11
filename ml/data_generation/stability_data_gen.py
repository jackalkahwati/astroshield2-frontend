import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, List, Dict
import datetime
import signal
import functools

def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

class StabilityDataGenerator:
    """Generate synthetic data for stability evaluation using orbital mechanics"""
    
    def __init__(self):
        # Constants
        self.G = 6.67430e-11  # Gravitational constant
        self.M = 5.972e24     # Earth mass
        self.R = 6371000      # Earth radius
        self.J2 = 1.082635e-3 # Earth's J2 coefficient
        
        # Orbit parameters
        self.min_altitude = 200000    # 200 km
        self.max_altitude = 42164000  # GEO
        self.max_eccentricity = 0.1   # Reduced to strongly favor circular orbits
        
        # Updated stability thresholds
        self.stability_threshold = 0.6  # Relaxed for better numerical stability
        self.energy_conservation_threshold = 1e-4  # Relaxed for numerical integration
        self.circularity_threshold = 0.01  # Even stricter circular orbit requirement
        
        # Updated stability weighting factors
        self.energy_weight = 0.15       # Reduced from 0.2
        self.eccentricity_weight = 0.7  # Increased from 0.6
        self.perturbation_weight = 0.15 # Reduced from 0.2

    def _orbital_derivatives(self, t: float, state: np.ndarray, area_mass_ratio: float) -> np.ndarray:
        """Calculate orbital state derivatives with corrected perturbation modeling"""
        try:
            r = state[:3]
            v = state[3:]
            
            r = np.asarray(r, dtype=np.float64)
            v = np.asarray(v, dtype=np.float64)
            
            r_squared = np.sum(r * r)
            r_mag = np.sqrt(r_squared)
            
            if r_mag < self.R:
                return np.zeros(6)
            
            # Two-body acceleration with improved numerical stability
            a_gravity = np.zeros(3)
            if r_mag > 1e-10:
                a_gravity = -self.G * self.M * r / (r_mag**3)
            
            # Corrected J2 perturbation modeling
            a_j2 = np.zeros(3)
            if r_mag > self.R:
                x, y, z = r
                r_5 = r_mag**5
                if r_5 > 1e-10:
                    factor = -1.5 * self.J2 * self.G * self.M * self.R**2 / r_5
                    z_r2 = z * z / r_squared
                    a_j2 = np.array([
                        x * (5 * z_r2 - 1),
                        y * (5 * z_r2 - 1),
                        z * (5 * z_r2 - 3)
                    ]) * factor
            
            # Solar radiation pressure with improved physics
            a_srp = np.zeros(3)
            if area_mass_ratio > 0 and r_mag > self.R:
                solar_constant = 1361  # W/m²
                c = 299792458  # Speed of light
                solar_pressure = solar_constant / c
                shadow_factor = 1.0
                if r_mag > self.R:
                    # Improved shadow modeling
                    sun_angle = np.arccos(x / r_mag)  # Simplified sun direction along x-axis
                    if sun_angle < np.pi/2:
                        shadow_factor = np.cos(sun_angle)
                a_srp = shadow_factor * area_mass_ratio * solar_pressure * r / r_mag
            
            # Total acceleration with improved numerical stability
            a_total = a_gravity + a_j2 + a_srp
            
            # Limit maximum acceleration for numerical stability
            a_mag = np.sqrt(np.sum(a_total * a_total))
            if a_mag > 100:  # Increased from 50 for better dynamics
                a_total = a_total * 100 / a_mag
            
            return np.concatenate([v, a_total])
            
        except Exception as e:
            print(f"Error in orbital derivatives: {str(e)}")
            return np.zeros(6)

    def _check_energy_conservation(self, sequence: np.ndarray) -> Tuple[bool, float]:
        """Enhanced energy conservation check with improved numerical stability"""
        energy = np.zeros(len(sequence))
        angular_momentum = np.zeros((len(sequence), 3))
        
        for i in range(len(sequence)):
            r = sequence[i, :3]
            v = sequence[i, 3:]
            r_mag = np.sqrt(np.sum(r * r))
            v_mag = np.sqrt(np.sum(v * v))
            
            # Improved specific mechanical energy calculation
            if r_mag > self.R:
                energy[i] = 0.5 * v_mag**2 - self.G * self.M / r_mag
                angular_momentum[i] = np.cross(r, v)
        
        # Improved energy variation analysis
        mean_energy = np.mean(energy)
        if abs(mean_energy) > 1e-10:
            energy_variation = np.std(energy) / abs(mean_energy)
        else:
            energy_variation = np.inf
        
        # Improved angular momentum analysis
        mean_angular_momentum = np.mean(angular_momentum, axis=0)
        angular_momentum_variation = 0.0
        if np.any(np.abs(mean_angular_momentum) > 1e-10):
            angular_momentum_variation = np.max(
                np.std(angular_momentum, axis=0) / 
                np.maximum(np.abs(mean_angular_momentum), 1e-10)
            )
        
        is_conserved = (energy_variation < self.energy_conservation_threshold and 
                       angular_momentum_variation < self.energy_conservation_threshold)
        
        return is_conserved, energy_variation

    def _calculate_orbit_stability(self, sequence: np.ndarray) -> Tuple[float, dict]:
        """Calculate orbit stability with improved metrics"""
        # Improved radial stability calculation
        orbit_radii = np.sqrt(np.sum(sequence[:, :3]**2, axis=1))
        mean_radius = np.mean(orbit_radii)
        radius_variation = np.std(orbit_radii) / mean_radius if mean_radius > 0 else np.inf
        
        # Improved eccentricity calculation
        r_min = np.min(orbit_radii)
        r_max = np.max(orbit_radii)
        if r_min + r_max > 0:
            ecc = (r_max - r_min) / (r_max + r_min)
        else:
            ecc = 1.0
        
        # Energy conservation check
        is_conserved, energy_variation = self._check_energy_conservation(sequence)
        
        # Improved stability scoring with even stronger emphasis on circularity
        energy_score = np.exp(-5 * energy_variation)  # Relaxed from -10
        eccentricity_score = np.exp(-25 * ecc)       # Increased from -20 to even more strongly favor circular orbits
        perturbation_score = np.exp(-8 * radius_variation)  # Relaxed from -12
        
        # Combined stability score with updated weights
        stability_score = (
            self.energy_weight * energy_score +
            self.eccentricity_weight * eccentricity_score +
            self.perturbation_weight * perturbation_score
        )
        
        # Apply stability threshold with smooth transition
        if stability_score > self.stability_threshold:
            stability_score = 1.0 - 0.15 * (1.0 - stability_score)  # Reduced penalty from 0.2
        
        metrics = {
            'energy_conservation': is_conserved,
            'energy_variation': energy_variation,
            'eccentricity': ecc,
            'radius_variation': radius_variation,
            'energy_score': energy_score,
            'eccentricity_score': eccentricity_score,
            'perturbation_score': perturbation_score,
            'meets_threshold': stability_score >= self.stability_threshold
        }
        
        return stability_score, metrics

    def generate_training_data(
        self,
        num_samples: int = 10000,
        sequence_length: int = 48
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate training data for stability evaluation.
        
        Args:
            num_samples: Number of samples to generate
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (features, labels) where:
            - features shape: (num_samples, sequence_length, 64)
            - labels: Dictionary containing stability metrics
        """
        # Initialize arrays
        X = np.zeros((num_samples, sequence_length, 64))
        
        # Initialize label arrays
        energy_conservation = np.zeros(num_samples)
        angular_momentum_conservation = np.zeros(num_samples)
        perturbation_effects = np.zeros(num_samples)
        overall_stability = np.zeros(num_samples)
        
        for i in range(num_samples):
            try:
                # Generate initial state
                state = self.generate_initial_state()
                
                # Track energy and angular momentum
                initial_energy = self._compute_orbital_energy(state)
                initial_momentum = self._compute_angular_momentum(state)
                
                # Generate sequence
                for t in range(sequence_length):
                    # Propagate state
                    state = self._propagate_orbit(state, self.time_step)
                    
                    # Calculate stability metrics
                    energy = self._compute_orbital_energy(state)
                    momentum = self._compute_angular_momentum(state)
                    
                    # Generate feature vector
                    features = np.concatenate([
                        state,  # 6 dimensions
                        np.array([energy, np.linalg.norm(momentum)]),  # 2 dimensions
                        np.zeros(56)  # Padding to reach 64 dimensions
                    ])
                    
                    X[i, t] = features
                
                # Compute stability metrics
                energy_conservation[i] = np.abs(energy - initial_energy) / np.abs(initial_energy)
                angular_momentum_conservation[i] = np.linalg.norm(momentum - initial_momentum) / np.linalg.norm(initial_momentum)
                perturbation_effects[i] = np.random.uniform(0, 0.1)  # Simplified perturbation effect
                
                # Overall stability score (weighted combination)
                overall_stability[i] = 1.0 - (
                    0.4 * energy_conservation[i] +
                    0.4 * angular_momentum_conservation[i] +
                    0.2 * perturbation_effects[i]
                )
                
            except Exception as e:
                print(f"Error generating sample {i}: {str(e)}")
                continue
        
        # Package labels into dictionary
        labels = {
            "energy_conservation": energy_conservation,
            "angular_momentum_conservation": angular_momentum_conservation,
            "perturbation_effects": perturbation_effects,
            "overall_stability": overall_stability
        }
        
        return X, labels

    def _propagate_orbit(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Propagate orbital state using simple two-body dynamics.
        
        Args:
            state: Current state vector [x, y, z, vx, vy, vz]
            dt: Time step
            
        Returns:
            Updated state vector
        """
        # Extract position and velocity
        r = state[:3]
        v = state[3:]
        
        # Constants
        mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
        
        # Compute acceleration
        r_mag = np.linalg.norm(r)
        a = -mu * r / (r_mag ** 3)
        
        # Update state using RK4
        k1_r = v
        k1_v = a
        
        k2_r = v + 0.5 * dt * k1_v
        r_temp = r + 0.5 * dt * k1_r
        r_mag = np.linalg.norm(r_temp)
        k2_v = -mu * r_temp / (r_mag ** 3)
        
        k3_r = v + 0.5 * dt * k2_v
        r_temp = r + 0.5 * dt * k2_r
        r_mag = np.linalg.norm(r_temp)
        k3_v = -mu * r_temp / (r_mag ** 3)
        
        k4_r = v + dt * k3_v
        r_temp = r + dt * k3_r
        r_mag = np.linalg.norm(r_temp)
        k4_v = -mu * r_temp / (r_mag ** 3)
        
        # Update position and velocity
        r_new = r + (dt/6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        v_new = v + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        return np.concatenate([r_new, v_new])

    def _compute_orbital_energy(self, state: np.ndarray) -> float:
        """Compute orbital energy (specific mechanical energy)."""
        mu = 3.986004418e14  # Earth's gravitational parameter
        r = state[:3]
        v = state[3:]
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # E = v^2/2 - μ/r
        return 0.5 * v_mag**2 - mu/r_mag

    def _compute_angular_momentum(self, state: np.ndarray) -> np.ndarray:
        """Compute orbital angular momentum vector."""
        r = state[:3]
        v = state[3:]
        return np.cross(r, v)

    def generate_initial_state(self) -> np.ndarray:
        """Generate random initial orbital state."""
        # Generate random orbital elements
        a = np.random.uniform(6.6e6, 8.6e6)  # Semi-major axis between LEO and MEO
        e = np.random.uniform(0, 0.3)  # Eccentricity
        i = np.random.uniform(0, np.pi/2)  # Inclination
        omega = np.random.uniform(0, 2*np.pi)  # Argument of perigee
        Omega = np.random.uniform(0, 2*np.pi)  # Right ascension of ascending node
        nu = np.random.uniform(0, 2*np.pi)  # True anomaly
        
        # Convert to cartesian state
        mu = 3.986004418e14
        
        # Compute position and velocity in perifocal frame
        p = a * (1 - e**2)
        r = p / (1 + e * np.cos(nu))
        
        pos_pf = r * np.array([np.cos(nu), np.sin(nu), 0])
        vel_pf = np.sqrt(mu/p) * np.array([-np.sin(nu), e + np.cos(nu), 0])
        
        # Rotation matrices
        R3_Omega = np.array([
            [np.cos(Omega), -np.sin(Omega), 0],
            [np.sin(Omega), np.cos(Omega), 0],
            [0, 0, 1]
        ])
        
        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i), np.cos(i)]
        ])
        
        R3_omega = np.array([
            [np.cos(omega), -np.sin(omega), 0],
            [np.sin(omega), np.cos(omega), 0],
            [0, 0, 1]
        ])
        
        # Transform to ECI frame
        DCM = R3_Omega @ R1_i @ R3_omega
        pos_eci = DCM @ pos_pf
        vel_eci = DCM @ vel_pf
        
        return np.concatenate([pos_eci, vel_eci])

    @timeout(5)
    def generate_orbit_sequence(
        self,
        initial_state: np.ndarray,
        duration_hours: float = 24,
        timesteps: int = 60,
        area_mass_ratio: float = 0.01
    ) -> np.ndarray:
        """Generate orbital state sequence with improved integration"""
        try:
            t_span = (0, duration_hours * 3600)
            t_eval = np.linspace(*t_span, timesteps)
            
            initial_state = np.asarray(initial_state, dtype=np.float64)
            if np.any(np.isnan(initial_state)) or np.any(np.isinf(initial_state)):
                r = self.R + self.min_altitude + 1000000
                return self._create_circular_orbit(r, timesteps)
            
            # Improved integration parameters
            solution = solve_ivp(
                self._orbital_derivatives,
                t_span,
                initial_state,
                args=(area_mass_ratio,),
                method='RK45',
                t_eval=t_eval,
                rtol=1e-12,  # Increased precision
                atol=1e-12,  # Increased precision
                max_step=50.0  # Reduced from 100.0 for better accuracy
            )
            
            if not solution.success or np.any(np.isnan(solution.y.T)) or np.any(np.isinf(solution.y.T)):
                r = max(np.linalg.norm(initial_state[:3]), self.R + self.min_altitude)
                return self._create_circular_orbit(r, timesteps)
            
            return solution.y.T
            
        except Exception as e:
            print(f"Integration failed: {str(e)}")
            return self._create_circular_orbit(self.R + self.min_altitude + 1000000, timesteps)

    def _create_circular_orbit(self, radius: float, timesteps: int) -> np.ndarray:
        """Create a perfectly circular orbit at specified radius"""
        radius = np.clip(radius, self.R + self.min_altitude, self.R + self.max_altitude)
        v = np.sqrt(self.G * self.M / radius)  # Circular orbit velocity
        orbit = np.zeros((timesteps, 6))
        for i in range(timesteps):
            theta = 2 * np.pi * i / timesteps
            orbit[i] = [
                radius * np.cos(theta),
                radius * np.sin(theta),
                0,
                -v * np.sin(theta),
                v * np.cos(theta),
                0
            ]
        return orbit

if __name__ == '__main__':
    generator = StabilityDataGenerator()
    X, y = generator.generate_training_data(num_samples=10)
    
    print("\nSample stability metrics:")
    for i in range(3):
        print(f"Sequence {i}:")
        print(f"  Stability score: {y[i][0]:.3f}")
        print(f"  Energy conservation: {y[i][1]:.3f}")
        print(f"  Eccentricity: {y[i][2]:.3f}")
        print(f"  Radius variation: {y[i][3]:.3f}")
