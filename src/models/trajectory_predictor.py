"""Enhanced trajectory prediction with multiple atmospheric models and breakup modeling."""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
import warnings

@dataclass
class AtmosphericConditions:
    """Container for atmospheric conditions at a given altitude."""
    density: float  # kg/m³
    temperature: float  # K
    pressure: float  # Pa
    wind_velocity: Dict[str, float]  # m/s
    turbulence: Optional[Dict[str, float]] = None  # m/s, turbulence components

@dataclass
class BreakupFragment:
    """Container for breakup fragment characteristics."""
    mass: float  # kg
    area: float  # m²
    cd: float  # drag coefficient
    velocity_delta: Dict[str, float]  # m/s, velocity perturbation from breakup

class TrajectoryPredictor:
    """Enhanced trajectory predictor with multiple atmospheric models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trajectory predictor.
        
        Args:
            config: Configuration dictionary containing:
                - atmospheric_model: Model name ('exponential', 'nrlmsise', 'jacchia')
                - wind_model: Wind model name ('hwm14', 'custom')
                - monte_carlo_samples: Number of Monte Carlo samples
                - breakup_model: Breakup model configuration
                - object_properties: Default object properties
        """
        self.config = config
        self.R = 6371000.0  # Earth radius in meters
        self.g0 = 9.81  # Standard gravity in m/s²
        self.monte_carlo_samples = config.get('monte_carlo_samples', 1000)
        
        # Load atmospheric model coefficients
        self._load_atmospheric_model()
        self._load_wind_model()
        
        # Initialize breakup model parameters
        self.breakup_config = config.get('breakup_model', {})
        
    def _load_atmospheric_model(self):
        """Load the specified atmospheric model."""
        model_name = self.config.get('atmospheric_model', 'exponential')
        
        if model_name == 'nrlmsise':
            try:
                from nrlmsise_00_dummy import gtd7  # Replace with actual import
                self.nrlmsise_model = gtd7
            except ImportError:
                warnings.warn("NRLMSISE-00 model not available, falling back to exponential")
                model_name = 'exponential'
        
        elif model_name == 'jacchia':
            try:
                from jacchia_dummy import jach70  # Replace with actual import
                self.jacchia_model = jach70
            except ImportError:
                warnings.warn("Jacchia model not available, falling back to exponential")
                model_name = 'exponential'
        
        self.atmospheric_model = model_name
        
    def _load_wind_model(self):
        """Load the specified wind model."""
        model_name = self.config.get('wind_model', 'hwm14')
        
        if model_name == 'hwm14':
            try:
                from hwm14_dummy import hwm14  # Replace with actual import
                self.hwm14_model = hwm14
            except ImportError:
                warnings.warn("HWM14 model not available, falling back to basic wind model")
                model_name = 'custom'
        
        self.wind_model = model_name

    def get_atmospheric_conditions(self, altitude: float, lat: float = 0.0, lon: float = 0.0, 
                                 time: Optional[datetime] = None) -> AtmosphericConditions:
        """Get atmospheric conditions using the selected model.
        
        Args:
            altitude: Altitude in meters
            lat: Latitude in degrees
            lon: Longitude in degrees
            time: Optional datetime for time-dependent models
            
        Returns:
            AtmosphericConditions object
        """
        if self.atmospheric_model == 'nrlmsise':
            # Use NRLMSISE-00 model
            conditions = self.nrlmsise_model(time or datetime.utcnow(), lat, lon, altitude)
            density = conditions['total_density']
            temperature = conditions['temperature']
            pressure = conditions['pressure']
            
        elif self.atmospheric_model == 'jacchia':
            # Use Jacchia model
            conditions = self.jacchia_model(time or datetime.utcnow(), lat, lon, altitude)
            density = conditions['density']
            temperature = conditions['temperature']
            pressure = conditions['pressure']
            
        else:
            # Use exponential model with scale height variation
            h0 = 7400.0  # Scale height in meters
            rho0 = 1.225  # Sea level density in kg/m³
            T0 = 288.15  # Sea level temperature in K
            P0 = 101325.0  # Sea level pressure in Pa
            
            # Add latitude-dependent scale height variation
            h0_lat = h0 * (1.0 + 0.02 * np.cos(np.radians(lat)))
            
            density = rho0 * np.exp(-altitude / h0_lat)
            temperature = T0 * np.exp(-altitude / (7.0 * h0_lat))
            pressure = P0 * np.exp(-altitude / h0_lat)
        
        # Get wind velocity
        wind = self._get_wind_velocity(altitude, lat, lon, time)
        
        # Add turbulence modeling
        turbulence = self._calculate_turbulence(altitude, density)
        
        return AtmosphericConditions(
            density=density,
            temperature=temperature,
            pressure=pressure,
            wind_velocity=wind,
            turbulence=turbulence
        )
    
    def _get_wind_velocity(self, altitude: float, lat: float, lon: float, 
                          time: Optional[datetime] = None) -> Dict[str, float]:
        """Get wind velocity using the selected model."""
        if self.wind_model == 'hwm14':
            # Use HWM14 model
            wind = self.hwm14_model(time or datetime.utcnow(), lat, lon, altitude)
            return {
                'vx': wind['meridional'],
                'vy': wind['zonal'],
                'vz': wind['vertical']
            }
        else:
            # Use custom wind model with altitude variation
            base_wind = 10.0  # Base wind speed in m/s
            max_wind = 150.0  # Maximum wind speed in m/s
            peak_alt = 12000.0  # Altitude of maximum wind in meters
            
            # Calculate wind speed with altitude variation
            wind_speed = base_wind + (max_wind - base_wind) * \
                        np.exp(-(altitude - peak_alt)**2 / (2 * 5000.0**2))
            
            # Add latitudinal variation
            wind_speed *= (1.0 + 0.3 * np.cos(np.radians(4.0 * lat)))
            
            return {
                'vx': wind_speed * np.cos(np.radians(lon)),
                'vy': wind_speed * np.sin(np.radians(lon)),
                'vz': 0.0
            }
    
    def _calculate_turbulence(self, altitude: float, density: float) -> Dict[str, float]:
        """Calculate turbulence components based on altitude and density."""
        # Turbulence intensity decreases with altitude and density
        base_turbulence = 5.0  # Base turbulence in m/s
        intensity = base_turbulence * np.exp(-altitude / 20000.0) * (density / 1.225)**0.5
        
        return {
            'vx': np.random.normal(0, intensity),
            'vy': np.random.normal(0, intensity),
            'vz': np.random.normal(0, intensity)
        }

    def _calculate_drag(self, velocity: np.ndarray, conditions: AtmosphericConditions) -> np.ndarray:
        """Calculate drag force with enhanced modeling."""
        # Calculate relative velocity including wind and turbulence
        v_wind = np.array([
            conditions.wind_velocity['vx'],
            conditions.wind_velocity['vy'],
            conditions.wind_velocity['vz']
        ])
        
        if conditions.turbulence:
            v_wind += np.array([
                conditions.turbulence['vx'],
                conditions.turbulence['vy'],
                conditions.turbulence['vz']
            ])
        
        v_rel = velocity - v_wind
        v_mag = np.linalg.norm(v_rel)
        
        # Calculate Mach number
        gamma = 1.4  # Heat capacity ratio
        R = 287.05  # Gas constant for air
        a = np.sqrt(gamma * R * conditions.temperature)  # Speed of sound
        mach = v_mag / a
        
        # Get object properties
        cd = self._get_drag_coefficient(mach)
        area = self.config['object_properties']['area']
        
        # Calculate drag force
        drag_force = -0.5 * conditions.density * cd * area * v_mag * v_rel
        
        return drag_force

    def _get_drag_coefficient(self, mach: float) -> float:
        """Get Mach number-dependent drag coefficient."""
        # Basic drag coefficient variation with Mach number
        cd0 = self.config['object_properties'].get('cd', 2.2)
        
        if mach < 0.8:
            return cd0
        elif mach < 1.2:
            # Transonic regime
            return cd0 * (1.0 + 0.5 * (mach - 0.8))
        else:
            # Supersonic regime
            return cd0 * (1.0 + 0.2 / mach)

    def _dynamics(self, state: np.ndarray, t: float, mass: float) -> np.ndarray:
        """Calculate state derivatives including all forces."""
        # Extract position and velocity
        r = state[:3]
        v = state[3:]
        
        # Calculate altitude and position-dependent quantities
        alt = np.linalg.norm(r) - self.R
        lat = np.degrees(np.arcsin(r[2] / np.linalg.norm(r)))
        lon = np.degrees(np.arctan2(r[1], r[0]))
        
        # Get atmospheric conditions
        conditions = self.get_atmospheric_conditions(alt, lat, lon)
        
        # Calculate forces
        # 1. Gravity with J2 effect
        r_mag = np.linalg.norm(r)
        g = -self.g0 * (self.R / r_mag)**2
        
        J2 = 1.08263e-3
        z_r = r[2] / r_mag
        g_factor = 1.5 * J2 * (self.R / r_mag)**2 * (1.0 - 5.0 * z_r**2)
        
        g_vec = (g * (1.0 + g_factor)) * r / r_mag
        
        # 2. Drag force
        f_drag = self._calculate_drag(v, conditions)
        
        # 3. Coriolis and centrifugal forces
        omega_earth = np.array([0, 0, 7.2921159e-5])  # Earth's rotation vector
        f_coriolis = -2.0 * np.cross(omega_earth, v)
        f_centrifugal = -np.cross(omega_earth, np.cross(omega_earth, r))
        
        # Sum all accelerations
        a = g_vec + f_drag/mass + f_coriolis + f_centrifugal
        
        return np.concatenate([v, a])

    def predict_impact(self, initial_state: np.ndarray, mass: float,
                      time_step: float = 1.0, max_time: float = 3600.0,
                      monte_carlo: bool = True) -> Dict[str, Any]:
        """Predict impact location with uncertainty using Monte Carlo analysis.
        
        Args:
            initial_state: Initial state vector [x, y, z, vx, vy, vz]
            mass: Object mass in kg
            time_step: Integration time step in seconds
            max_time: Maximum integration time in seconds
            monte_carlo: Whether to perform Monte Carlo analysis
            
        Returns:
            Dictionary containing impact prediction and uncertainty
        """
        if monte_carlo:
            return self._monte_carlo_prediction(initial_state, mass, time_step, max_time)
        
        # Single trajectory prediction
        t = np.arange(0, max_time, time_step)
        trajectory = odeint(self._dynamics, initial_state, t, args=(mass,))
        
        # Find impact point
        impact_idx = None
        for i in range(len(trajectory)):
            alt = np.linalg.norm(trajectory[i, :3]) - self.R
            if alt <= 0:
                impact_idx = i
                break
        
        if impact_idx is None:
            return None
        
        # Calculate impact location
        impact_pos = trajectory[impact_idx, :3]
        impact_vel = trajectory[impact_idx, 3:]
        
        r_mag = np.linalg.norm(impact_pos)
        lat = np.degrees(np.arcsin(impact_pos[2] / r_mag))
        lon = np.degrees(np.arctan2(impact_pos[1], impact_pos[0]))
        
        # Calculate basic uncertainty
        uncertainty = self._calculate_uncertainty(
            t[impact_idx],
            initial_state,
            trajectory[:impact_idx+1]
        )
        
        return {
            'time': datetime.utcnow() + timedelta(seconds=float(t[impact_idx])),
            'location': {
                'lat': float(lat),
                'lon': float(lon)
            },
            'velocity': {
                'magnitude': float(np.linalg.norm(impact_vel)),
                'direction': {
                    'x': float(impact_vel[0]),
                    'y': float(impact_vel[1]),
                    'z': float(impact_vel[2])
                }
            },
            'uncertainty_radius_km': float(uncertainty),
            'confidence': 0.95
        }

    def _monte_carlo_prediction(self, initial_state: np.ndarray, mass: float,
                              time_step: float, max_time: float) -> Dict[str, Any]:
        """Perform Monte Carlo analysis for impact prediction."""
        impact_points = []
        impact_times = []
        impact_velocities = []
        
        # Generate Monte Carlo samples
        for _ in range(self.monte_carlo_samples):
            # Add perturbations to initial state and parameters
            perturbed_state = self._add_state_perturbations(initial_state)
            perturbed_mass = mass * (1.0 + np.random.normal(0, 0.1))  # 10% uncertainty
            
            # Predict trajectory
            t = np.arange(0, max_time, time_step)
            trajectory = odeint(self._dynamics, perturbed_state, t, args=(perturbed_mass,))
            
            # Find impact point
            for i in range(len(trajectory)):
                alt = np.linalg.norm(trajectory[i, :3]) - self.R
                if alt <= 0:
                    impact_points.append(trajectory[i, :3])
                    impact_times.append(t[i])
                    impact_velocities.append(trajectory[i, 3:])
                    break
        
        if not impact_points:
            return None
        
        # Statistical analysis of Monte Carlo results
        impact_points = np.array(impact_points)
        impact_times = np.array(impact_times)
        impact_velocities = np.array(impact_velocities)
        
        # Calculate mean impact point
        mean_pos = np.mean(impact_points, axis=0)
        r_mag = np.linalg.norm(mean_pos)
        mean_lat = np.degrees(np.arcsin(mean_pos[2] / r_mag))
        mean_lon = np.degrees(np.arctan2(mean_pos[1], mean_pos[0]))
        
        # Calculate uncertainty ellipse
        covariance = np.cov(impact_points, rowvar=False)
        eigenvals, eigenvecs = np.linalg.eig(covariance[:2, :2])
        uncertainty_radius = np.sqrt(np.max(eigenvals)) / 1000.0  # Convert to km
        
        # Calculate confidence based on spread
        confidence = 1.0 - np.std(impact_times) / np.mean(impact_times)
        confidence = max(0.5, min(0.95, confidence))
        
        return {
            'time': datetime.utcnow() + timedelta(seconds=float(np.mean(impact_times))),
            'location': {
                'lat': float(mean_lat),
                'lon': float(mean_lon)
            },
            'velocity': {
                'magnitude': float(np.mean(np.linalg.norm(impact_velocities, axis=1))),
                'direction': {
                    'x': float(np.mean(impact_velocities[:, 0])),
                    'y': float(np.mean(impact_velocities[:, 1])),
                    'z': float(np.mean(impact_velocities[:, 2]))
                }
            },
            'uncertainty_radius_km': float(uncertainty_radius),
            'confidence': float(confidence),
            'monte_carlo_stats': {
                'samples': self.monte_carlo_samples,
                'time_std': float(np.std(impact_times)),
                'position_std': float(np.std(np.linalg.norm(impact_points, axis=1))),
                'velocity_std': float(np.std(np.linalg.norm(impact_velocities, axis=1)))
            }
        }

    def _add_state_perturbations(self, state: np.ndarray) -> np.ndarray:
        """Add random perturbations to the state vector."""
        # Position uncertainty (1% of position magnitude)
        pos_scale = np.linalg.norm(state[:3]) * 0.01
        pos_perturbation = np.random.normal(0, pos_scale, 3)
        
        # Velocity uncertainty (1% of velocity magnitude)
        vel_scale = np.linalg.norm(state[3:]) * 0.01
        vel_perturbation = np.random.normal(0, vel_scale, 3)
        
        return state + np.concatenate([pos_perturbation, vel_perturbation])

    def model_breakup(self, state: np.ndarray, mass: float, 
                     breakup_altitude: float) -> List[BreakupFragment]:
        """Model object breakup and generate fragments.
        
        Args:
            state: State vector at breakup
            mass: Object mass in kg
            breakup_altitude: Altitude of breakup in meters
            
        Returns:
            List of BreakupFragment objects
        """
        # Get breakup model parameters
        fragment_count = self.breakup_config.get('fragment_count', 10)
        mass_distribution = self.breakup_config.get('mass_distribution', 'log_normal')
        velocity_perturbation = self.breakup_config.get('velocity_perturbation', 100.0)
        
        fragments = []
        
        if mass_distribution == 'log_normal':
            # Generate log-normally distributed fragment masses
            total_mass = mass
            mu = np.log(total_mass / fragment_count)
            sigma = 0.5
            
            masses = np.random.lognormal(mu, sigma, fragment_count)
            masses = masses * (total_mass / np.sum(masses))  # Normalize to total mass
            
        else:
            # Simple equal mass distribution
            masses = np.ones(fragment_count) * mass / fragment_count
        
        # Generate fragments
        for fragment_mass in masses:
            # Calculate fragment area assuming constant density
            area = (fragment_mass / mass)**(2/3) * self.config['object_properties']['area']
            
            # Generate random velocity perturbation
            v_pert = np.random.normal(0, velocity_perturbation, 3)
            
            fragments.append(BreakupFragment(
                mass=float(fragment_mass),
                area=float(area),
                cd=self.config['object_properties'].get('cd', 2.2),
                velocity_delta={
                    'vx': float(v_pert[0]),
                    'vy': float(v_pert[1]),
                    'vz': float(v_pert[2])
                }
            ))
        
        return fragments

    def _calculate_uncertainty(self,
                             impact_time: float,
                             initial_state: np.ndarray,
                             trajectory: np.ndarray) -> float:
        """Calculate uncertainty radius for impact prediction.
        
        Args:
            impact_time: Time to impact in seconds
            initial_state: Initial state vector
            trajectory: Predicted trajectory points
            
        Returns:
            Uncertainty radius in meters
        """
        # Base uncertainty increases with prediction time
        time_uncertainty = 100 * np.sqrt(impact_time)  # meters per hour
        
        # Additional uncertainty from atmospheric conditions
        altitude_profile = np.array([
            np.linalg.norm(trajectory[i, :3]) - self.R
            for i in range(len(trajectory))
        ])
        
        # Uncertainty from wind variations
        wind_uncertainty = 0.0
        for alt in altitude_profile:
            conditions = self.get_atmospheric_conditions(alt)
            wind_speed = np.sqrt(
                conditions.wind_velocity['vx']**2 +
                conditions.wind_velocity['vy']**2
            )
            wind_uncertainty += wind_speed * 0.2  # 20% wind uncertainty
        
        # Uncertainty from initial state errors (assumed 1% for each component)
        state_uncertainty = 0.01 * np.linalg.norm(initial_state[:3])
        
        # Combine uncertainties (root sum square)
        total_uncertainty = np.sqrt(
            time_uncertainty**2 +
            wind_uncertainty**2 +
            state_uncertainty**2
        )
        
        return total_uncertainty 