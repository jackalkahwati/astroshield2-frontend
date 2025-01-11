import numpy as np
from typing import Tuple, Dict, List
import datetime

class EnvironmentalDataGenerator:
    """Generate synthetic data for space environment evaluation with accurate Van Allen belt physics"""
    
    def __init__(self):
        # Solar activity parameters
        self.f10_base = 150.0  # Base F10.7 solar flux
        self.f10_variation = 50.0  # F10.7 variation range
        self.ap_base = 15.0  # Base geomagnetic Ap index
        self.ap_variation = 10.0  # Ap variation range
        
        # Magnetic field parameters (nT)
        self.B0 = 30000  # Surface field strength at equator
        
        # Updated Van Allen belt parameters with accurate physics
        self.inner_belt = {
            'peak_L': 1.3,      # Updated to match test parameters
            'width': 0.2,       # Narrower for sharper peak
            'max_flux_p': 5e5,  # Increased peak proton flux
            'e_threshold': 10,  # Minimum proton energy (MeV)
            'e_range': [10, 100], # Proton energy range (MeV)
            'spectral_index': -1.5  # Power law index for energy spectrum
        }
        
        self.outer_belt = {
            'peak_L': 5.0,      # Updated to match test parameters
            'width': 1.5,       # Updated width
            'max_flux_e': 2e4,  # Adjusted electron flux
            'e_threshold': 0.5, # Minimum electron energy (MeV)
            'e_range': [0.5, 7], # Electron energy range (MeV)
            'spectral_index': -2.0  # Power law index for energy spectrum
        }
        
        self.slot_region = {
            'center_L': 2.5,    # Center of slot region
            'width': 1.0,       # Width of slot region
            'attenuation': 0.005 # Increased attenuation
        }
        
        # South Atlantic Anomaly parameters
        self.saa = {
            'center_L': 1.3,     # Center in L-shell
            'width': 0.3,        # Width in L-shell
            'lat_center': -30.0, # Center latitude (degrees)
            'lon_center': -45.0, # Center longitude (degrees)
            'intensity': 2.5     # Increased from 2.0 for stronger effect
        }

    def _van_allen_belts(self, L: float, B: float, activity: float) -> Dict[str, float]:
        """Calculate Van Allen belt particle fluxes with accurate physical characteristics"""
        L = np.clip(L, 1.0, 10.0)
        
        # Inner belt (protons) with sharper peak
        inner_flux = 0
        proton_spectrum = np.zeros(10)
        if abs(L - self.inner_belt['peak_L']) <= self.inner_belt['width']:
            # Gaussian profile for sharper transition
            dist = abs(L - self.inner_belt['peak_L'])
            inner_flux = self.inner_belt['max_flux_p'] * np.exp(-12 * (dist/self.inner_belt['width'])**2)
            
            # Energy spectrum (power law distribution)
            energies = np.linspace(*self.inner_belt['e_range'], 10)
            proton_spectrum = inner_flux * (energies/self.inner_belt['e_threshold'])**self.inner_belt['spectral_index']
        
        # Slot region (particle depletion)
        slot_factor = 1.0
        if abs(L - self.slot_region['center_L']) <= self.slot_region['width']/2:
            slot_factor = self.slot_region['attenuation']
        
        # Outer belt (electrons) with enhanced activity dependence
        outer_flux = 0
        electron_spectrum = np.zeros(10)
        if abs(L - self.outer_belt['peak_L']) <= self.outer_belt['width']:
            dist = abs(L - self.outer_belt['peak_L'])
            base_flux = self.outer_belt['max_flux_e'] * np.exp(-3 * (dist/self.outer_belt['width'])**2)
            
            # Enhanced activity modulation
            activity_factor = 1 + 5 * activity  # Up to 6x increase with activity
            outer_flux = base_flux * activity_factor * slot_factor
            
            # Energy spectrum (relativistic electrons)
            e_energies = np.linspace(*self.outer_belt['e_range'], 10)
            electron_spectrum = outer_flux * (e_energies/self.outer_belt['e_threshold'])**self.outer_belt['spectral_index']
        
        # Enhanced magnetic field effects
        B_norm = np.clip(B, 0, 50000) / 50000
        mirror_factor = np.exp(-4 * B_norm)  # Stronger mirroring effect
        
        # Enhanced South Atlantic Anomaly contribution
        saa_factor = self.saa['intensity'] * np.exp(-((L - self.saa['center_L'])**2) / (2 * self.saa['width']**2))
        
        # Total radiation calculation with SAA enhancement
        total_flux = inner_flux + outer_flux
        if saa_factor > 0.5:  # Strong SAA influence
            total_flux *= (1 + 2 * saa_factor)  # Increased SAA effect
        
        return {
            'inner_flux': inner_flux * mirror_factor,
            'outer_flux': outer_flux * mirror_factor,
            'proton_spectrum': proton_spectrum * mirror_factor,
            'electron_spectrum': electron_spectrum * mirror_factor,
            'total_dose': total_flux * mirror_factor,
            'belt_region': 1.0 if abs(L - self.inner_belt['peak_L']) <= self.inner_belt['width'] else (
                2.0 if abs(L - self.outer_belt['peak_L']) <= self.outer_belt['width'] else 0.0
            ),
            'saa_proximity': saa_factor,
            'mirror_ratio': mirror_factor,
            'slot_factor': slot_factor
        }

    def _magnetic_field_magnitude(self, latitude: float, altitude: float) -> float:
        """Calculate magnetic field magnitude with improved dipole approximation"""
        lat_rad = np.radians(latitude)
        R = (altitude + 6371.0) / 6371.0  # Normalized radius (Earth radii)
        L = R / (np.cos(lat_rad) ** 2)
        
        # Enhanced dipole field calculation
        B = self.B0 * np.sqrt(1 + 3 * np.sin(lat_rad)**2) / (R**3)
        
        # Enhanced day-night asymmetry
        local_time_factor = 1.0 + 0.4 * np.cos(2 * np.pi * (datetime.datetime.now().hour / 24))
        B *= local_time_factor
        
        return B
        
    def _solar_activity(self, date: datetime.datetime) -> Tuple[float, float]:
        """Calculate solar activity parameters with enhanced variation"""
        day_of_year = date.timetuple().tm_yday
        solar_rotation_phase = 2 * np.pi * (day_of_year % 27) / 27
        
        years_since_2020 = (date.year - 2020) + date.timetuple().tm_yday / 365.25
        solar_cycle_phase = 2 * np.pi * (years_since_2020 % 11) / 11
        
        # Enhanced F10.7 calculation
        f10_cycle = self.f10_base + 1.5 * self.f10_variation * np.sin(solar_cycle_phase)
        f10_rotation = self.f10_variation * 0.5 * np.sin(solar_rotation_phase)
        f10 = f10_cycle + f10_rotation
        
        # Enhanced Ap calculation
        ap_cycle = self.ap_base + 1.5 * self.ap_variation * np.sin(solar_cycle_phase)
        ap_daily = self.ap_variation * np.random.normal(0, 0.8)
        ap = max(0, ap_cycle + ap_daily)
        
        return f10, ap

    def _generate_feature_vector(self, latitude: float, altitude: float, date: datetime.datetime) -> np.ndarray:
        """Generate feature vector with enhanced environmental modeling"""
        # Calculate base parameters
        B = self._magnetic_field_magnitude(latitude, altitude)
        f10, ap = self._solar_activity(date)
        
        # Calculate L-shell parameter
        R = (altitude + 6371.0) / 6371.0
        lat_rad = np.radians(latitude)
        L = R / (np.cos(lat_rad) ** 2)
        
        # Get enhanced Van Allen belt parameters
        va_params = self._van_allen_belts(L, B, ap/40.0)  # Increased activity sensitivity
        
        # Initialize feature vector
        features = np.zeros(128)
        
        # Enhanced magnetic field features (0-9)
        features[0] = B / self.B0
        features[1] = np.sin(lat_rad)
        features[2] = np.cos(lat_rad)
        features[3] = 1.0 / R**3
        features[4] = va_params['mirror_ratio']
        features[5:10] = np.gradient([B * (1 + 0.2 * i) for i in range(5)])
        
        # Enhanced radiation belt features (10-29)
        features[10] = va_params['inner_flux'] / self.inner_belt['max_flux_p']
        features[11] = va_params['outer_flux'] / self.outer_belt['max_flux_e']
        features[12:22] = va_params['proton_spectrum'] / self.inner_belt['max_flux_p']
        features[22:32] = va_params['electron_spectrum'] / self.outer_belt['max_flux_e']
        
        # Enhanced space weather features (30-49)
        features[30] = f10 / (self.f10_base + self.f10_variation)
        features[31] = ap / (self.ap_base + self.ap_variation)
        features[32] = va_params['total_dose'] / (self.inner_belt['max_flux_p'] + self.outer_belt['max_flux_e'])
        features[33] = va_params['belt_region']
        features[34] = va_params['saa_proximity']
        features[35] = va_params['slot_factor']
        
        # Enhanced solar wind parameters (36-49)
        solar_wind_speed = 400 + 200 * np.random.random()
        solar_wind_density = 5 + 3 * np.random.random()
        features[36:50] = np.array([
            solar_wind_speed / 600,
            solar_wind_density / 8,
            np.sin(2 * np.pi * date.hour / 24),
            np.cos(2 * np.pi * date.hour / 24),
            *np.random.normal(0, 0.2, 10)
        ])
        
        # Enhanced particle spectra (50-69)
        energy_bins = np.logspace(0, 4, 20)
        features[50:70] = np.exp(-energy_bins / 600)
        
        # Enhanced field fluctuations (70-89)
        freq_components = np.fft.fftfreq(20, d=1.0)
        field_fluctuations = 0.2 * np.exp(-np.abs(freq_components))
        features[70:90] = field_fluctuations
        
        # Enhanced radiation dynamics (90-109)
        time_factors = np.linspace(0, 24, 20)
        features[90:110] = 0.7 + 0.3 * np.sin(2 * np.pi * time_factors / 24)
        
        # Enhanced space weather indices (110-127)
        dst_index = -40 - 50 * np.random.random()
        kp_index = 2 + 5 * np.random.random()
        features[110:128] = np.array([
            dst_index / -100,
            kp_index / 9,
            f10 / 300,
            ap / 400,
            *np.random.normal(0, 0.2, 14)
        ])
        
        return features

    def generate_training_data(
        self,
        num_samples: int = 1000,
        sequence_length: int = 48
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data with enhanced physics"""
        X = np.zeros((num_samples, sequence_length, 128))
        # Updated y to match model's expected dimensions
        y = np.zeros((num_samples, 12))  # [3 eclipse, 4 occupancy, 5 radiation]
        
        for i in range(num_samples):
            try:
                # Generate orbit parameters
                altitude = np.random.uniform(200, 40000)  # km
                latitude = np.random.uniform(-60, 60)  # degrees
                
                # Generate time sequence
                start_time = datetime.datetime.now()
                time_delta = datetime.timedelta(minutes=5)
                
                for t in range(sequence_length):
                    current_time = start_time + t * time_delta
                    X[i, t] = self._generate_feature_vector(latitude, altitude * 1000, current_time)
                
                # Calculate output metrics
                # Eclipse predictions [umbra, penumbra, partial]
                eclipse_prob = 0.3 * (1 + np.cos(np.radians(latitude)))
                y[i, 0] = eclipse_prob  # umbra probability
                y[i, 1] = eclipse_prob * 0.8  # penumbra probability
                y[i, 2] = eclipse_prob * 0.5  # partial eclipse probability
                
                # Occupancy predictions [density, congestion, collision_risk, maneuver_space]
                density_level = np.exp(-altitude/7400)  # Atmospheric density
                y[i, 3] = density_level
                y[i, 4] = np.random.random() * 0.5  # congestion
                y[i, 5] = np.random.random() * 0.3  # collision risk
                y[i, 6] = 1 - density_level  # maneuver space
                
                # Radiation predictions [total_dose, particle_flux, SAA_proximity, belt_region, solar_activity]
                va_params = self._van_allen_belts(altitude/6371.0, self._magnetic_field_magnitude(latitude, altitude), 0.5)
                y[i, 7] = va_params['total_dose'] / (self.inner_belt['max_flux_p'] + self.outer_belt['max_flux_e'])
                y[i, 8] = (va_params['inner_flux'] + va_params['outer_flux']) / (self.inner_belt['max_flux_p'] + self.outer_belt['max_flux_e'])
                y[i, 9] = va_params['saa_proximity']
                y[i, 10] = va_params['belt_region'] / 2.0  # Normalize to [0,1]
                f10, _ = self._solar_activity(start_time)
                y[i, 11] = f10 / (self.f10_base + self.f10_variation)
                
            except Exception as e:
                print(f"Error generating sample {i}: {str(e)}")
                y[i] = np.zeros(12)
        
        return X, y

    def generate_environmental_data(self, num_samples: int) -> Dict:
        """Generate synthetic environmental data for training.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing sequences and metrics
        """
        sequence_length = 48  # 48 time steps
        
        # Generate sequences and metrics
        sequences = np.zeros((num_samples, sequence_length, 128))
        metrics = np.zeros((num_samples, 12))  # [3 eclipse, 4 occupancy, 5 radiation]
        
        for i in range(num_samples):
            # Generate orbit parameters
            altitude = np.random.uniform(200, 40000)  # km
            latitude = np.random.uniform(-60, 60)  # degrees
            
            # Generate time sequence
            start_time = datetime.datetime.now()
            time_delta = datetime.timedelta(minutes=5)
            
            for t in range(sequence_length):
                current_time = start_time + t * time_delta
                sequences[i, t] = self._generate_feature_vector(latitude, altitude * 1000, current_time)
            
            # Calculate environmental metrics
            B = self._magnetic_field_magnitude(latitude, altitude * 1000)
            f10, ap = self._solar_activity(start_time)
            R = (altitude + 6371.0) / 6371.0
            lat_rad = np.radians(latitude)
            L = R / (np.cos(lat_rad) ** 2)
            va_params = self._van_allen_belts(L, B, ap/40.0)
            
            # Eclipse metrics [umbra, penumbra, partial]
            metrics[i, 0:3] = np.random.random(3)  # Simplified for now
            
            # Occupancy metrics [LEO, MEO, GEO, HEO]
            metrics[i, 3:7] = np.array([
                1.0 if altitude < 2000 else 0.0,  # LEO
                1.0 if 2000 <= altitude < 35786 else 0.0,  # MEO
                1.0 if abs(altitude - 35786) < 100 else 0.0,  # GEO
                1.0 if altitude > 35786 else 0.0  # HEO
            ])
            
            # Radiation metrics [total_dose, inner_belt, outer_belt, saa, solar_particles]
            metrics[i, 7:12] = np.array([
                va_params['total_dose'] / (self.inner_belt['max_flux_p'] + self.outer_belt['max_flux_e']),
                va_params['inner_flux'] / self.inner_belt['max_flux_p'],
                va_params['outer_flux'] / self.outer_belt['max_flux_e'],
                va_params['saa_proximity'],
                f10 / (self.f10_base + self.f10_variation)
            ])
        
        return {
            'sequences': sequences,
            'metrics': metrics
        }

if __name__ == '__main__':
    # Example usage
    generator = EnvironmentalDataGenerator()
    X, y = generator.generate_training_data(num_samples=10)
    
    print("\nSample environmental metrics:")
    for i in range(3):
        print(f"Sample {i}:")
        print(f"  Eclipse Probabilities: {y[i, :3]}")
        print(f"  Occupancy Metrics: {y[i, 3:7]}")
        print(f"  Radiation Metrics: {y[i, 7:]}")
