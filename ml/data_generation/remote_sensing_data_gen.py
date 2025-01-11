import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy.stats import multivariate_normal

class RemoteSensingDataGenerator:
    """Enhanced generator for synthetic remote sensing data"""
    
    def __init__(
        self,
        sequence_length: int = 100,
        feature_dim: int = 64,
        time_step: float = 1.0  # seconds
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.time_step = time_step
        
        # Define sensor characteristics
        self.optical_sensors = {
            'visible': {
                'wavelength_range': (0.4, 0.7),  # micrometers
                'resolution': 0.3,  # meters
                'snr': 100.0,  # signal-to-noise ratio
                'fov': 1.0,  # degrees
                'integration_time': 0.001  # seconds
            },
            'infrared': {
                'wavelength_range': (8.0, 12.0),
                'resolution': 1.0,
                'snr': 80.0,
                'fov': 2.0,
                'integration_time': 0.01
            }
        }
        
        self.sar_sensors = {
            'x_band': {
                'frequency': 10.0,  # GHz
                'bandwidth': 300.0,  # MHz
                'resolution': 1.0,  # meters
                'snr': 15.0,  # dB
                'swath_width': 10.0  # km
            },
            'c_band': {
                'frequency': 5.0,
                'bandwidth': 100.0,
                'resolution': 3.0,
                'snr': 12.0,
                'swath_width': 30.0
            }
        }
        
        self.hyperspectral_sensors = {
            'vnir': {
                'bands': np.linspace(0.4, 1.0, 100),  # micrometers
                'resolution': 2.0,  # meters
                'snr': 200.0,
                'fov': 5.0,
                'integration_time': 0.05
            },
            'swir': {
                'bands': np.linspace(1.0, 2.5, 100),
                'resolution': 3.0,
                'snr': 150.0,
                'fov': 5.0,
                'integration_time': 0.1
            }
        }
        
        # Define target characteristics
        self.target_types = {
            'satellite': {
                'size_range': (1.0, 10.0),  # meters
                'albedo_range': (0.1, 0.9),
                'temperature_range': (250, 350),  # Kelvin
                'material_types': ['aluminum', 'solar_cells', 'mlid']
            },
            'debris': {
                'size_range': (0.1, 1.0),
                'albedo_range': (0.05, 0.3),
                'temperature_range': (200, 300),
                'material_types': ['aluminum', 'paint', 'composite']
            },
            'rocket_body': {
                'size_range': (3.0, 20.0),
                'albedo_range': (0.2, 0.6),
                'temperature_range': (270, 320),
                'material_types': ['aluminum', 'steel', 'composite']
            }
        }
    
    def generate_optical_signature(
        self,
        target_type: str,
        sensor_type: str,
        range_km: float
    ) -> np.ndarray:
        """Generate optical signature for a target"""
        target = self.target_types[target_type]
        sensor = self.optical_sensors[sensor_type]
        
        # Calculate target size and albedo
        size = np.random.uniform(*target['size_range'])
        albedo = np.random.uniform(*target['albedo_range'])
        
        # Calculate apparent magnitude
        apparent_size = size / (range_km * 1000)  # angular size in radians
        
        # Calculate signal strength with atmospheric attenuation
        atm_transmittance = np.exp(-range_km / 100.0)  # simplified atmospheric model
        signal = albedo * apparent_size * atm_transmittance
        
        # Add noise based on SNR
        noise_sigma = signal / sensor['snr']
        noise = np.random.normal(0, noise_sigma)
        
        # Generate spectral signature
        num_bands = 10  # number of spectral bands
        spectral_signature = np.zeros(num_bands)
        
        for i in range(num_bands):
            wavelength = np.linspace(
                sensor['wavelength_range'][0],
                sensor['wavelength_range'][1],
                num_bands
            )[i]
            
            # Simulate wavelength-dependent reflectance
            reflectance = albedo * (1 + 0.1 * np.sin(2 * np.pi * wavelength))
            spectral_signature[i] = signal * reflectance + noise
        
        return spectral_signature
    
    def generate_sar_signature(
        self,
        target_type: str,
        sensor_type: str,
        range_km: float
    ) -> np.ndarray:
        """Generate SAR signature for a target"""
        target = self.target_types[target_type]
        sensor = self.sar_sensors[sensor_type]
        
        # Calculate RCS based on target size and material
        size = np.random.uniform(*target['size_range'])
        base_rcs = np.pi * (size/2)**2  # simplified RCS model
        
        # Add material-dependent scaling
        material = np.random.choice(target['material_types'])
        material_factors = {
            'aluminum': 1.0,
            'steel': 1.2,
            'composite': 0.7,
            'solar_cells': 0.8,
            'paint': 0.5,
            'mlid': 0.9
        }
        rcs = base_rcs * material_factors[material]
        
        # Calculate received power using radar equation
        wavelength = 0.3 / sensor['frequency']  # wavelength in meters
        tx_power = 1000.0  # transmit power in watts
        antenna_gain = 30.0  # antenna gain in dB
        
        received_power = (
            tx_power * antenna_gain**2 * wavelength**2 * rcs /
            ((4*np.pi)**3 * range_km**4)
        )
        
        # Add noise based on SNR
        noise_power = received_power / (10**(sensor['snr']/10))
        noise = np.random.normal(0, np.sqrt(noise_power))
        
        # Generate range-Doppler signature
        num_range_bins = 10
        num_doppler_bins = 10
        signature = np.zeros((num_range_bins, num_doppler_bins))
        
        # Simulate range-Doppler response
        range_bin = int(range_km * num_range_bins / sensor['swath_width'])
        doppler_bin = np.random.randint(0, num_doppler_bins)
        
        if 0 <= range_bin < num_range_bins:
            # Add target response with speckle
            speckle = np.random.rayleigh(1.0)
            signature[range_bin, doppler_bin] = received_power * speckle + noise
        
        return signature.flatten()
    
    def generate_hyperspectral_signature(
        self,
        target_type: str,
        sensor_type: str,
        range_km: float
    ) -> np.ndarray:
        """Generate hyperspectral signature for a target"""
        target = self.target_types[target_type]
        sensor = self.hyperspectral_sensors[sensor_type]
        
        # Get target parameters
        size = np.random.uniform(*target['size_range'])
        temperature = np.random.uniform(*target['temperature_range'])
        material = np.random.choice(target['material_types'])
        
        # Generate spectral signature across all bands
        num_bands = len(sensor['bands'])
        signature = np.zeros(num_bands)
        
        for i, wavelength in enumerate(sensor['bands']):
            # Simulate material-specific spectral features
            if material == 'aluminum':
                reflectance = 0.9 - 0.1 * np.exp(-(wavelength - 0.8)**2 / 0.1)
            elif material == 'solar_cells':
                reflectance = 0.3 + 0.4 * (wavelength > 1.0)
            else:
                reflectance = 0.5 + 0.2 * np.sin(2 * np.pi * wavelength)
            
            # Add thermal emission (simplified blackbody)
            emissivity = 1.0 - reflectance
            thermal = emissivity * (wavelength * temperature / 3000)**(-4)
            
            # Calculate signal with atmospheric attenuation
            atm_transmittance = np.exp(-range_km / (100.0 * wavelength))
            signal = (reflectance + thermal) * size * atm_transmittance
            
            # Add noise based on SNR
            noise_sigma = signal / sensor['snr']
            noise = np.random.normal(0, noise_sigma)
            
            signature[i] = signal + noise
        
        return signature
    
    def generate_features(
        self,
        optical_sig: np.ndarray,
        sar_sig: np.ndarray,
        hsi_sig: np.ndarray,
        range_km: float
    ) -> np.ndarray:
        """Generate feature vector from signatures"""
        # Initialize feature vector
        num_samples = len(optical_sig)
        feature_dim = 30  # Fixed feature dimension (must be divisible by 3)
        features = np.zeros((num_samples, feature_dim))
        
        # Ensure all signatures have the same shape
        optical_sig = optical_sig.reshape(num_samples, -1)
        sar_sig = sar_sig.reshape(num_samples, -1)
        hsi_sig = hsi_sig.reshape(num_samples, -1)
        
        # Normalize signatures to fixed size
        feature_per_type = feature_dim // 3  # Equal size for each feature type
        optical_features = self._normalize_signature(optical_sig, feature_per_type)
        sar_features = self._normalize_signature(sar_sig, feature_per_type)
        hsi_features = self._normalize_signature(hsi_sig, feature_per_type)
        
        # Combine features
        features[:, :feature_per_type] = optical_features
        features[:, feature_per_type:2*feature_per_type] = sar_features
        features[:, 2*feature_per_type:] = hsi_features
        
        # Add range information by scaling features
        range_scale = np.exp(-range_km / 1000.0)  # Exponential decay with range
        features *= range_scale
        
        return features
    
    def _normalize_signature(
        self,
        signature: np.ndarray,
        target_size: int
    ) -> np.ndarray:
        """Normalize signature to fixed size"""
        num_samples = signature.shape[0]
        signature_normalized = np.zeros((num_samples, target_size))
        
        for i in range(num_samples):
            # Get original signature
            sig = signature[i]
            
            # Create interpolation points
            x_old = np.linspace(0, 1, len(sig))
            x_new = np.linspace(0, 1, target_size)
            
            # Interpolate
            signature_normalized[i] = np.interp(x_new, x_old, sig)
        
        # Normalize to [0, 1] range
        signature_normalized = (signature_normalized - np.min(signature_normalized)) / (
            np.max(signature_normalized) - np.min(signature_normalized) + 1e-10
        )
        
        return signature_normalized
    
    def generate_labels(
        self,
        target_type: str,
        range_km: float,
        signatures: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Generate labels for the observation"""
        # One-hot encode target type
        target_types = list(self.target_types.keys())
        target_onehot = np.zeros(len(target_types))
        target_onehot[target_types.index(target_type)] = 1.0
        
        # Calculate detection confidence based on range and signatures
        optical_snr = np.max(signatures['optical']) / np.std(signatures['optical'])
        sar_snr = np.max(signatures['sar']) / np.std(signatures['sar'])
        hsi_snr = np.max(signatures['hyperspectral']) / np.std(signatures['hyperspectral'])
        
        detection_conf = np.mean([optical_snr, sar_snr, hsi_snr]) / (1 + range_km/100)
        
        # Combine labels
        labels = np.concatenate([
            target_onehot,
            [detection_conf],
            [range_km]
        ])
        
        return labels
    
    def generate_data(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic remote sensing data"""
        features_list = []
        labels_list = []
        
        for _ in range(num_samples):
            # Select random target type and range
            target_type = np.random.choice(list(self.target_types.keys()))
            range_km = np.random.uniform(10.0, 1000.0)
            
            # Generate signatures from each sensor type
            optical_sig = self.generate_optical_signature(
                target_type, 'visible', range_km
            )
            
            sar_sig = self.generate_sar_signature(
                target_type, 'x_band', range_km
            )
            
            hsi_sig = self.generate_hyperspectral_signature(
                target_type, 'vnir', range_km
            )
            
            # Combine signatures into features
            features = self.generate_features(
                optical_sig, sar_sig, hsi_sig, range_km
            )
            features_list.append(features)
            
            # Generate labels
            signatures = {
                'optical': optical_sig,
                'sar': sar_sig,
                'hyperspectral': hsi_sig
            }
            labels = self.generate_labels(target_type, range_km, signatures)
            labels_list.append(labels)
        
        # Convert to tensors
        features = torch.tensor(np.stack(features_list), dtype=torch.float32)
        labels = torch.tensor(np.stack(labels_list), dtype=torch.float32)
        
        return {
            'features': features,
            'labels': labels
        } 