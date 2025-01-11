import numpy as np
from typing import Dict, List, Tuple
import datetime
from scipy import signal

class RFDataGenerator:
    """Generate synthetic RF data with realistic satellite communication patterns"""
    
    def __init__(self):
        # RF bands and frequencies (Hz)
        self.comm_bands = {
            'UHF': (300e6, 3e9),
            'S-band': (2e9, 4e9),
            'X-band': (8e9, 12e9),
            'Ka-band': (26.5e9, 40e9)
        }
        
        # Signal characteristics
        self.modulation_types = {
            'BPSK': {'bandwidth': 1.0, 'symbol_rate_range': (1e3, 1e6)},
            'QPSK': {'bandwidth': 2.0, 'symbol_rate_range': (1e4, 5e6)},
            'QAM16': {'bandwidth': 4.0, 'symbol_rate_range': (5e4, 10e6)}
        }
        
        # Power levels (dBW)
        self.power_levels = {
            'low': (-10, 0),
            'medium': (0, 10),
            'high': (10, 20)
        }
        
        # Pattern of Life characteristics
        self.typical_contact_duration = 600  # seconds
        self.typical_contact_interval = 5400  # seconds
        self.jitter_factor = 0.2  # 20% random variation
        
        # Anomaly parameters
        self.anomaly_types = {
            'frequency_drift': 0.1,    # MHz/s drift
            'power_surge': 3.0,        # dB increase
            'modulation_loss': 0.5,    # 50% symbol rate degradation
            'interference': 0.3,       # 30% noise increase
            'unauthorized_transmission': 0.8  # 80% power of normal
        }

    def _generate_base_signal(
        self,
        duration: float,
        sample_rate: float,
        frequency: float,
        power: float,
        modulation: str
    ) -> np.ndarray:
        """Generate base signal with specified characteristics"""
        t = np.arange(0, duration, 1/sample_rate)
        mod_params = self.modulation_types[modulation]
        symbol_rate = np.random.uniform(*mod_params['symbol_rate_range'])
        
        # Generate carrier
        carrier = np.sqrt(10**(power/10)) * np.sin(2 * np.pi * frequency * t)
        
        # Generate random symbols
        num_symbols = int(duration * symbol_rate)
        if modulation == 'BPSK':
            symbols = 2 * np.random.randint(0, 2, num_symbols) - 1
        elif modulation == 'QPSK':
            symbols = (np.random.randint(0, 4, num_symbols) * 2 * np.pi / 4) + np.pi/4
            symbols = np.exp(1j * symbols)
        else:  # QAM16
            symbols = (np.random.randint(0, 16, num_symbols) * 2 * np.pi / 16)
            symbols = np.exp(1j * symbols)
        
        # Upsample and shape symbols
        samples_per_symbol = int(sample_rate / symbol_rate)
        pulse = signal.rcosfilter(samples_per_symbol, 0.35, 1/symbol_rate, sample_rate)[1]
        shaped_symbols = np.convolve(np.repeat(symbols, samples_per_symbol), pulse, mode='same')
        
        return carrier * shaped_symbols.real

    def _add_anomaly(
        self,
        signal_data: np.ndarray,
        anomaly_type: str,
        severity: float
    ) -> Tuple[np.ndarray, Dict]:
        """Add specified anomaly to the signal"""
        modified_signal = signal_data.copy()
        anomaly_info = {
            'type': anomaly_type,
            'severity': severity,
            'detection_confidence': 0.0
        }
        
        if anomaly_type == 'frequency_drift':
            t = np.arange(len(modified_signal))
            drift = severity * self.anomaly_types['frequency_drift'] * t
            modified_signal *= np.cos(2 * np.pi * drift * t)
            anomaly_info['detection_confidence'] = min(severity * 0.8, 1.0)
            
        elif anomaly_type == 'power_surge':
            surge_factor = 1 + severity * self.anomaly_types['power_surge']
            modified_signal *= surge_factor
            anomaly_info['detection_confidence'] = min(severity * 0.9, 1.0)
            
        elif anomaly_type == 'modulation_loss':
            degradation = 1 - severity * self.anomaly_types['modulation_loss']
            modified_signal *= degradation
            modified_signal += np.random.normal(0, 0.1 * severity, len(modified_signal))
            anomaly_info['detection_confidence'] = min(severity * 0.7, 1.0)
            
        elif anomaly_type == 'interference':
            interference = severity * self.anomaly_types['interference']
            noise = np.random.normal(0, interference, len(modified_signal))
            modified_signal += noise
            anomaly_info['detection_confidence'] = min(severity * 0.85, 1.0)
            
        elif anomaly_type == 'unauthorized_transmission':
            unauth_power = severity * self.anomaly_types['unauthorized_transmission']
            unauthorized = np.random.normal(0, unauth_power, len(modified_signal))
            modified_signal += unauthorized
            anomaly_info['detection_confidence'] = min(severity * 0.95, 1.0)
        
        return modified_signal, anomaly_info

    def generate_contact_sequence(
        self,
        duration_hours: float = 24,
        sample_rate: float = 1000,
        base_pol: str = 'normal'
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Generate a sequence of RF contacts with optional anomalies"""
        total_samples = int(duration_hours * 3600 * sample_rate)
        signal_data = np.zeros(total_samples)
        contact_events = []
        
        # Select communication parameters
        band = np.random.choice(list(self.comm_bands.keys()))
        freq_range = self.comm_bands[band]
        frequency = np.random.uniform(*freq_range)
        modulation = np.random.choice(list(self.modulation_types.keys()))
        power_level = np.random.choice(list(self.power_levels.keys()))
        power_range = self.power_levels[power_level]
        power = np.random.uniform(*power_range)
        
        # Generate contact schedule
        current_time = 0
        while current_time < duration_hours * 3600:
            # Add jitter to contact timing
            contact_start = current_time + np.random.normal(
                0, self.typical_contact_interval * self.jitter_factor
            )
            contact_duration = self.typical_contact_duration * (
                1 + np.random.normal(0, self.jitter_factor)
            )
            
            if base_pol == 'normal':
                # Normal contact pattern
                start_idx = int(contact_start * sample_rate)
                duration_samples = int(contact_duration * sample_rate)
                if start_idx + duration_samples <= total_samples:
                    contact_signal = self._generate_base_signal(
                        contact_duration, sample_rate, frequency, power, modulation
                    )
                    signal_data[start_idx:start_idx + duration_samples] = contact_signal
                    
                    contact_events.append({
                        'start_time': contact_start,
                        'duration': contact_duration,
                        'frequency': frequency,
                        'power': power,
                        'modulation': modulation,
                        'band': band,
                        'type': 'normal'
                    })
            
            else:
                # Anomalous pattern
                if np.random.random() < 0.3:  # 30% chance of anomaly
                    anomaly_type = np.random.choice(list(self.anomaly_types.keys()))
                    severity = np.random.uniform(0.3, 1.0)
                    
                    start_idx = int(contact_start * sample_rate)
                    duration_samples = int(contact_duration * sample_rate)
                    if start_idx + duration_samples <= total_samples:
                        contact_signal = self._generate_base_signal(
                            contact_duration, sample_rate, frequency, power, modulation
                        )
                        modified_signal, anomaly_info = self._add_anomaly(
                            contact_signal, anomaly_type, severity
                        )
                        signal_data[start_idx:start_idx + duration_samples] = modified_signal
                        
                        contact_events.append({
                            'start_time': contact_start,
                            'duration': contact_duration,
                            'frequency': frequency,
                            'power': power,
                            'modulation': modulation,
                            'band': band,
                            'type': 'anomaly',
                            'anomaly_info': anomaly_info
                        })
            
            current_time = contact_start + contact_duration
        
        return signal_data, contact_events

    def generate_training_data(
        self,
        num_samples: int = 1000,
        sequence_length: int = 48,
        anomaly_probability: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for RF pattern analysis"""
        X = np.zeros((num_samples, sequence_length, 128))  # 128 features per timestep
        y = np.zeros((num_samples, 5))  # [is_anomaly, freq_conf, power_conf, mod_conf, pol_conf]
        
        for i in range(num_samples):
            # Determine if this sample should contain anomalies
            has_anomaly = np.random.random() < anomaly_probability
            base_pol = 'anomaly' if has_anomaly else 'normal'
            
            # Generate RF sequence
            signal_data, events = self.generate_contact_sequence(
                duration_hours=sequence_length/2,  # 30-minute intervals
                base_pol=base_pol
            )
            
            # Extract features
            for t in range(sequence_length):
                # Time-domain features
                X[i, t, 0:32] = signal_data[t*1000:(t+1)*1000:32]  # Downsample
                
                # Frequency-domain features
                freq_features = np.abs(np.fft.fft(signal_data[t*1000:(t+1)*1000]))
                X[i, t, 32:64] = freq_features[:32]
                
                # Statistical features
                X[i, t, 64:68] = [
                    np.mean(signal_data[t*1000:(t+1)*1000]),
                    np.std(signal_data[t*1000:(t+1)*1000]),
                    np.max(signal_data[t*1000:(t+1)*1000]),
                    np.min(signal_data[t*1000:(t+1)*1000])
                ]
                
                # Power spectral density
                f, psd = signal.welch(signal_data[t*1000:(t+1)*1000])
                X[i, t, 68:100] = psd[:32]
                
                # Modulation features
                X[i, t, 100:] = np.random.random(28)  # Placeholder for modulation features
            
            # Generate labels
            if has_anomaly:
                anomaly_events = [e for e in events if e['type'] == 'anomaly']
                if anomaly_events:
                    avg_confidence = np.mean([
                        e['anomaly_info']['detection_confidence']
                        for e in anomaly_events
                    ])
                    y[i] = [1.0, avg_confidence, avg_confidence, avg_confidence, avg_confidence]
            else:
                y[i] = [0.0, 0.9, 0.9, 0.9, 0.9]  # High confidence in normal behavior
        
        return X, y 