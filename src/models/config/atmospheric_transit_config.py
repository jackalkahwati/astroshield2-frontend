"""Configuration for atmospheric transit detection."""

# Altitude range for detection (km)
ALTITUDE_CONFIG = {
    'min': 30.0,
    'max': 300.0
}

# Velocity thresholds (km/s)
VELOCITY_CONFIG = {
    'space_capable': 7.8,  # Minimum orbital velocity
    'hypersonic': 5.0,    # Minimum hypersonic velocity
    'supersonic': 1.0     # Minimum supersonic velocity
}

# Ionospheric analysis configuration
IONOSPHERIC_CONFIG = {
    'tec_window_size': 300,  # seconds
    'tec_threshold': 0.5,    # TECU
    'min_perturbation_duration': 10,  # seconds
    'max_perturbation_duration': 300,  # seconds
    'confidence_threshold': 0.7
}

# Magnetic field analysis configuration
MAGNETIC_CONFIG = {
    'window_size': 300,  # seconds
    'field_threshold': 50.0,  # nT
    'min_disturbance_duration': 10,  # seconds
    'max_disturbance_duration': 300,  # seconds
    'confidence_threshold': 0.7
}

# SDR analysis configuration
SDR_CONFIG = {
    'frequency_bands': [
        {
            'name': 'VHF',
            'min_freq': 30e6,
            'max_freq': 300e6
        },
        {
            'name': 'UHF',
            'min_freq': 300e6,
            'max_freq': 3e9
        }
    ],
    'doppler_threshold': 100.0,  # Hz
    'power_threshold': -70.0,    # dBm
    'min_track_duration': 5,     # seconds
    'max_track_duration': 300,   # seconds
    'confidence_threshold': 0.7
}

# Correlation configuration
CORRELATION_CONFIG = {
    'max_time_difference': 10.0,  # seconds
    'max_position_difference': 50.0,  # km
    'min_correlation_score': 0.7
}

# Impact prediction configuration
IMPACT_CONFIG = {
    'prediction_interval': 1.0,  # seconds
    'max_prediction_time': 3600.0,  # seconds
    'min_confidence': 0.8,
    'atmospheric_model': {
        'density_model': 'NRLMSISE-00',
        'wind_model': 'HWM14'
    }
}

# Detection confidence thresholds
CONFIDENCE_CONFIG = {
    'min_detection': 0.7,
    'min_tracking': 0.8,
    'min_impact_prediction': 0.9
}

# Time window configuration
TIME_CONFIG = {
    'window_size': 300,  # seconds
    'overlap': 30,      # seconds
    'max_gap': 60       # seconds
}

# Environmental factors
ENVIRONMENTAL_CONFIG = {
    'max_solar_kp': 4.0,
    'max_radiation_belt_activity': 3.0,
    'max_ionospheric_disturbance': 2.0
}

# Combined detector configuration
DETECTOR_CONFIG = {
    'altitude_range': ALTITUDE_CONFIG,
    'velocity_threshold': VELOCITY_CONFIG['space_capable'],
    'confidence_threshold': CONFIDENCE_CONFIG['min_detection'],
    'ionospheric': IONOSPHERIC_CONFIG,
    'magnetic': MAGNETIC_CONFIG,
    'sdr': SDR_CONFIG,
    'correlation': CORRELATION_CONFIG,
    'impact': IMPACT_CONFIG,
    'time': TIME_CONFIG,
    'environmental': ENVIRONMENTAL_CONFIG
} 