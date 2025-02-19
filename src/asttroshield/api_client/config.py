"""Configuration for the UDL API client and Supabase."""

# UDL API Configuration
UDL_API_CONFIG = {
    'base_url': 'https://unifieddatalibrary.com/udl',  # Base URL from the API docs
    'api_version': '1.30.0',  # Current API version (Daiquiri)
    'timeout': 30,  # Request timeout in seconds
}

# Space Weather Thresholds
SPACE_WEATHER_CONFIG = {
    'kp_index_threshold': 5,  # Geomagnetic storm threshold
    'radiation_belt_threshold': 3,  # Radiation belt activity threshold
    'solar_wind_speed_threshold': 500,  # km/s
    'check_interval': 300,  # Check every 5 minutes
}

# Conjunction Analysis
CONJUNCTION_CONFIG = {
    'warning_threshold_km': 10,  # Distance in km for conjunction warning
    'critical_threshold_km': 1,  # Distance in km for critical alert
    'probability_threshold': 0.0001,  # Probability threshold for collision
    'look_ahead_hours': 72,  # Hours to look ahead for conjunctions
}

# RF Interference
RF_CONFIG = {
    'interference_threshold_db': -70,  # dBm
    'frequency_ranges': {
        'uplink': {'min': 5925, 'max': 6425},  # MHz
        'downlink': {'min': 3700, 'max': 4200},  # MHz
    },
    'check_interval': 60,  # Check every minute
}

# Orbital Parameters
ORBITAL_CONFIG = {
    'position_uncertainty_threshold': 1000,  # meters
    'velocity_uncertainty_threshold': 10,  # m/s
    'update_interval': 300,  # Update every 5 minutes
}

# Maneuver Detection
MANEUVER_CONFIG = {
    'delta_v_threshold': 0.1,  # m/s
    'position_change_threshold': 100,  # meters
    'confidence_threshold': 0.9,  # Confidence level for maneuver detection
}

# Link Status
LINK_STATUS_CONFIG = {
    'signal_strength_threshold': -90,  # dBm
    'bit_error_rate_threshold': 1e-6,
    'latency_threshold': 500,  # ms
    'check_interval': 60,  # Check every minute
}

# Object Health Monitoring
HEALTH_CONFIG = {
    'battery_threshold': 0.2,  # 20% remaining
    'temperature_range': {
        'min': -40,  # Celsius
        'max': 85,  # Celsius
    },
    'check_interval': 300,  # Check every 5 minutes
}

# System Status Codes
STATUS_CODES = {
    # Object Status
    'OPERATIONAL': 'Object is functioning normally',
    'DEGRADED': 'Object is operating with reduced capabilities',
    'MAINTENANCE': 'Object is undergoing maintenance',
    'OFFLINE': 'Object is not responding',
    'ERROR': 'Object has encountered an error',
    
    # Conjunction Status
    'SAFE': 'No conjunction threats detected',
    'WARNING': 'Potential conjunction detected',
    'CRITICAL': 'High-risk conjunction detected',
    
    # Space Weather Status
    'QUIET': 'Space weather conditions are normal',
    'ACTIVE': 'Enhanced space weather activity',
    'STORM': 'Space weather storm in progress',
    
    # Communication Status
    'STRONG': 'Strong communication signal',
    'WEAK': 'Weak communication signal',
    'INTERFERENCE': 'Signal experiencing interference',
    'LOST': 'Communication signal lost'
}

# Alert Severity Levels
ALERT_LEVELS = {
    'INFO': 'Informational message',
    'WARNING': 'Potential issue detected',
    'CRITICAL': 'Immediate attention required',
    'EMERGENCY': 'Severe situation requiring immediate action'
}

# Batch Processing
BATCH_CONFIG = {
    'max_objects_per_request': 100,
    'max_concurrent_requests': 5,
    'request_timeout': 30,
}

# Sensor Configuration
SENSOR_CONFIG = {
    'heartbeat_interval': 60,  # Send heartbeat every 60 seconds
    'status_check_interval': 300,  # Check sensor status every 5 minutes
    'maintenance_check_interval': 3600,  # Check maintenance status every hour
}

# Notification Configuration
NOTIFICATION_CONFIG = {
    'default_severity': 'INFO',
    'alert_severity': 'ALERT',
    'error_severity': 'ERROR',
}

# Supabase Configuration
SUPABASE_CONFIG = {
    'url': 'YOUR_SUPABASE_URL',  # Set in environment
    'anon_key': 'YOUR_ANON_KEY',  # Set in environment
    'service_role_key': 'YOUR_SERVICE_ROLE_KEY',  # Set in environment
    'realtime': {
        'enabled': True,
        'channels': {
            'threat_indicators': '*',
            'sensor_status': '*',
            'alerts': '*'
        }
    },
    'storage': {
        'enabled': True,
        'buckets': {
            'sensor_data': 'sensor-data',
            'analysis_results': 'analysis-results'
        }
    }
}

# Database Tables Configuration
DB_TABLES = {
    'threat_indicators': {
        'name': 'threat_indicators',
        'schema': {
            'id': 'uuid',
            'type': 'text',
            'severity': 'text',
            'confidence': 'numeric',
            'metadata': 'jsonb',
            'created_at': 'timestamp with time zone',
            'updated_at': 'timestamp with time zone'
        },
        'rls_enabled': True
    },
    'sensor_status': {
        'name': 'sensor_status',
        'schema': {
            'id': 'uuid',
            'sensor_id': 'text',
            'status': 'text',
            'telemetry': 'jsonb',
            'last_update': 'timestamp with time zone'
        },
        'rls_enabled': True
    },
    'analysis_results': {
        'name': 'analysis_results',
        'schema': {
            'id': 'uuid',
            'indicator_id': 'uuid',
            'model_version': 'text',
            'results': 'jsonb',
            'created_at': 'timestamp with time zone'
        },
        'rls_enabled': True
    }
}

# Security Configuration
SECURITY_CONFIG = {
    'rls_policies': {
        'threat_indicators': [
            {
                'name': 'tenant_isolation',
                'using': '(auth.uid() = created_by)',
                'with_check': '(auth.uid() = created_by)'
            }
        ],
        'sensor_status': [
            {
                'name': 'sensor_access',
                'using': '(auth.uid() IN (SELECT user_id FROM sensor_access WHERE sensor_id = sensor_status.sensor_id))',
                'with_check': 'false'
            }
        ]
    }
}

# Cache Configuration
CACHE_CONFIG = {
    'enabled': True,
    'ttl': {
        'threat_indicators': 300,  # 5 minutes
        'sensor_status': 60,      # 1 minute
        'analysis_results': 3600   # 1 hour
    }
} 