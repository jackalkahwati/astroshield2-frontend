import numpy as np
from typing import Dict, List, Tuple
import datetime

class RegulatoryDataGenerator:
    """Generate synthetic data for regulatory compliance analysis"""
    
    def __init__(self):
        # ITU filing parameters
        self.itu_bands = {
            'C': (4e9, 8e9),
            'Ku': (12e9, 18e9),
            'Ka': (26.5e9, 40e9)
        }
        
        self.orbital_slots = {
            'GEO': {
                'altitude': 35786,  # km
                'inclination_max': 0.1,  # degrees
                'slot_width': 0.1,  # degrees
                'spacing': {
                    'C': 2.0,    # degrees
                    'Ku': 1.0,   # degrees
                    'Ka': 0.5    # degrees
                }
            },
            'LEO': {
                'altitude_range': (300, 1200),  # km
                'inclination_range': (0, 98),   # degrees
                'min_separation': 10  # km
            }
        }
        
        # FCC license parameters
        self.fcc_requirements = {
            'debris_mitigation': {
                'max_lifetime': 25,  # years
                'disposal_reliability': 0.9,
                'collision_risk': 0.001
            },
            'spectrum_usage': {
                'max_power_density': -150,  # dBW/m²
                'interference_threshold': -10,  # dB
                'out_of_band_emissions': -60  # dBc
            },
            'operational': {
                'telemetry_required': True,
                'maneuver_capability': True,
                'collision_avoidance': True
            }
        }
        
        # UN registry requirements
        self.un_registry_fields = {
            'required': [
                'launch_date',
                'launch_site',
                'basic_orbital_parameters',
                'general_function'
            ],
            'optional': [
                'operator',
                'spacecraft_mass',
                'power_source',
                'expected_lifetime'
            ]
        }
        
        # Common non-compliance patterns
        self.non_compliance_patterns = {
            'itu': {
                'frequency_violation': 0.3,
                'power_excess': 0.2,
                'orbital_slot_violation': 0.4,
                'coordination_failure': 0.1
            },
            'fcc': {
                'debris_violation': 0.25,
                'spectrum_violation': 0.35,
                'operational_violation': 0.4
            },
            'un': {
                'missing_required': 0.5,
                'incorrect_parameters': 0.3,
                'delayed_registration': 0.2
            }
        }

    def _check_itu_compliance(
        self,
        frequency: float,
        power: float,
        position: Dict[str, float],
        bandwidth: float
    ) -> Dict[str, float]:
        """Check compliance with ITU regulations"""
        violations = []
        compliance_score = 1.0
        
        # Check frequency allocation
        band_violation = True
        for band, (min_freq, max_freq) in self.itu_bands.items():
            if min_freq <= frequency <= max_freq:
                band_violation = False
                break
        if band_violation:
            violations.append('frequency_out_of_band')
            compliance_score *= 0.7
        
        # Check orbital slot compliance for GEO
        if abs(position['altitude'] - self.orbital_slots['GEO']['altitude']) < 100:
            if abs(position['inclination']) > self.orbital_slots['GEO']['inclination_max']:
                violations.append('inclination_violation')
                compliance_score *= 0.8
            
            # Check slot spacing
            for band, spacing in self.orbital_slots['GEO']['spacing'].items():
                if self.itu_bands[band][0] <= frequency <= self.itu_bands[band][1]:
                    if position.get('slot_separation', 0) < spacing:
                        violations.append('slot_spacing_violation')
                        compliance_score *= 0.6
        
        return {
            'compliant': len(violations) == 0,
            'score': compliance_score,
            'violations': violations
        }

    def _check_fcc_compliance(
        self,
        satellite_params: Dict,
        operations: Dict
    ) -> Dict[str, float]:
        """Check compliance with FCC regulations"""
        violations = []
        compliance_score = 1.0
        
        # Check debris mitigation compliance
        if satellite_params.get('lifetime', 0) > self.fcc_requirements['debris_mitigation']['max_lifetime']:
            violations.append('lifetime_violation')
            compliance_score *= 0.7
        
        if satellite_params.get('disposal_reliability', 0) < self.fcc_requirements['debris_mitigation']['disposal_reliability']:
            violations.append('disposal_reliability_violation')
            compliance_score *= 0.8
        
        # Check spectrum usage compliance
        if operations.get('power_density', 0) > self.fcc_requirements['spectrum_usage']['max_power_density']:
            violations.append('power_density_violation')
            compliance_score *= 0.6
        
        if operations.get('interference', 0) > self.fcc_requirements['spectrum_usage']['interference_threshold']:
            violations.append('interference_violation')
            compliance_score *= 0.7
        
        # Check operational requirements
        if not operations.get('has_telemetry', True):
            violations.append('telemetry_violation')
            compliance_score *= 0.8
        
        if not operations.get('has_maneuver', True):
            violations.append('maneuver_violation')
            compliance_score *= 0.7
        
        return {
            'compliant': len(violations) == 0,
            'score': compliance_score,
            'violations': violations
        }

    def _check_un_registry_compliance(
        self,
        registration: Dict
    ) -> Dict[str, float]:
        """Check compliance with UN registry requirements"""
        violations = []
        compliance_score = 1.0
        
        # Check required fields
        for field in self.un_registry_fields['required']:
            if field not in registration or not registration[field]:
                violations.append(f'missing_{field}')
                compliance_score *= 0.6
        
        # Check registration timing
        if registration.get('registration_delay', 0) > 60:  # More than 60 days
            violations.append('late_registration')
            compliance_score *= 0.9
        
        # Check parameter validity
        if 'basic_orbital_parameters' in registration:
            params = registration['basic_orbital_parameters']
            if not all(key in params for key in ['period', 'inclination', 'apogee', 'perigee']):
                violations.append('incomplete_orbital_parameters')
                compliance_score *= 0.8
        
        return {
            'compliant': len(violations) == 0,
            'score': compliance_score,
            'violations': violations
        }

    def generate_satellite_parameters(
        self,
        orbit_type: str = 'GEO'
    ) -> Dict:
        """Generate realistic satellite parameters"""
        if orbit_type == 'GEO':
            altitude = self.orbital_slots['GEO']['altitude']
            inclination = np.random.normal(0, 0.05)
        else:
            altitude = np.random.uniform(*self.orbital_slots['LEO']['altitude_range'])
            inclination = np.random.uniform(*self.orbital_slots['LEO']['inclination_range'])
        
        return {
            'altitude': altitude,
            'inclination': inclination,
            'frequency': np.random.choice([6e9, 14e9, 30e9]),  # Common frequencies
            'power': np.random.uniform(-10, 10),  # dBW
            'bandwidth': np.random.uniform(10e6, 500e6),  # Hz
            'lifetime': np.random.uniform(5, 30),  # years
            'disposal_reliability': np.random.uniform(0.8, 0.99),
            'has_telemetry': np.random.random() > 0.1,
            'has_maneuver': np.random.random() > 0.1
        }

    def generate_compliance_sequence(
        self,
        duration_days: int = 30,
        sample_rate: int = 24,  # hours
        anomaly_probability: float = 0.1
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate a sequence of compliance checks"""
        num_samples = duration_days * (24 // sample_rate)
        compliance_records = []
        anomaly_records = []
        
        # Generate base satellite parameters
        sat_params = self.generate_satellite_parameters()
        
        for t in range(num_samples):
            # Generate operational parameters
            operations = {
                'power_density': sat_params['power'] - 10 * np.log10(4 * np.pi * (sat_params['altitude'] * 1000)**2),
                'interference': np.random.normal(-20, 5),
                'has_telemetry': sat_params['has_telemetry'],
                'has_maneuver': sat_params['has_maneuver']
            }
            
            # Generate registration data
            registration = {
                'launch_date': '2023-01-01',
                'launch_site': 'Cape Canaveral',
                'basic_orbital_parameters': {
                    'period': 86400 if sat_params['altitude'] > 35000 else 5400,
                    'inclination': sat_params['inclination'],
                    'apogee': sat_params['altitude'],
                    'perigee': sat_params['altitude']
                },
                'general_function': 'Communications',
                'registration_delay': np.random.exponential(30)  # days
            }
            
            # Check compliance
            itu_check = self._check_itu_compliance(
                sat_params['frequency'],
                sat_params['power'],
                {'altitude': sat_params['altitude'], 'inclination': sat_params['inclination']},
                sat_params['bandwidth']
            )
            
            fcc_check = self._check_fcc_compliance(sat_params, operations)
            un_check = self._check_un_registry_compliance(registration)
            
            # Record compliance status
            compliance_record = {
                'timestamp': t * sample_rate,
                'itu_compliance': itu_check,
                'fcc_compliance': fcc_check,
                'un_compliance': un_check,
                'overall_compliance': all([
                    itu_check['compliant'],
                    fcc_check['compliant'],
                    un_check['compliant']
                ])
            }
            
            # Add anomalies
            if np.random.random() < anomaly_probability:
                anomaly_type = np.random.choice(['itu', 'fcc', 'un'])
                if anomaly_type == 'itu':
                    pattern = np.random.choice(
                        list(self.non_compliance_patterns['itu'].keys()),
                        p=list(self.non_compliance_patterns['itu'].values())
                    )
                    sat_params['power'] *= 1.5  # Power violation
                elif anomaly_type == 'fcc':
                    pattern = np.random.choice(
                        list(self.non_compliance_patterns['fcc'].keys()),
                        p=list(self.non_compliance_patterns['fcc'].values())
                    )
                    operations['interference'] *= 2  # Interference violation
                else:
                    pattern = np.random.choice(
                        list(self.non_compliance_patterns['un'].keys()),
                        p=list(self.non_compliance_patterns['un'].values())
                    )
                    registration['registration_delay'] = 90  # Registration delay
                
                anomaly_records.append({
                    'timestamp': t * sample_rate,
                    'type': anomaly_type,
                    'pattern': pattern,
                    'severity': np.random.uniform(0.5, 1.0)
                })
            
            compliance_records.append(compliance_record)
        
        return compliance_records, anomaly_records

    def generate_training_data(
        self,
        num_samples: int = 1000,
        sequence_length: int = 30,
        anomaly_probability: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for compliance analysis"""
        X = np.zeros((num_samples, sequence_length, 32))  # 32 features per timestep
        y = np.zeros((num_samples, 4))  # [itu_violation, fcc_violation, un_violation, confidence]
        
        for i in range(num_samples):
            # Generate compliance sequence
            compliance_records, anomaly_records = self.generate_compliance_sequence(
                duration_days=sequence_length,
                sample_rate=24,
                anomaly_probability=anomaly_probability
            )
            
            # Extract features
            for t in range(sequence_length):
                record = compliance_records[t]
                
                # Compliance scores
                X[i, t, 0:3] = [
                    record['itu_compliance']['score'],
                    record['fcc_compliance']['score'],
                    record['un_compliance']['score']
                ]
                
                # Violation counts
                X[i, t, 3:6] = [
                    len(record['itu_compliance']['violations']),
                    len(record['fcc_compliance']['violations']),
                    len(record['un_compliance']['violations'])
                ]
                
                # Additional features (placeholder)
                X[i, t, 6:] = np.random.random(26)
            
            # Generate labels
            itu_violations = any(
                not record['itu_compliance']['compliant']
                for record in compliance_records
            )
            fcc_violations = any(
                not record['fcc_compliance']['compliant']
                for record in compliance_records
            )
            un_violations = any(
                not record['un_compliance']['compliant']
                for record in compliance_records
            )
            
            # Calculate confidence based on violation severity
            confidence = 0.0
            if anomaly_records:
                confidence = np.mean([record.get('severity', 0.5) for record in anomaly_records])
            else:
                confidence = 0.9  # High confidence in compliance
            
            y[i] = [
                float(itu_violations),
                float(fcc_violations),
                float(un_violations),
                confidence
            ]
        
        return X, y 