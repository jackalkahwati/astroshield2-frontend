import numpy as np
from typing import Tuple, Dict, List
import datetime

class ComplianceDataGenerator:
    """Generate synthetic data for regulatory compliance evaluation"""
    
    def __init__(self):
        # Frequency allocation bands (MHz)
        self.frequency_bands = {
            'UHF': (300, 3000),
            'L': (1000, 2000),
            'S': (2000, 4000),
            'C': (4000, 8000),
            'X': (8000, 12000),
            'Ku': (12000, 18000),
            'Ka': (26500, 40000)
        }
        
        # ITU filing requirements
        self.itu_requirements = {
            'advance_publication': 7 * 365,  # Days before launch
            'coordination': 5 * 365,
            'notification': 3 * 365,
            'bringing_into_use': 7 * 365
        }
        
        # FCC licensing parameters
        self.fcc_parameters = {
            'application_window': 180,  # Days
            'modification_window': 90,
            'renewal_period': 15 * 365
        }
        
        # UN registry requirements
        self.un_requirements = {
            'registration_deadline': 60,  # Days after launch
            'required_fields': [
                'launch_date',
                'orbital_parameters',
                'general_function',
                'operator'
            ]
        }
        
        # Orbital slot allocations
        self.gso_slots = np.arange(-180, 180, 2)  # 2-degree spacing
        self.occupied_slots = np.random.choice(
            self.gso_slots,
            size=int(len(self.gso_slots) * 0.7),
            replace=False
        )

    def _generate_orbital_elements(self, orbit_type: str = 'LEO') -> Dict:
        """Generate orbital elements based on orbit type"""
        if orbit_type == 'GEO':
            a = 42164000  # Geostationary altitude
            e = np.random.normal(0, 0.0001)  # Near-circular
            i = np.random.normal(0, 0.1)  # Near-equatorial
            slot = np.random.choice(self.gso_slots)
        elif orbit_type == 'MEO':
            a = np.random.uniform(20000000, 35000000)
            e = np.random.uniform(0, 0.1)
            i = np.random.uniform(0, np.pi/3)
            slot = None
        else:  # LEO
            a = np.random.uniform(6800000, 8000000)
            e = np.random.uniform(0, 0.02)
            i = np.random.uniform(0, np.pi)
            slot = None
        
        return {
            'semi_major_axis': a,
            'eccentricity': e,
            'inclination': i,
            'raan': np.random.uniform(0, 2*np.pi),
            'arg_perigee': np.random.uniform(0, 2*np.pi),
            'mean_anomaly': np.random.uniform(0, 2*np.pi),
            'gso_slot': slot
        }

    def _generate_frequency_allocation(self, orbit_type: str) -> Dict:
        """Generate frequency allocation data"""
        # Select random frequency bands
        num_bands = np.random.randint(1, 4)
        selected_bands = np.random.choice(list(self.frequency_bands.keys()), size=num_bands, replace=False)
        
        allocations = {}
        for band in selected_bands:
            band_range = self.frequency_bands[band]
            # Uplink and downlink frequencies
            uplink = np.random.uniform(band_range[0], band_range[1])
            downlink = np.random.uniform(band_range[0], band_range[1])
            
            # Bandwidth
            bandwidth = np.random.uniform(10, 100)  # MHz
            
            allocations[band] = {
                'uplink': uplink,
                'downlink': downlink,
                'bandwidth': bandwidth,
                'polarization': np.random.choice(['linear', 'circular']),
                'power': np.random.uniform(0, 50)  # dBW
            }
        
        return allocations

    def _generate_registration_status(
        self,
        launch_date: datetime.datetime,
        frequency_data: Dict
    ) -> Dict:
        """Generate registration and licensing status"""
        now = datetime.datetime.now()
        
        # ITU filing status
        itu_status = {
            'advance_publication': {
                'submitted': np.random.random() > 0.1,
                'date': launch_date - datetime.timedelta(
                    days=np.random.uniform(
                        self.itu_requirements['advance_publication'],
                        self.itu_requirements['advance_publication'] + 365
                    )
                )
            },
            'coordination': {
                'submitted': np.random.random() > 0.15,
                'completed': np.random.random() > 0.2
            },
            'notification': {
                'submitted': np.random.random() > 0.1,
                'accepted': np.random.random() > 0.1
            }
        }
        
        # FCC license status
        fcc_status = {
            'application_submitted': np.random.random() > 0.1,
            'license_granted': np.random.random() > 0.15,
            'grant_date': launch_date - datetime.timedelta(
                days=np.random.uniform(180, 365)
            ),
            'expiration_date': launch_date + datetime.timedelta(
                days=self.fcc_parameters['renewal_period']
            )
        }
        
        # UN registry status
        un_status = {
            'submitted': np.random.random() > 0.1,
            'submission_date': launch_date + datetime.timedelta(
                days=np.random.uniform(0, self.un_requirements['registration_deadline'])
            ),
            'complete': np.random.random() > 0.1
        }
        
        return {
            'itu_status': itu_status,
            'fcc_status': fcc_status,
            'un_status': un_status
        }

    def _check_compliance(
        self,
        orbital_elements: Dict,
        frequency_data: Dict,
        registration: Dict
    ) -> Dict:
        """Check compliance with regulations"""
        violations = []
        compliance_score = 1.0
        
        # Check GSO slot compliance
        if orbital_elements.get('gso_slot') is not None:
            if orbital_elements['gso_slot'] in self.occupied_slots:
                violations.append('GSO_SLOT_VIOLATION')
                compliance_score -= 0.2
        
        # Check frequency interference
        for band, alloc in frequency_data.items():
            # Check band edges
            band_range = self.frequency_bands[band]
            if (alloc['uplink'] < band_range[0] or 
                alloc['uplink'] > band_range[1] or
                alloc['downlink'] < band_range[0] or
                alloc['downlink'] > band_range[1]):
                violations.append(f'FREQUENCY_BAND_VIOLATION_{band}')
                compliance_score -= 0.15
        
        # Check registration completeness
        if not registration['itu_status']['advance_publication']['submitted']:
            violations.append('ITU_ADVANCE_PUBLICATION_MISSING')
            compliance_score -= 0.25
        
        if not registration['fcc_status']['license_granted']:
            violations.append('FCC_LICENSE_MISSING')
            compliance_score -= 0.25
        
        if not registration['un_status']['submitted']:
            violations.append('UN_REGISTRY_MISSING')
            compliance_score -= 0.25
        
        return {
            'compliance_score': max(0, compliance_score),
            'violations': violations,
            'severity': len(violations) / 5  # Normalized by maximum possible violations
        }

    def generate_training_data(
        self,
        num_samples: int = 1000,
        sequence_length: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for compliance evaluation"""
        # Feature vector size: 15 components
        # [5 orbital elements, 4 frequency params, 6 registration flags]
        feature_size = 15
        
        # Initialize arrays
        X = np.zeros((num_samples, sequence_length, feature_size))
        y = np.zeros((num_samples, 3))  # [compliance_score, violation_count, severity]
        
        for i in range(num_samples):
            try:
                # Generate base launch date
                launch_date = datetime.datetime.now() - datetime.timedelta(
                    days=np.random.uniform(0, 365*2)
                )
                
                # Select orbit type
                orbit_type = np.random.choice(['LEO', 'MEO', 'GEO'], p=[0.7, 0.2, 0.1])
                
                # Generate compliance data
                orbital_elements = self._generate_orbital_elements(orbit_type)
                frequency_data = self._generate_frequency_allocation(orbit_type)
                registration = self._generate_registration_status(launch_date, frequency_data)
                
                # Check compliance
                compliance = self._check_compliance(
                    orbital_elements,
                    frequency_data,
                    registration
                )
                
                # Convert to feature vector
                features = []
                
                # Orbital features (5)
                features.extend([
                    orbital_elements['semi_major_axis'] / 42164000,  # Normalize by GEO radius
                    orbital_elements['eccentricity'],
                    orbital_elements['inclination'] / np.pi,
                    orbital_elements['raan'] / (2*np.pi),
                    orbital_elements['arg_perigee'] / (2*np.pi)
                ])
                
                # Frequency features (4) - use first band if multiple
                first_band = list(frequency_data.values())[0]
                features.extend([
                    first_band['uplink'] / 40000,    # Normalize by maximum frequency
                    first_band['downlink'] / 40000,
                    first_band['bandwidth'] / 100,
                    first_band['power'] / 50
                ])
                
                # Registration features (6)
                features.extend([
                    float(registration['itu_status']['advance_publication']['submitted']),
                    float(registration['itu_status']['coordination']['completed']),
                    float(registration['itu_status']['notification']['accepted']),
                    float(registration['fcc_status']['license_granted']),
                    float(registration['un_status']['submitted']),
                    float(registration['un_status']['complete'])
                ])
                
                # Store data
                X[i, 0] = features
                y[i] = [
                    compliance['compliance_score'],
                    len(compliance['violations']),
                    compliance['severity']
                ]
                
            except Exception as e:
                print(f"Error generating sample {i}: {str(e)}")
                # Fill with default values
                X[i, 0] = np.zeros(feature_size)
                y[i] = [0.0, 5.0, 1.0]
        
        return X, y

    def validate_data(self, features: np.ndarray) -> bool:
        """Validate generated data"""
        try:
            # Check feature bounds
            if not (0 <= features[0] <= 2):  # Normalized semi-major axis
                return False
            
            if not (0 <= features[1] <= 1):  # Eccentricity
                return False
            
            if not (0 <= features[2] <= 1):  # Normalized inclination
                return False
            
            if not (0 <= features[5] <= 1 and 0 <= features[6] <= 1):  # Normalized frequencies
                return False
            
            # Check binary features
            binary_indices = list(range(9, 15))  # Registration status features
            for idx in binary_indices:
                if features[idx] not in [0, 1]:
                    return False
            
            return True
            
        except Exception:
            return False

    def generate_compliance_data(self, num_samples: int) -> Dict:
        """Generate synthetic compliance data for training.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing sequences, rules, and scores
        """
        sequence_length = 1  # Single time step for compliance data
        feature_size = 15  # [5 orbital elements, 4 frequency params, 6 registration flags]
        
        # Initialize arrays
        sequences = np.zeros((num_samples, sequence_length, feature_size))
        rules = np.zeros((num_samples, 10))  # 10 different compliance rules
        scores = np.zeros((num_samples, 1))  # Overall compliance score
        
        for i in range(num_samples):
            # Generate base launch date
            launch_date = datetime.datetime.now() - datetime.timedelta(
                days=np.random.uniform(0, 365*2)
            )
            
            # Select orbit type
            orbit_type = np.random.choice(['LEO', 'MEO', 'GEO'], p=[0.7, 0.2, 0.1])
            
            # Generate compliance data
            orbital_elements = self._generate_orbital_elements(orbit_type)
            frequency_data = self._generate_frequency_allocation(orbit_type)
            registration = self._generate_registration_status(launch_date, frequency_data)
            
            # Check compliance
            compliance = self._check_compliance(
                orbital_elements,
                frequency_data,
                registration
            )
            
            # Convert to feature vector
            features = []
            
            # Orbital features (5)
            features.extend([
                orbital_elements['semi_major_axis'] / 42164000,  # Normalized by GEO radius
                orbital_elements['eccentricity'],
                orbital_elements['inclination'] / np.pi,
                orbital_elements['raan'] / (2*np.pi),
                orbital_elements['arg_perigee'] / (2*np.pi)
            ])
            
            # Frequency features (4)
            first_band = list(frequency_data.values())[0]
            features.extend([
                first_band['uplink'] / 40000,  # Normalized by max frequency
                first_band['downlink'] / 40000,
                first_band['bandwidth'] / 100,  # Normalized by max bandwidth
                first_band['power'] / 50  # Normalized by max power
            ])
            
            # Registration features (6)
            features.extend([
                float(registration['itu_status']['advance_publication']['submitted']),
                float(registration['itu_status']['coordination']['completed']),
                float(registration['itu_status']['notification']['accepted']),
                float(registration['fcc_status']['application_submitted']),
                float(registration['fcc_status']['license_granted']),
                float(registration['un_status']['submitted'])
            ])
            
            # Store features
            sequences[i, 0] = np.array(features)
            
            # Store compliance rules (binary flags for each rule)
            rules[i] = np.array([
                1.0 if 'GSO_SLOT_VIOLATION' not in compliance['violations'] else 0.0,
                1.0 if not any('FREQUENCY_BAND_VIOLATION' in v for v in compliance['violations']) else 0.0,
                1.0 if 'ITU_ADVANCE_PUBLICATION_MISSING' not in compliance['violations'] else 0.0,
                1.0 if 'FCC_LICENSE_MISSING' not in compliance['violations'] else 0.0,
                1.0 if 'UN_REGISTRY_MISSING' not in compliance['violations'] else 0.0,
                float(registration['itu_status']['coordination']['submitted']),
                float(registration['itu_status']['notification']['submitted']),
                float(registration['fcc_status']['application_submitted']),
                float(registration['un_status']['complete']),
                1.0 if compliance['severity'] < 0.5 else 0.0
            ])
            
            # Store compliance score
            scores[i] = compliance['compliance_score']
        
        return {
            'sequences': sequences,
            'rules': rules,
            'scores': scores
        }

if __name__ == '__main__':
    # Example usage and validation
    generator = ComplianceDataGenerator()
    
    # Generate sample data
    X, y = generator.generate_training_data(num_samples=10)
    
    # Validate data
    valid_samples = 0
    for sample in X:
        if generator.validate_data(sample[0]):
            valid_samples += 1
    
    print(f"Generated {len(X)} samples")
    print(f"Validation passed: {valid_samples}/{len(X)}")
    
    # Print sample metrics
    print("\nSample compliance metrics:")
    for i in range(3):
        print(f"Sample {i}:")
        print(f"  Compliance score: {y[i][0]:.3f}")
        print(f"  Violation count: {int(y[i][1])}")
        print(f"  Severity: {y[i][2]:.3f}")
