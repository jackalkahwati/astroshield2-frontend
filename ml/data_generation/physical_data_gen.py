import numpy as np
from typing import Dict, List, Tuple
import datetime
from scipy.spatial.transform import Rotation

class PhysicalPropertiesGenerator:
    """Generate synthetic physical properties data for space objects"""
    
    def __init__(self):
        # Object type characteristics
        self.object_types = {
            'rocket_body': {
                'mass_range': (1000, 5000),  # kg
                'length_range': (10, 30),     # meters
                'diameter_range': (2, 5),     # meters
                'amr_range': (0.005, 0.02),   # m²/kg
                'tumbling_prob': 0.7
            },
            'active_satellite': {
                'mass_range': (100, 1000),    # Reduced max mass for better AMR
                'size_range': (1, 3),         # Reduced size range
                'solar_array_area_range': (20, 100),  # Increased solar array area
                'amr_range': (0.01, 0.1),
                'tumbling_prob': 0.1
            },
            'debris': {
                'mass_range': (0.1, 100),
                'size_range': (0.1, 2),
                'amr_range': (0.02, 0.2),
                'tumbling_prob': 0.9
            },
            'cubesat': {
                'mass_range': (1, 10),
                'unit_size': 0.1,  # 10cm per unit
                'units_range': (1, 12),
                'amr_range': (0.015, 0.15),
                'tumbling_prob': 0.3
            }
        }
        
        # Deployment states
        self.deployment_states = {
            'folded': 0.3,    # Reduction factor for area
            'partial': 0.7,   # Reduction factor for area
            'deployed': 1.0   # Full area
        }
        
        # Material properties
        self.materials = {
            'aluminum': {
                'density': 2700,  # kg/m³
                'reflectivity': 0.9,
                'thermal_expansion': 23.1e-6  # per °C
            },
            'titanium': {
                'density': 4500,
                'reflectivity': 0.6,
                'thermal_expansion': 8.6e-6
            },
            'solar_cell': {
                'density': 2330,  # Silicon density
                'reflectivity': 0.3,
                'thermal_expansion': 3.0e-6
            },
            'mli': {  # Multi-Layer Insulation
                'density': 100,
                'reflectivity': 0.8,
                'thermal_expansion': 50.0e-6
            }
        }
        
        # Thermal conditions
        self.thermal_ranges = {
            'sunlit': (250, 350),  # Kelvin
            'eclipse': (200, 250),
            'deep_space': (100, 150)
        }

    def _calculate_amr(
        self,
        mass: float,
        projected_area: float,
        deployment_state: str
    ) -> float:
        """Calculate Area-to-Mass Ratio"""
        effective_area = projected_area * self.deployment_states[deployment_state]
        return effective_area / mass

    def _calculate_projected_area(
        self,
        dimensions: Dict[str, float],
        attitude: np.ndarray
    ) -> float:
        """Calculate projected area based on attitude"""
        # Create simplified box model
        length = dimensions.get('length', dimensions.get('size', 1.0))
        width = dimensions.get('width', dimensions.get('size', 1.0))
        height = dimensions.get('height', dimensions.get('size', 1.0))
        
        # Calculate base projected areas for each face
        areas = np.array([
            length * width,
            length * height,
            width * height
        ])
        
        # Calculate projection factors based on attitude
        rot = Rotation.from_euler('xyz', attitude)
        normal_vectors = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        
        # Rotate normal vectors
        rotated_normals = rot.apply(normal_vectors)
        
        # Calculate projection factors
        view_vector = np.array([0, 0, 1])  # Assuming viewing from +Z
        projection_factors = np.abs(np.dot(rotated_normals, view_vector))
        
        # Calculate base projected area
        base_area = np.sum(areas * projection_factors)
        
        # Add solar array area if present with optimal orientation
        if 'solar_array_area' in dimensions:
            # Solar arrays typically track the sun for maximum exposure
            # We'll assume they're oriented for maximum projection
            solar_array_area = dimensions['solar_array_area']
            base_area += solar_array_area  # Add full solar array area
            
        return base_area

    def _calculate_thermal_effects(
        self,
        base_dimensions: Dict[str, float],
        material: str,
        temperature: float
    ) -> Dict[str, float]:
        """Calculate thermal expansion effects"""
        material_props = self.materials[material]
        delta_t = temperature - 293.15  # Reference temperature (20°C)
        
        # Calculate dimensional changes
        expansion_factor = 1 + material_props['thermal_expansion'] * delta_t
        
        expanded_dimensions = {
            key: value * expansion_factor
            for key, value in base_dimensions.items()
        }
        
        return {
            'expansion_factor': expansion_factor,
            'dimensions': expanded_dimensions
        }

    def generate_object_properties(
        self,
        object_type: str,
        time: datetime.datetime = None
    ) -> Dict:
        """Generate physical properties for a space object"""
        config = self.object_types[object_type]
        
        # Generate base properties
        mass = np.random.uniform(*config['mass_range'])
        
        # Generate dimensions based on object type
        if object_type == 'rocket_body':
            dimensions = {
                'length': np.random.uniform(*config['length_range']),
                'diameter': np.random.uniform(*config['diameter_range'])
            }
        elif object_type == 'cubesat':
            units = np.random.randint(*config['units_range'])
            dimensions = {
                'size': config['unit_size'] * units
            }
        else:
            size = np.random.uniform(*config['size_range'])
            dimensions = {'size': size}
        
        # Add solar arrays for active satellites
        if object_type == 'active_satellite':
            dimensions['solar_array_area'] = np.random.uniform(
                *config['solar_array_area_range']
            )
        
        # Generate attitude
        is_tumbling = np.random.random() < config['tumbling_prob']
        if is_tumbling:
            attitude = np.random.uniform(0, 360, 3)  # Random orientation
            angular_velocity = np.random.uniform(-5, 5, 3)  # deg/s
        else:
            attitude = np.zeros(3)  # Stable orientation
            angular_velocity = np.zeros(3)
        
        # Select deployment state
        deployment_state = np.random.choice(
            list(self.deployment_states.keys()),
            p=[0.2, 0.3, 0.5]  # Bias towards deployed state
        )
        
        # Calculate projected area and AMR
        projected_area = self._calculate_projected_area(dimensions, attitude)
        amr = self._calculate_amr(mass, projected_area, deployment_state)
        
        # Add thermal effects
        if time:
            # Determine illumination state
            hour = time.hour
            if 6 <= hour < 18:  # Daytime
                temp_range = self.thermal_ranges['sunlit']
            else:  # Nighttime
                temp_range = self.thermal_ranges['eclipse']
        else:
            temp_range = self.thermal_ranges['sunlit']
        
        temperature = np.random.uniform(*temp_range)
        thermal_effects = self._calculate_thermal_effects(
            dimensions,
            'aluminum',  # Default material
            temperature
        )
        
        return {
            'mass': mass,
            'dimensions': dimensions,
            'attitude': attitude,
            'angular_velocity': angular_velocity,
            'deployment_state': deployment_state,
            'projected_area': projected_area,
            'amr': amr,
            'temperature': temperature,
            'thermal_effects': thermal_effects,
            'is_tumbling': is_tumbling
        }

    def generate_property_sequence(
        self,
        object_type: str,
        duration_hours: float = 24,
        sample_rate: float = 60,  # seconds
        anomaly_probability: float = 0.1
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate a sequence of physical property measurements"""
        num_samples = int(duration_hours * 3600 / sample_rate)
        property_records = []
        anomaly_records = []
        
        # Generate base object
        base_properties = self.generate_object_properties(object_type)
        
        for t in range(num_samples):
            time = datetime.datetime.now() + datetime.timedelta(seconds=t*sample_rate)
            
            # Update attitude based on angular velocity
            if base_properties['is_tumbling']:
                base_properties['attitude'] += base_properties['angular_velocity'] * sample_rate
                base_properties['attitude'] %= 360
            
            # Generate properties with current state
            properties = self.generate_object_properties(
                object_type,
                time=time
            )
            
            # Add measurement noise
            properties['amr'] *= 1 + np.random.normal(0, 0.05)  # 5% measurement noise
            properties['projected_area'] *= 1 + np.random.normal(0, 0.03)  # 3% noise
            
            # Add anomalies
            if np.random.random() < anomaly_probability:
                anomaly_type = np.random.choice([
                    'deployment_failure',
                    'tumbling_onset',
                    'thermal_anomaly',
                    'fragmentation'
                ])
                
                if anomaly_type == 'deployment_failure':
                    properties['deployment_state'] = 'folded'
                    properties['amr'] *= 0.3
                elif anomaly_type == 'tumbling_onset':
                    properties['is_tumbling'] = True
                    properties['angular_velocity'] = np.random.uniform(-10, 10, 3)
                elif anomaly_type == 'thermal_anomaly':
                    properties['temperature'] += 50  # Sudden temperature increase
                elif anomaly_type == 'fragmentation':
                    properties['mass'] *= 0.8  # Loss of mass
                    properties['amr'] *= 1.5  # Increased AMR
                
                anomaly_records.append({
                    'timestamp': t * sample_rate,
                    'type': anomaly_type,
                    'severity': np.random.uniform(0.5, 1.0)
                })
            
            property_records.append(properties)
        
        return property_records, anomaly_records

    def generate_training_data(
        self,
        num_samples: int = 10000,
        sequence_length: int = 48
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for physical property analysis.
        
        Args:
            num_samples: Number of samples to generate
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (features, labels)
        """
        # Initialize arrays
        X = np.zeros((num_samples, sequence_length, 128))  # 128 features per timestep
        y = np.zeros((num_samples, 22))  # [4 type, 3 dim, 7 attitude, 4 material, 3 thermal, 1 confidence]
        
        for i in range(num_samples):
            try:
                # Select random object type
                object_type = np.random.choice(list(self.object_types.keys()))
                
                # Generate property sequence
                property_records, anomaly_records = self.generate_property_sequence(
                    object_type,
                    duration_hours=sequence_length/2,  # 30-minute sequences
                    sample_rate=60.0  # 1-minute samples
                )
                
                # Generate feature vector for each timestep
                for t in range(sequence_length):
                    record = property_records[t]
                    
                    # Extract dimensions
                    if object_type == 'rocket_body':
                        dimensions = np.array([
                            record['dimensions']['length'],
                            record['dimensions']['diameter'],
                            record['dimensions']['diameter']
                        ])
                    elif object_type == 'cubesat':
                        size = record['dimensions']['size']
                        dimensions = np.array([size, size, size])
                    else:
                        size = record['dimensions']['size']
                        dimensions = np.array([size, size, size])
                    
                    # Add solar array area for active satellites
                    if object_type == 'active_satellite':
                        solar_array_area = record['dimensions'].get('solar_array_area', 0.0)
                        dimensions = np.append(dimensions, solar_array_area)
                    else:
                        dimensions = np.append(dimensions, 0.0)
                    
                    # Material properties
                    material_props = np.array([
                        self.materials['aluminum']['density'],
                        self.materials['aluminum']['reflectivity'],
                        self.materials['aluminum']['thermal_expansion']
                    ])
                    
                    # Combine all features
                    X[i, t] = np.concatenate([
                        np.array([record['mass']]),
                        dimensions,
                        record['attitude'],
                        record['angular_velocity'],
                        np.array([float(record['is_tumbling'])]),
                        np.array([record['amr']]),
                        material_props,
                        np.array([record['temperature']]),
                        np.array([record['thermal_effects']['expansion_factor']]),
                        np.array([float(record['deployment_state'] == state) 
                                for state in self.deployment_states.keys()]),
                        np.array([float(object_type == ot) 
                                for ot in self.object_types.keys()])
                    ])
                
                # Generate labels
                object_type_onehot = np.array([float(object_type == ot) 
                                             for ot in self.object_types.keys()])
                
                # Get final state for labels
                final_record = property_records[-1]
                
                y[i] = np.concatenate([
                    object_type_onehot,  # Object type classification
                    dimensions[:3],  # Physical dimensions
                    final_record['attitude'],  # Attitude angles
                    final_record['angular_velocity'],  # Angular velocities
                    np.array([float(final_record['is_tumbling'])]),  # Tumbling state
                    material_props,  # Material properties
                    np.array([final_record['temperature']]),  # Temperature
                    np.array([final_record['thermal_effects']['expansion_factor']]),  # Thermal expansion
                    np.array([0.9])  # High confidence for synthetic data
                ])
                
            except Exception as e:
                print(f"Error generating sample {i}: {str(e)}")
                # Fill with zeros if generation fails
                continue
        
        return X, y
