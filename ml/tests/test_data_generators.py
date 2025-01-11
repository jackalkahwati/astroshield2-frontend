import unittest
import numpy as np
import datetime
from ..data_generation.physical_data_gen import PhysicalPropertiesGenerator
from ..data_generation.proximity_data_gen import ProximityDataGenerator
from ..data_generation.remote_sensing_data_gen import RemoteSensingDataGenerator
from ..data_generation.eclipse_data_gen import EclipseDataGenerator
from ..data_generation.track_data_gen import TrackDataGenerator

class TestDataGenerators(unittest.TestCase):
    def setUp(self):
        """Initialize all data generators"""
        self.physical_gen = PhysicalPropertiesGenerator()
        self.proximity_gen = ProximityDataGenerator()
        self.remote_sensing_gen = RemoteSensingDataGenerator()
        self.eclipse_gen = EclipseDataGenerator()
        self.track_gen = TrackDataGenerator()

    def test_physical_properties_generation(self):
        """Test physical properties data generation"""
        # Test single state generation
        object_type = 'active_satellite'
        properties = self.physical_gen.generate_object_properties(object_type)
        
        # Verify basic properties exist and are within expected ranges
        self.assertIn('mass', properties)
        self.assertIn('dimensions', properties)
        self.assertIn('amr', properties)
        self.assertIn('temperature', properties)
        
        # Test sequence generation
        duration_hours = 2
        records, anomalies = self.physical_gen.generate_property_sequence(
            object_type,
            duration_hours=duration_hours
        )
        
        # Verify sequence length and continuity
        self.assertGreater(len(records), 0)
        self.assertEqual(len(records), duration_hours * 3600 // 60)  # 60s sample rate
        
        # Test training data generation
        X, y = self.physical_gen.generate_training_data(num_samples=10)
        self.assertEqual(X.shape[0], 10)
        self.assertEqual(y.shape[1], 5)  # 5 label components

    def test_proximity_operations_generation(self):
        """Test proximity operations data generation"""
        # Test single conjunction event
        orbit_type = 'LEO'
        event = self.proximity_gen.generate_conjunction_event(orbit_type)
        
        # Verify event properties
        self.assertIn('primary_state', event)
        self.assertIn('secondary_state', event)
        self.assertIn('collision_probability', event)
        self.assertGreaterEqual(event['collision_probability'], 0.0)
        self.assertLessEqual(event['collision_probability'], 1.0)
        
        # Test sequence generation
        duration_hours = 2
        records, anomalies = self.proximity_gen.generate_conjunction_sequence(
            duration_hours=duration_hours
        )
        
        # Verify sequence properties
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertIn('range_km', record)
            self.assertGreater(record['range_km'], 0)

    def test_remote_sensing_generation(self):
        """Test remote sensing data generation"""
        # Test single observation
        target_type = 'active_satellite'
        sensor_type = 'ground_telescope'
        atm_condition = 'clear'
        
        obs = self.remote_sensing_gen.generate_observation(
            target_type,
            sensor_type,
            atm_condition
        )
        
        # Verify observation properties
        self.assertIn('measurements', obs)
        self.assertIn('optical_snr', obs['measurements'])
        self.assertIn('radar_snr', obs['measurements'])
        
        # Test sequence generation
        duration_hours = 2
        records, anomalies = self.remote_sensing_gen.generate_observation_sequence(
            duration_hours=duration_hours
        )
        
        # Verify sequence properties
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertIn('signature', record)
            self.assertIn('measurements', record)

    def test_eclipse_generation(self):
        """Test eclipse data generation"""
        # Test single eclipse state
        satellite_type = 'smallsat'
        orbit_type = 'LEO'
        eclipse_type = 'umbra'
        temperature = 293.15
        battery_state = 1.0
        
        state = self.eclipse_gen.generate_eclipse_state(
            satellite_type,
            orbit_type,
            eclipse_type,
            temperature,
            battery_state
        )
        
        # Verify state properties
        self.assertIn('thermal', state)
        self.assertIn('power', state)
        self.assertIn('timing', state)
        
        # Test sequence generation
        duration_hours = 2
        records, anomalies = self.eclipse_gen.generate_eclipse_sequence(
            duration_hours=duration_hours
        )
        
        # Verify sequence properties
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertIn('thermal', record)
            self.assertIn('power', record)

    def test_track_generation(self):
        """Test track data generation"""
        # Test single track generation
        track_type = 'debris'
        orbit_region = 'LEO'
        
        states, observations = self.track_gen.generate_track(
            track_type,
            orbit_region
        )
        
        # Verify track properties
        self.assertGreater(len(states), 0)
        for state in states:
            self.assertIn('position', state)
            self.assertIn('velocity', state)
        
        # Test sequence generation
        duration_hours = 2
        records, anomalies = self.track_gen.generate_track_sequence(
            duration_hours=duration_hours
        )
        
        # Verify sequence properties
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertIn('track_type', record)
            self.assertIn('states', record)

    def test_physical_consistency(self):
        """Test physical consistency of generated data"""
        # Test AMR consistency
        properties = self.physical_gen.generate_object_properties('active_satellite')
        if 'solar_array_area' in properties['dimensions']:
            self.assertGreater(properties['amr'], 0.01)  # Minimum AMR for satellite with arrays
        
        # Test eclipse thermal balance
        state = self.eclipse_gen.generate_eclipse_state(
            'smallsat', 'LEO', 'umbra', 293.15, 1.0
        )
        if state['eclipse_type'] == 'umbra':
            self.assertLess(state['thermal']['net_heat_flow'], 0)  # Should cool in umbra
        
        # Test orbital mechanics
        event = self.proximity_gen.generate_conjunction_event('LEO')
        r_mag = np.linalg.norm(event['primary_state'][:3])
        v_mag = np.linalg.norm(event['primary_state'][3:])
        
        # Verify orbital velocity is consistent with altitude (vis-viva equation)
        mu = 398600.4418
        expected_v = np.sqrt(mu/r_mag)
        self.assertAlmostEqual(v_mag, expected_v, delta=1.0)

    def test_anomaly_detection(self):
        """Test anomaly detection capabilities"""
        # Test physical anomalies
        _, anomalies = self.physical_gen.generate_property_sequence(
            'active_satellite',
            anomaly_probability=1.0  # Force anomaly generation
        )
        self.assertGreater(len(anomalies), 0)
        
        # Test proximity anomalies
        _, anomalies = self.proximity_gen.generate_conjunction_sequence(
            anomaly_probability=1.0
        )
        self.assertGreater(len(anomalies), 0)
        
        # Test remote sensing anomalies
        _, anomalies = self.remote_sensing_gen.generate_observation_sequence(
            anomaly_probability=1.0
        )
        self.assertGreater(len(anomalies), 0)

    def test_error_handling(self):
        """Test error handling in data generators"""
        # Test invalid object type
        with self.assertRaises(KeyError):
            self.physical_gen.generate_object_properties('invalid_type')
        
        # Test invalid orbit type
        with self.assertRaises(KeyError):
            self.proximity_gen.generate_conjunction_event('invalid_orbit')
        
        # Test invalid sensor type
        with self.assertRaises(KeyError):
            self.remote_sensing_gen.generate_observation(
                'active_satellite',
                'invalid_sensor',
                'clear'
            )

if __name__ == '__main__':
    unittest.main() 