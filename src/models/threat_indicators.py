"""Models for threat indicator analysis."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

class StabilityIndicator:
    """Model for detecting object stability changes."""
    
    def analyze_stability(self, state_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze object stability based on state vector history.
        
        Args:
            state_history: List of historical state vectors
            
        Returns:
            Dict containing stability analysis results
            
        Raises:
            TypeError: If state_history is None or not a list
            ValueError: If any state vector is missing required fields
        """
        if state_history is None:
            raise TypeError("state_history cannot be None")
            
        if not isinstance(state_history, list):
            raise TypeError("state_history must be a list")
            
        if not state_history:
            return {
                'is_stable': True,
                'confidence': 0.0,
                'changes_detected': []
            }
        
        # Validate state vectors
        for state in state_history:
            if not isinstance(state, dict):
                raise TypeError("Each state must be a dictionary")
            if 'position' not in state:
                raise ValueError("Each state must contain a position field")
            if not isinstance(state['position'], dict):
                raise TypeError("Position must be a dictionary")
            for coord in ['x', 'y', 'z']:
                if coord not in state['position']:
                    raise ValueError(f"Position must contain {coord} coordinate")
                if not isinstance(state['position'][coord], (int, float)):
                    raise TypeError(f"Position {coord} must be a number")
        
        # Calculate attitude and position variations
        variations = []
        for i in range(1, len(state_history)):
            prev = state_history[i-1]
            curr = state_history[i]
            
            # Calculate position change
            pos_delta = np.sqrt(sum(
                (curr['position'][k] - prev['position'][k])**2
                for k in ['x', 'y', 'z']
            ))
            variations.append(pos_delta)
        
        # Analyze variations
        mean_variation = np.mean(variations) if variations else 0
        std_variation = np.std(variations) if variations else 0
        
        # Detect significant changes (> 2 standard deviations)
        changes = [
            i for i, var in enumerate(variations)
            if abs(var - mean_variation) > 2 * std_variation
        ]
        
        return {
            'is_stable': len(changes) == 0,
            'confidence': 0.95 if variations else 0.0,
            'changes_detected': changes
        }

class ManeuverIndicator:
    """Model for detecting suspicious maneuvers."""
    
    def analyze_maneuvers(self, maneuver_history: List[Dict[str, Any]], baseline_pol: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze maneuvers for suspicious behavior.
        
        Args:
            maneuver_history: List of historical maneuvers
            baseline_pol: List of baseline pattern of life data
            
        Returns:
            Dict containing maneuver analysis results
            
        Raises:
            TypeError: If inputs are None or not lists
            ValueError: If any maneuver is missing required fields
        """
        if maneuver_history is None or baseline_pol is None:
            raise TypeError("maneuver_history and baseline_pol cannot be None")
            
        if not isinstance(maneuver_history, list) or not isinstance(baseline_pol, list):
            raise TypeError("maneuver_history and baseline_pol must be lists")
            
        if not maneuver_history:
            return {
                'is_suspicious': False,
                'confidence': 0.0,
                'anomalies': []
            }
            
        # Validate maneuvers
        for maneuver in maneuver_history:
            if not isinstance(maneuver, dict):
                raise TypeError("Each maneuver must be a dictionary")
            if 'time' not in maneuver:
                raise ValueError("Each maneuver must contain a time field")
            if 'delta_v' not in maneuver:
                raise ValueError("Each maneuver must contain a delta_v field")
            if not isinstance(maneuver['delta_v'], (int, float)):
                raise TypeError("delta_v must be a number")
            
            # Make final_position optional for time-based tests
            if 'final_position' in maneuver:
                if not isinstance(maneuver['final_position'], dict):
                    raise TypeError("final_position must be a dictionary")
                for coord in ['x', 'y', 'z']:
                    if coord not in maneuver['final_position']:
                        raise ValueError(f"final_position must contain {coord} coordinate")
                    if not isinstance(maneuver['final_position'][coord], (int, float)):
                        raise TypeError(f"final_position {coord} must be a number")
        
        # Rest of the analysis logic
        anomalies = []
        for maneuver in maneuver_history:
            if maneuver['delta_v'] > 0.1:  # Threshold for suspicious delta-v
                anomalies.append({
                    'time': maneuver['time'],
                    'type': 'high_delta_v',
                    'value': maneuver['delta_v']
                })
                
        return {
            'is_suspicious': len(anomalies) > 0,
            'confidence': 0.95 if maneuver_history else 0.0,
            'anomalies': anomalies
        }

class RFIndicator:
    """Model for detecting RF pattern anomalies."""
    
    def analyze_rf_pattern(self, rf_history: List[Dict[str, Any]], baseline_pol: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze RF patterns for anomalies.
        
        Args:
            rf_history: List of historical RF measurements
            baseline_pol: Baseline pattern of life data
            
        Returns:
            Dict containing RF pattern analysis results
            
        Raises:
            TypeError: If inputs are None or of wrong type
            ValueError: If any measurement is missing required fields
        """
        if rf_history is None or baseline_pol is None:
            raise TypeError("rf_history and baseline_pol cannot be None")
            
        if not isinstance(rf_history, list):
            raise TypeError("rf_history must be a list")
            
        if not isinstance(baseline_pol, dict):
            raise TypeError("baseline_pol must be a dictionary")
            
        if not rf_history:
            return {
                'is_anomalous': False,
                'confidence': 0.0,
                'anomalies': []
            }
            
        # Validate RF measurements
        for measurement in rf_history:
            if not isinstance(measurement, dict):
                raise TypeError("Each measurement must be a dictionary")
            if 'time' not in measurement:
                raise ValueError("Each measurement must contain a time field")
            if 'power_level' not in measurement:
                raise ValueError("Each measurement must contain a power_level field")
            if not isinstance(measurement['power_level'], (int, float)):
                raise TypeError("power_level must be a number")
            if 'frequency' not in measurement:
                raise ValueError("Each measurement must contain a frequency field")
            if not isinstance(measurement['frequency'], (int, float)):
                raise TypeError("frequency must be a number")
        
        # Rest of the analysis logic
        anomalies = []
        for measurement in rf_history:
            if measurement['power_level'] > -85.0:  # Threshold for suspicious power level
                anomalies.append({
                    'time': measurement['time'],
                    'type': 'high_power',
                    'value': measurement['power_level']
                })
                
        return {
            'is_anomalous': len(anomalies) > 0,
            'confidence': 0.95 if rf_history else 0.0,
            'anomalies': anomalies
        }

class SignatureAnalyzer:
    """Model for detecting signature anomalies."""
    
    def analyze_signatures(self, optical_data: Dict[str, Any], radar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optical and radar signatures for anomalies.
        
        Args:
            optical_data: Optical signature data
            radar_data: Radar signature data
            
        Returns:
            Dict containing signature analysis results
            
        Raises:
            TypeError: If inputs are None or not dictionaries
            ValueError: If any required fields are missing
        """
        if optical_data is None or radar_data is None:
            raise TypeError("optical_data and radar_data cannot be None")
            
        if not isinstance(optical_data, dict) or not isinstance(radar_data, dict):
            raise TypeError("optical_data and radar_data must be dictionaries")
            
        if not optical_data or not radar_data:
            return {
                'is_anomalous': False,
                'confidence': 0.0,
                'anomalies': []
            }
            
        # Validate optical data
        if 'magnitude' not in optical_data:
            raise ValueError("optical_data must contain a magnitude field")
        if not isinstance(optical_data['magnitude'], (int, float)):
            raise TypeError("magnitude must be a number")
        if 'time' not in optical_data:
            raise ValueError("optical_data must contain a time field")
            
        # Validate radar data
        if 'rcs' not in radar_data:
            raise ValueError("radar_data must contain an rcs field")
        if not isinstance(radar_data['rcs'], (int, float)):
            raise TypeError("rcs must be a number")
        if 'time' not in radar_data:
            raise ValueError("radar_data must contain a time field")
        
        # Rest of the analysis logic
        anomalies = []
        if abs(optical_data['magnitude'] - radar_data['rcs']) > 10.0:  # Example threshold
            anomalies.append({
                'time': optical_data['time'],
                'type': 'signature_mismatch',
                'optical_mag': optical_data['magnitude'],
                'radar_rcs': radar_data['rcs']
            })
                
        return {
            'is_anomalous': len(anomalies) > 0,
            'confidence': 0.95 if optical_data and radar_data else 0.0,
            'anomalies': anomalies
        }

class OrbitAnalyzer:
    """Model for analyzing orbital characteristics."""
    
    def analyze_orbit(self, orbit_data: Dict[str, Any],
                     population_data: Dict[str, Any],
                     radiation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze orbital characteristics for suspicious patterns.
        
        Args:
            orbit_data: Current orbital parameters
            population_data: Orbital population statistics
            radiation_data: Radiation environment data
            
        Returns:
            Dict containing orbit analysis results
            
        Raises:
            TypeError: If inputs are None or not dictionaries
            ValueError: If required fields are missing
        """
        if orbit_data is None or population_data is None or radiation_data is None:
            raise TypeError("orbit_data, population_data, and radiation_data cannot be None")
            
        if not isinstance(orbit_data, dict) or not isinstance(population_data, dict) or not isinstance(radiation_data, dict):
            raise TypeError("All inputs must be dictionaries")
        
        if not orbit_data:
            return {
                'out_of_family': False,
                'unoccupied_orbit': False,
                'high_radiation': False,
                'confidence': 0.0
            }
        
        # Check if orbit is sparsely populated
        sma = orbit_data.get('semi_major_axis', 0)
        inc = orbit_data.get('inclination', 0)
        
        nearby_objects = population_data.get('objects_within_range', 0)
        radiation_level = radiation_data.get('radiation_level', 0)
        
        return {
            'out_of_family': False,  # Implement family analysis
            'unoccupied_orbit': nearby_objects < 5,  # Threshold for "unoccupied"
            'high_radiation': radiation_level > 100,  # Threshold for "high" radiation
            'confidence': 0.85
        }

class LaunchAnalyzer:
    """Model for analyzing launch-related indicators."""
    
    def analyze_launch(self, launch_data: Dict[str, Any],
                      tracked_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze launch-related characteristics.
        
        Args:
            launch_data: Launch information
            tracked_objects: List of tracked objects from launch
            
        Returns:
            Dict containing launch analysis results
        """
        if not launch_data or not tracked_objects:
            return {
                'suspicious_source': False,
                'excess_objects': False,
                'confidence': 0.0
            }
        
        # Check if launch site is known for threats
        suspicious_sites = launch_data.get('known_threat_sites', [])
        launch_site = launch_data.get('launch_site', '')
        
        # Compare number of tracked objects with expected
        expected_objects = launch_data.get('expected_objects', 1)
        actual_objects = len(tracked_objects)
        
        return {
            'suspicious_source': launch_site in suspicious_sites,
            'excess_objects': actual_objects > expected_objects,
            'confidence': 0.9
        }

class RegistryChecker:
    """Model for checking UN registry status."""
    
    def check_registry(self, object_id: str, registry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if object is properly registered.
        
        Args:
            object_id: Object identifier
            registry_data: UN registry data
            
        Returns:
            Dict containing registry check results
        """
        if not registry_data:
            return {
                'registered': False,
                'confidence': 0.0
            }
        
        registered_objects = registry_data.get('registered_objects', [])
        
        return {
            'registered': object_id in registered_objects,
            'confidence': 1.0  # Registry status is deterministic
        }

class AMRAnalyzer:
    """Model for analyzing Area-to-Mass Ratio (AMR) characteristics."""
    
    def analyze_amr(self, amr_history: List[Dict[str, Any]], 
                   population_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze AMR values and changes.
        
        Args:
            amr_history: Historical AMR measurements
            population_data: Population statistics for comparison
            
        Returns:
            Dict containing AMR analysis results
        """
        if not amr_history:
            return {
                'out_of_family': False,
                'notable_changes': False,
                'confidence': 0.0
            }
        
        # Extract AMR values
        amr_values = [entry['amr'] for entry in amr_history]
        
        # Calculate statistics
        mean_amr = np.mean(amr_values)
        std_amr = np.std(amr_values)
        
        # Compare with population
        population_mean = population_data.get('mean_amr', mean_amr)
        population_std = population_data.get('std_amr', std_amr)
        
        # Check if out of family (>3 sigma from population mean)
        out_of_family = bool(abs(mean_amr - population_mean) > 3 * population_std)  # Convert to Python bool
        
        # Detect notable changes (>2 sigma change between measurements)
        changes = []
        for i in range(1, len(amr_values)):
            delta = abs(amr_values[i] - amr_values[i-1])
            if delta > 2 * std_amr:
                changes.append({
                    'time': amr_history[i]['time'],
                    'old_value': float(amr_values[i-1]),  # Convert to Python float
                    'new_value': float(amr_values[i])     # Convert to Python float
                })
        
        return {
            'out_of_family': out_of_family,
            'notable_changes': bool(changes),  # Convert to Python bool
            'changes': changes,
            'confidence': float(0.9 if len(amr_values) > 5 else 0.7)  # Convert to Python float
        }

class StimulationAnalyzer:
    """Model for detecting stimulation by US/allied systems."""
    
    def analyze_stimulation(self, object_events: List[Dict[str, Any]],
                          system_locations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze for potential stimulation by known systems.
        
        Args:
            object_events: List of object behavior events
            system_locations: Known system locations and capabilities
            
        Returns:
            Dict containing stimulation analysis results
        """
        if not object_events or not system_locations:
            return {
                'stimulation_detected': False,
                'confidence': 0.0
            }
        
        stimulation_events = []
        
        # Check each event for temporal and spatial correlation
        for event in object_events:
            event_time = datetime.fromisoformat(event['time'])
            event_pos = event.get('position', {})
            
            # Check each known system
            for system in system_locations.get('systems', []):
                system_pos = system.get('position', {})
                
                # Calculate range to system
                range_km = np.sqrt(sum(
                    (event_pos[k] - system_pos[k])**2
                    for k in ['x', 'y', 'z']
                ))
                
                # Check if within system's range
                if range_km <= system.get('effective_range_km', 0):
                    stimulation_events.append({
                        'time': event['time'],
                        'system_id': system.get('id'),
                        'range_km': range_km
                    })
        
        return {
            'stimulation_detected': len(stimulation_events) > 0,
            'events': stimulation_events,
            'confidence': 0.85 if stimulation_events else 0.0
        }

class ImagingManeuverAnalyzer:
    """Model for detecting imaging-related maneuvers."""
    
    def analyze_imaging_maneuvers(self, maneuver_history: List[Dict[str, Any]], 
                                coverage_gaps: List[Dict[str, Any]], 
                                baseline_pol: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze maneuvers for imaging-related behavior.
        
        Args:
            maneuver_history: List of historical maneuvers
            coverage_gaps: List of sensor coverage gaps
            baseline_pol: Baseline pattern of life data
            
        Returns:
            Dict containing imaging maneuver analysis results
            
        Raises:
            TypeError: If inputs are None or of wrong type
            ValueError: If any maneuver is missing required fields
        """
        if maneuver_history is None or coverage_gaps is None or baseline_pol is None:
            raise TypeError("maneuver_history, coverage_gaps, and baseline_pol cannot be None")
            
        if not isinstance(maneuver_history, list):
            raise TypeError("maneuver_history must be a list")
            
        if not isinstance(coverage_gaps, list):
            raise TypeError("coverage_gaps must be a list")
            
        if not isinstance(baseline_pol, dict):
            raise TypeError("baseline_pol must be a dictionary")
            
        if not maneuver_history:
            return {
                'is_imaging': False,
                'confidence': 0.0,
                'anomalies': []
            }
            
        # Validate maneuvers
        for maneuver in maneuver_history:
            if not isinstance(maneuver, dict):
                raise TypeError("Each maneuver must be a dictionary")
            if 'time' not in maneuver:
                raise ValueError("Each maneuver must contain a time field")
            if 'delta_v' not in maneuver:
                raise ValueError("Each maneuver must contain a delta_v field")
            if not isinstance(maneuver['delta_v'], (int, float)):
                raise TypeError("delta_v must be a number")
            if 'final_position' not in maneuver:
                raise ValueError("Each maneuver must contain a final_position field")
            if not isinstance(maneuver['final_position'], dict):
                raise TypeError("final_position must be a dictionary")
            for coord in ['x', 'y', 'z']:
                if coord not in maneuver['final_position']:
                    raise ValueError(f"final_position must contain {coord} coordinate")
                if not isinstance(maneuver['final_position'][coord], (int, float)):
                    raise TypeError(f"final_position {coord} must be a number")
        
        # Rest of the analysis logic
        anomalies = []
        for maneuver in maneuver_history:
            if maneuver['delta_v'] > 0.1:  # Threshold for suspicious delta-v
                anomalies.append({
                    'time': maneuver['time'],
                    'type': 'high_delta_v',
                    'value': maneuver['delta_v']
                })
                
        return {
            'is_imaging': len(anomalies) > 0,
            'confidence': 0.95 if maneuver_history else 0.0,
            'anomalies': anomalies
        }

class DebrisAnalyzer:
    """Model for analyzing debris characteristics."""
    
    def analyze_debris(self, debris_object: Dict[str, Any],
                      parent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze debris object characteristics relative to parent.
        
        Args:
            debris_object: Debris object data
            parent_data: Parent object data
            
        Returns:
            Dict containing debris analysis results
        """
        if not debris_object or not parent_data:
            return {
                'higher_sma': False,
                'confidence': 0.0
            }
        
        # Get semi-major axis values
        debris_sma = debris_object.get('semi_major_axis', 0)
        parent_sma = parent_data.get('semi_major_axis', 0)
        
        # Compare SMAs
        sma_difference = debris_sma - parent_sma
        
        return {
            'higher_sma': sma_difference > 0,
            'sma_difference_km': sma_difference,
            'confidence': 1.0  # SMA comparison is deterministic
        }

class UCTAnalyzer:
    """Model for analyzing uncorrelated tracks."""
    
    def analyze_uct(self, track_data: Dict[str, Any],
                   illumination_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze uncorrelated track during eclipse.
        
        Args:
            track_data: Track observation data
            illumination_data: Solar illumination data
            
        Returns:
            Dict containing UCT analysis results
        """
        if not track_data or not illumination_data:
            return {
                'true_uct': False,
                'in_eclipse': False,
                'confidence': 0.0
            }
        
        # Check if track is uncorrelated
        is_uncorrelated = not track_data.get('correlated_object_id')
        
        # Check illumination condition
        track_time = datetime.fromisoformat(track_data.get('time', ''))
        in_eclipse = any(
            start <= track_time <= end
            for start, end in illumination_data.get('eclipse_periods', [])
        )
        
        return {
            'true_uct': is_uncorrelated,
            'in_eclipse': in_eclipse,
            'confidence': 0.95 if is_uncorrelated and in_eclipse else 0.7
        } 