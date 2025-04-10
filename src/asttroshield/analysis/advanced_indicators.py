"""Models for advanced threat indicators."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

class SignatureAnalyzer:
    """Model for analyzing optical and RADAR signatures."""
    
    def analyze_signatures(self, optical_data: Dict[str, Any],
                         radar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optical and RADAR signatures for mismatches.
        
        Args:
            optical_data: Optical signature measurements
            radar_data: RADAR signature measurements
            
        Returns:
            Dict containing signature analysis results
        """
        if not optical_data or not radar_data:
            return {
                'signature_mismatch': False,
                'out_of_family': False,
                'confidence': 0.0
            }
        
        # Compare size estimates
        optical_size = optical_data.get('estimated_size', 0)
        radar_size = radar_data.get('estimated_size', 0)
        
        # Check for significant discrepancy (>20%)
        size_mismatch = abs(optical_size - radar_size) / max(optical_size, radar_size) > 0.2
        
        # Check if either signature is out of family
        optical_typical = optical_data.get('matches_typical', True)
        radar_typical = radar_data.get('matches_typical', True)
        
        return {
            'signature_mismatch': size_mismatch,
            'out_of_family': not (optical_typical and radar_typical),
            'confidence': 0.9
        }

class SubsatelliteAnalyzer:
    """Model for detecting subsatellite deployments."""
    
    def analyze_deployments(self, state_history: List[Dict[str, Any]],
                          radar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze for potential subsatellite deployments.
        
        Args:
            state_history: Historical state vectors
            radar_data: RADAR signature data
            
        Returns:
            Dict containing deployment analysis results
        """
        if not state_history or not radar_data:
            return {
                'deployment_detected': False,
                'confidence': 0.0
            }
        
        # Look for sudden changes in radar cross section
        rcs_values = [data.get('rcs', 0) for data in radar_data.get('history', [])]
        if not rcs_values:
            return {
                'deployment_detected': False,
                'confidence': 0.0
            }
        
        # Calculate RCS variations
        rcs_mean = np.mean(rcs_values)
        rcs_std = np.std(rcs_values)
        
        # Detect significant RCS changes (>3 sigma)
        significant_changes = [
            i for i, rcs in enumerate(rcs_values)
            if abs(rcs - rcs_mean) > 3 * rcs_std
        ]
        
        return {
            'deployment_detected': len(significant_changes) > 0,
            'deployment_times': [radar_data['history'][i]['time'] for i in significant_changes],
            'confidence': 0.85 if rcs_values else 0.0
        }

class PatternOfLifeAnalyzer:
    """Model for analyzing pattern-of-life violations."""
    
    def analyze_pol(self, maneuver_history: List[Dict[str, Any]],
                   rf_history: List[Dict[str, Any]],
                   baseline_pol: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze for pattern-of-life violations.
        
        Args:
            maneuver_history: Historical maneuvers
            rf_history: Historical RF emissions
            baseline_pol: Baseline pattern of life data
            
        Returns:
            Dict containing POL analysis results
        """
        if not maneuver_history or not rf_history or not baseline_pol:
            return {
                'pol_violation': False,
                'confidence': 0.0
            }
        
        violations = []
        
        # Check maneuver patterns
        if baseline_pol.get('maneuver_windows'):
            for maneuver in maneuver_history:
                maneuver_time = datetime.fromisoformat(maneuver['time'])
                in_window = any(
                    window['start'] <= maneuver_time <= window['end']
                    for window in baseline_pol['maneuver_windows']
                )
                if not in_window:
                    violations.append({
                        'type': 'maneuver',
                        'time': maneuver['time']
                    })
        
        # Check RF patterns
        if baseline_pol.get('rf_windows'):
            for emission in rf_history:
                emission_time = datetime.fromisoformat(emission['time'])
                in_window = any(
                    window['start'] <= emission_time <= window['end']
                    for window in baseline_pol['rf_windows']
                )
                if not in_window:
                    violations.append({
                        'type': 'rf_emission',
                        'time': emission['time']
                    })
        
        return {
            'pol_violation': len(violations) > 0,
            'violations': violations,
            'confidence': 0.9 if violations else 0.7
        }

class RemoteSensingAnalyzer:
    """Model for analyzing potential remote sensing activities."""
    
    def analyze_sensing_activity(self, state_history: List[Dict[str, Any]],
                               target_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze for potential remote sensing activities.
        
        Args:
            state_history: Historical state vectors
            target_objects: Potential target objects
            
        Returns:
            Dict containing remote sensing analysis results
        """
        if not state_history or not target_objects:
            return {
                'sensing_detected': False,
                'confidence': 0.0
            }
        
        sensing_events = []
        
        # Analyze each potential target
        for target in target_objects:
            target_pos = target.get('position', {})
            
            # Check for close approaches with favorable geometry
            for state in state_history:
                state_pos = state.get('position', {})
                
                # Calculate range
                range_km = np.sqrt(sum(
                    (state_pos[k] - target_pos[k])**2
                    for k in ['x', 'y', 'z']
                ))
                
                # Check if range is within typical remote sensing distances
                if range_km < 1000:  # Example threshold
                    sensing_events.append({
                        'time': state['epoch'],
                        'target_id': target.get('id'),
                        'range_km': range_km
                    })
        
        return {
            'sensing_detected': len(sensing_events) > 0,
            'events': sensing_events,
            'confidence': 0.85 if sensing_events else 0.0
        }

class AnalystDisagreementDetector:
    """Model for detecting analyst classification disagreements."""
    
    def analyze_classifications(self, 
                             classifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze for disagreements in analyst classifications.
        
        Args:
            classifications: List of analyst classifications
            
        Returns:
            Dict containing disagreement analysis results
        """
        if not classifications:
            return {
                'disagreement_detected': False,
                'confidence': 0.0
            }
        
        # Count classifications by type
        class_counts = {}
        for classification in classifications:
            class_type = classification.get('classification')
            if class_type:
                class_counts[class_type] = class_counts.get(class_type, 0) + 1
        
        # Check for significant disagreement (no clear majority)
        total = len(classifications)
        has_majority = any(
            count > total * 0.6  # 60% threshold for majority
            for count in class_counts.values()
        )
        
        return {
            'disagreement_detected': not has_majority,
            'classification_counts': class_counts,
            'confidence': 0.95 if total >= 3 else 0.7  # Higher confidence with more analysts
        }

class RegistryChecker:
    """Model for checking UN registry status."""
    
    def check_registry(self, object_id: str, 
                      registry_data: Dict[str, Any]) -> Dict[str, Any]:
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