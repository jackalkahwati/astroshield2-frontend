"""Cross-Tag Correlation Service for AstroShield.

This service provides functionality for matching sensor observations across different sensors
to ensure accurate identification of space objects.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy.spatial import KDTree
from infrastructure.circuit_breaker import circuit_breaker
from infrastructure.monitoring import MonitoringService
from infrastructure.cache import CacheManager
from infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)
monitoring = MonitoringService()
cache = CacheManager()
event_bus = EventBus()

class CrossTagCorrelationService:
    """Service for correlating observations across different sensors."""
    
    def __init__(self):
        """Initialize the Cross-Tag Correlation service."""
        self.observation_buffer = {}  # Buffer to store recent observations
        self.buffer_ttl = timedelta(minutes=30)  # Time to live for buffer entries
        self.match_threshold = 0.85  # Threshold for considering a match
        self.spatial_index = None  # KDTree for spatial indexing
        self.spatial_index_timestamp = None  # When the index was last updated
        self.spatial_index_ttl = timedelta(minutes=5)  # How often to rebuild the index
    
    @circuit_breaker
    async def add_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new observation to the correlation system.
        
        Args:
            observation: The observation data
            
        Returns:
            Dictionary with correlation results
        """
        with monitoring.create_span("cross_tag_add_observation") as span:
            try:
                # Validate observation
                self._validate_observation(observation)
                
                # Add timestamp if not present
                if 'timestamp' not in observation:
                    observation['timestamp'] = datetime.utcnow()
                
                # Generate observation ID if not present
                if 'observation_id' not in observation:
                    observation['observation_id'] = f"{observation['sensor_id']}_{observation['timestamp'].isoformat()}"
                
                span.set_attribute("observation_id", observation['observation_id'])
                span.set_attribute("sensor_id", observation['sensor_id'])
                
                # Store in buffer
                self.observation_buffer[observation['observation_id']] = observation
                
                # Clean up old observations
                self._clean_buffer()
                
                # Find correlations
                correlations = await self._find_correlations(observation)
                
                # Publish event
                event_bus.publish('cross_tag_observation_added', {
                    'observation_id': observation['observation_id'],
                    'sensor_id': observation['sensor_id'],
                    'timestamp': observation['timestamp'].isoformat(),
                    'correlation_count': len(correlations)
                })
                
                return {
                    'observation_id': observation['observation_id'],
                    'correlations': correlations,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error adding observation: {str(e)}")
                span.record_exception(e)
                return {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
    
    @circuit_breaker
    async def find_matches(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find matches for a query across all observations.
        
        Args:
            query: Query parameters
            
        Returns:
            List of matching observations
        """
        with monitoring.create_span("cross_tag_find_matches") as span:
            try:
                # Extract query parameters
                position = query.get('position')
                time_range = query.get('time_range', {})
                sensor_ids = query.get('sensor_ids', [])
                
                span.set_attribute("position", str(position) if position else "None")
                span.set_attribute("sensor_count", len(sensor_ids))
                
                # Find matches
                if position:
                    matches = self._find_spatial_matches(position, sensor_ids)
                else:
                    matches = self._find_temporal_matches(time_range, sensor_ids)
                
                # Publish event
                event_bus.publish('cross_tag_query_executed', {
                    'timestamp': datetime.utcnow().isoformat(),
                    'match_count': len(matches)
                })
                
                return matches
                
            except Exception as e:
                logger.error(f"Error finding matches: {str(e)}")
                span.record_exception(e)
                return []
    
    @circuit_breaker
    async def analyze_correlation_quality(self, observation_ids: List[str]) -> Dict[str, Any]:
        """Analyze the quality of correlations between observations.
        
        Args:
            observation_ids: List of observation IDs to analyze
            
        Returns:
            Dictionary with correlation quality metrics
        """
        with monitoring.create_span("cross_tag_analyze_correlation_quality") as span:
            try:
                span.set_attribute("observation_count", len(observation_ids))
                
                # Get observations
                observations = [self.observation_buffer.get(obs_id) for obs_id in observation_ids]
                observations = [obs for obs in observations if obs is not None]
                
                if len(observations) < 2:
                    return {
                        'error': 'Not enough valid observations for analysis',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                
                # Calculate metrics
                position_variance = self._calculate_position_variance(observations)
                temporal_consistency = self._calculate_temporal_consistency(observations)
                sensor_diversity = self._calculate_sensor_diversity(observations)
                
                # Calculate overall quality score
                quality_score = (position_variance * 0.5 + 
                                temporal_consistency * 0.3 + 
                                sensor_diversity * 0.2)
                
                return {
                    'quality_score': quality_score,
                    'metrics': {
                        'position_variance': position_variance,
                        'temporal_consistency': temporal_consistency,
                        'sensor_diversity': sensor_diversity
                    },
                    'observation_count': len(observations),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error analyzing correlation quality: {str(e)}")
                span.record_exception(e)
                return {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
    
    def _validate_observation(self, observation: Dict[str, Any]) -> None:
        """Validate an observation.
        
        Args:
            observation: The observation to validate
            
        Raises:
            ValueError: If the observation is invalid
        """
        required_fields = ['sensor_id', 'position']
        for field in required_fields:
            if field not in observation:
                raise ValueError(f"Missing required field: {field}")
        
        if 'position' in observation and len(observation['position']) != 3:
            raise ValueError("Position must be a 3D vector")
    
    def _clean_buffer(self) -> None:
        """Clean up old observations from the buffer."""
        current_time = datetime.utcnow()
        to_remove = []
        
        for obs_id, obs in self.observation_buffer.items():
            if current_time - obs['timestamp'] > self.buffer_ttl:
                to_remove.append(obs_id)
        
        for obs_id in to_remove:
            del self.observation_buffer[obs_id]
        
        if to_remove:
            logger.info(f"Removed {len(to_remove)} old observations from buffer")
    
    async def _find_correlations(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find correlations for a new observation.
        
        Args:
            observation: The new observation
            
        Returns:
            List of correlation results
        """
        correlations = []
        
        # Skip if no position data
        if 'position' not in observation:
            return correlations
        
        # Get position and timestamp
        position = observation['position']
        timestamp = observation['timestamp']
        
        # Find potential matches
        for obs_id, obs in self.observation_buffer.items():
            # Skip self
            if obs_id == observation['observation_id']:
                continue
            
            # Skip if from same sensor
            if obs['sensor_id'] == observation['sensor_id']:
                continue
            
            # Skip if no position data
            if 'position' not in obs:
                continue
            
            # Calculate position difference
            pos_diff = self._calculate_position_difference(position, obs['position'])
            
            # Calculate time difference
            time_diff = abs((timestamp - obs['timestamp']).total_seconds())
            
            # Calculate match score
            match_score = self._calculate_match_score(pos_diff, time_diff)
            
            # Add if above threshold
            if match_score >= self.match_threshold:
                correlations.append({
                    'observation_id': obs_id,
                    'sensor_id': obs['sensor_id'],
                    'match_score': match_score,
                    'position_difference': pos_diff,
                    'time_difference': time_diff
                })
        
        # Sort by match score
        correlations.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Publish event if matches found
        if correlations:
            event_bus.publish('cross_tag_match_found', {
                'observation_id': observation['observation_id'],
                'match_count': len(correlations),
                'best_match_score': correlations[0]['match_score'] if correlations else 0
            })
        
        return correlations
    
    def _calculate_position_difference(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate the Euclidean distance between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def _calculate_match_score(self, pos_diff: float, time_diff: float) -> float:
        """Calculate a match score based on position and time differences.
        
        Args:
            pos_diff: Position difference
            time_diff: Time difference
            
        Returns:
            Match score between 0 and 1
        """
        # Position weight is higher than time weight
        pos_weight = 0.7
        time_weight = 0.3
        
        # Normalize differences
        max_pos_diff = 100.0  # km
        max_time_diff = 300.0  # seconds
        
        norm_pos_diff = min(pos_diff / max_pos_diff, 1.0)
        norm_time_diff = min(time_diff / max_time_diff, 1.0)
        
        # Calculate score (higher is better)
        pos_score = 1.0 - norm_pos_diff
        time_score = 1.0 - norm_time_diff
        
        return pos_weight * pos_score + time_weight * time_score
    
    def _find_spatial_matches(self, position: List[float], sensor_ids: List[str]) -> List[Dict[str, Any]]:
        """Find matches based on spatial proximity.
        
        Args:
            position: Position to match
            sensor_ids: List of sensor IDs to filter by
            
        Returns:
            List of matching observations
        """
        # Build spatial index if needed
        self._ensure_spatial_index()
        
        # Query the index
        distances, indices = self.spatial_index.query(position, k=10)
        
        # Get the observations
        matches = []
        for dist, idx in zip(distances, indices):
            if idx < len(self.spatial_index_data):
                obs_id = self.spatial_index_data[idx]
                obs = self.observation_buffer.get(obs_id)
                
                if obs and (not sensor_ids or obs['sensor_id'] in sensor_ids):
                    matches.append({
                        'observation_id': obs_id,
                        'sensor_id': obs['sensor_id'],
                        'position': obs['position'],
                        'timestamp': obs['timestamp'].isoformat(),
                        'distance': float(dist)
                    })
        
        return matches
    
    def _find_temporal_matches(self, time_range: Dict[str, str], sensor_ids: List[str]) -> List[Dict[str, Any]]:
        """Find matches based on temporal proximity.
        
        Args:
            time_range: Dictionary with start and end times
            sensor_ids: List of sensor IDs to filter by
            
        Returns:
            List of matching observations
        """
        matches = []
        
        # Parse time range
        start_time = datetime.fromisoformat(time_range.get('start', '2000-01-01T00:00:00'))
        end_time = datetime.fromisoformat(time_range.get('end', '2100-01-01T00:00:00'))
        
        # Find matches
        for obs_id, obs in self.observation_buffer.items():
            if start_time <= obs['timestamp'] <= end_time:
                if not sensor_ids or obs['sensor_id'] in sensor_ids:
                    matches.append({
                        'observation_id': obs_id,
                        'sensor_id': obs['sensor_id'],
                        'position': obs.get('position'),
                        'timestamp': obs['timestamp'].isoformat()
                    })
        
        return matches
    
    def _ensure_spatial_index(self) -> None:
        """Ensure the spatial index is up to date."""
        current_time = datetime.utcnow()
        
        # Check if we need to rebuild the index
        if (self.spatial_index is None or 
            self.spatial_index_timestamp is None or 
            current_time - self.spatial_index_timestamp > self.spatial_index_ttl):
            
            # Get positions and IDs
            positions = []
            obs_ids = []
            
            for obs_id, obs in self.observation_buffer.items():
                if 'position' in obs:
                    positions.append(obs['position'])
                    obs_ids.append(obs_id)
            
            # Build the index
            if positions:
                self.spatial_index = KDTree(positions)
                self.spatial_index_data = obs_ids
                self.spatial_index_timestamp = current_time
                logger.info(f"Rebuilt spatial index with {len(positions)} points")
    
    def _calculate_position_variance(self, observations: List[Dict[str, Any]]) -> float:
        """Calculate the position variance metric.
        
        Args:
            observations: List of observations
            
        Returns:
            Position variance metric between 0 and 1
        """
        positions = [obs['position'] for obs in observations if 'position' in obs]
        
        if not positions:
            return 0.0
        
        # Calculate variance
        positions_array = np.array(positions)
        variance = np.var(positions_array, axis=0)
        mean_variance = np.mean(variance)
        
        # Normalize
        max_variance = 100.0  # km^2
        normalized_variance = min(mean_variance / max_variance, 1.0)
        
        # Convert to quality metric (lower variance is better)
        return 1.0 - normalized_variance
    
    def _calculate_temporal_consistency(self, observations: List[Dict[str, Any]]) -> float:
        """Calculate the temporal consistency metric.
        
        Args:
            observations: List of observations
            
        Returns:
            Temporal consistency metric between 0 and 1
        """
        timestamps = [obs['timestamp'] for obs in observations if 'timestamp' in obs]
        
        if len(timestamps) < 2:
            return 0.0
        
        # Sort timestamps
        timestamps.sort()
        
        # Calculate time differences
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                     for i in range(len(timestamps)-1)]
        
        # Calculate variance of time differences
        variance = np.var(time_diffs) if time_diffs else 0.0
        
        # Normalize
        max_variance = 3600.0  # seconds^2
        normalized_variance = min(variance / max_variance, 1.0)
        
        # Convert to quality metric (lower variance is better)
        return 1.0 - normalized_variance
    
    def _calculate_sensor_diversity(self, observations: List[Dict[str, Any]]) -> float:
        """Calculate the sensor diversity metric.
        
        Args:
            observations: List of observations
            
        Returns:
            Sensor diversity metric between 0 and 1
        """
        sensor_ids = set(obs['sensor_id'] for obs in observations if 'sensor_id' in obs)
        
        # Calculate diversity as ratio of unique sensors to observations
        diversity = len(sensor_ids) / len(observations) if observations else 0.0
        
        return diversity 