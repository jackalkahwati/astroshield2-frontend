"""TMDB Comparison Service for AstroShield.

This service provides functionality for comparing trajectories with the Trajectory Message Database (TMDB)
to help distinguish between actual maneuvers and environmental effects.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from infrastructure.circuit_breaker import circuit_breaker
from infrastructure.monitoring import MonitoringService
from infrastructure.cache import CacheManager
from infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)
monitoring = MonitoringService()
cache = CacheManager()
event_bus = EventBus()

class TMDBComparisonService:
    """Service for comparing trajectories with the TMDB."""
    
    def __init__(self, tmdb_base_url: str = None, tmdb_api_key: str = None):
        """Initialize the TMDB Comparison service.
        
        Args:
            tmdb_base_url: Base URL for the TMDB API
            tmdb_api_key: API key for authentication
        """
        self.tmdb_base_url = tmdb_base_url or "https://tmdb.example.com/api"
        self.tmdb_api_key = tmdb_api_key or "dummy_key"
        self.cache_ttl = timedelta(hours=1)  # Time to live for cache entries
    
    @circuit_breaker
    async def compare_trajectory(self, object_id: str, trajectory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare a trajectory with TMDB data to identify environmental vs. intentional deviations.
        
        Args:
            object_id: ID of the object
            trajectory_data: List of trajectory points
            
        Returns:
            Dictionary with comparison results
        """
        with monitoring.create_span("tmdb_compare_trajectory") as span:
            try:
                span.set_attribute("object_id", object_id)
                span.set_attribute("trajectory_points", len(trajectory_data))
                
                # Check cache first
                cache_key = f"tmdb_comparison_{object_id}_{hash(str(trajectory_data))}"
                cached_result = await cache.get(cache_key)
                if cached_result:
                    logger.info(f"Retrieved TMDB comparison from cache for {object_id}")
                    return cached_result
                
                # Get reference trajectory from TMDB
                reference_trajectory = await self._get_reference_trajectory(object_id)
                
                # If no reference trajectory is available, return early
                if not reference_trajectory:
                    return {
                        "object_id": object_id,
                        "status": "no_reference_data",
                        "message": "No reference trajectory available in TMDB",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # Align trajectories in time and space
                aligned_data = self._align_trajectories(trajectory_data, reference_trajectory)
                
                # Calculate deviations
                deviations = self._calculate_deviations(aligned_data)
                
                # Analyze deviations to determine if they're environmental or intentional
                analysis = self._analyze_deviations(deviations)
                
                # Prepare result
                result = {
                    "object_id": object_id,
                    "status": "analyzed",
                    "deviations": deviations,
                    "analysis": analysis,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Cache the result
                await cache.set(cache_key, result, ttl=self.cache_ttl)
                
                # Publish event
                event_bus.publish("tmdb_comparison_completed", {
                    "object_id": object_id,
                    "is_environmental": analysis["is_environmental"],
                    "confidence_level": analysis["confidence_level"],
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # If the analysis indicates this is environmental rather than a maneuver,
                # also create an Anti-CCDM indicator
                if analysis["is_environmental"] and analysis["confidence_level"] > 0.7:
                    anti_ccdm = {
                        "object_id": object_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "indicator_type": "tmdb_comparison",
                        "environmental_factor": analysis["primary_factor"],
                        "confidence_level": analysis["confidence_level"],
                        "expected_deviation": deviations["expected"],
                        "actual_deviation": deviations["actual"],
                        "is_environmental": True,
                        "details": {
                            "deviation_ratio": analysis["deviation_ratio"],
                            "environmental_factors": analysis["environmental_factors"]
                        }
                    }
                    event_bus.publish("anti_ccdm_indicator_created", anti_ccdm)
                
                return result
                
            except Exception as e:
                logger.error(f"Error comparing trajectory with TMDB: {str(e)}")
                span.record_exception(e)
                return {
                    "object_id": object_id,
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    @circuit_breaker
    async def get_environmental_factors(self, timestamp: datetime = None) -> Dict[str, Any]:
        """Get environmental factors for a specific time.
        
        Args:
            timestamp: Time for which to get environmental factors (defaults to now)
            
        Returns:
            Dictionary with environmental factors
        """
        with monitoring.create_span("tmdb_get_environmental_factors") as span:
            try:
                # Use provided timestamp or current time
                timestamp = timestamp or datetime.utcnow()
                span.set_attribute("timestamp", timestamp.isoformat())
                
                # Check cache first
                cache_key = f"environmental_factors_{timestamp.strftime('%Y%m%d%H')}"
                cached_result = await cache.get(cache_key)
                if cached_result:
                    logger.info(f"Retrieved environmental factors from cache for {timestamp}")
                    return cached_result
                
                # In a real implementation, this would query an external API
                # For now, return sample data
                
                # Sample environmental factors
                factors = {
                    "timestamp": timestamp.isoformat(),
                    "solar_activity": {
                        "f10.7_index": 150.5,
                        "sunspot_number": 75,
                        "solar_flares": "moderate"
                    },
                    "geomagnetic_activity": {
                        "kp_index": 3.5,
                        "dst_index": -25,
                        "auroral_activity": "low"
                    },
                    "atmospheric_conditions": {
                        "density_scale": 1.2,
                        "temperature_scale": 1.1,
                        "wind_scale": 0.9
                    }
                }
                
                # Cache the result
                await cache.set(cache_key, factors, ttl=timedelta(hours=6))
                
                return factors
                
            except Exception as e:
                logger.error(f"Error getting environmental factors: {str(e)}")
                span.record_exception(e)
                return {
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    async def _get_reference_trajectory(self, object_id: str) -> List[Dict[str, Any]]:
        """Get reference trajectory from TMDB.
        
        Args:
            object_id: ID of the object
            
        Returns:
            List of reference trajectory points
        """
        # In a real implementation, this would query the TMDB API
        # For now, return sample data
        
        # Generate a sample reference trajectory
        current_time = datetime.utcnow()
        reference_trajectory = []
        
        # Create trajectory points for the past 24 hours
        for i in range(24):
            timestamp = current_time - timedelta(hours=i)
            
            # Simple circular orbit
            angle = i * 15  # 15 degrees per hour
            radius = 7000.0  # 7000 km
            
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            z = 0.0
            
            # Add some noise to make it realistic
            x += np.random.normal(0, 1.0)
            y += np.random.normal(0, 1.0)
            z += np.random.normal(0, 1.0)
            
            reference_trajectory.append({
                "timestamp": timestamp.isoformat(),
                "position": [x, y, z],
                "velocity": [-y * 0.001, x * 0.001, 0.0]  # Simple circular velocity
            })
        
        # Sort by timestamp (oldest first)
        reference_trajectory.sort(key=lambda p: p["timestamp"])
        
        return reference_trajectory
    
    def _align_trajectories(self, trajectory_data: List[Dict[str, Any]], reference_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Align the actual trajectory with the reference trajectory.
        
        Args:
            trajectory_data: Actual trajectory data
            reference_trajectory: Reference trajectory data
            
        Returns:
            Dictionary with aligned trajectory data
        """
        # Convert timestamps to datetime objects
        trajectory_times = [
            datetime.fromisoformat(point["timestamp"]) if isinstance(point["timestamp"], str) else point["timestamp"]
            for point in trajectory_data
        ]
        reference_times = [
            datetime.fromisoformat(point["timestamp"]) if isinstance(point["timestamp"], str) else point["timestamp"]
            for point in reference_trajectory
        ]
        
        # Find overlapping time range
        start_time = max(min(trajectory_times), min(reference_times))
        end_time = min(max(trajectory_times), max(reference_times))
        
        # Filter points within the overlapping time range
        filtered_trajectory = [
            point for point, time in zip(trajectory_data, trajectory_times)
            if start_time <= time <= end_time
        ]
        filtered_reference = [
            point for point, time in zip(reference_trajectory, reference_times)
            if start_time <= time <= end_time
        ]
        
        # Interpolate reference trajectory to match actual trajectory times
        interpolated_reference = []
        
        for point, time in zip(filtered_trajectory, trajectory_times):
            if time < start_time or time > end_time:
                continue
                
            # Find reference points before and after this time
            before_points = [
                (i, ref_point) for i, ref_point in enumerate(filtered_reference)
                if (datetime.fromisoformat(ref_point["timestamp"]) 
                    if isinstance(ref_point["timestamp"], str) 
                    else ref_point["timestamp"]) <= time
            ]
            after_points = [
                (i, ref_point) for i, ref_point in enumerate(filtered_reference)
                if (datetime.fromisoformat(ref_point["timestamp"]) 
                    if isinstance(ref_point["timestamp"], str) 
                    else ref_point["timestamp"]) > time
            ]
            
            if not before_points or not after_points:
                continue
                
            # Get closest points before and after
            before_idx, before_point = max(before_points, key=lambda x: datetime.fromisoformat(x[1]["timestamp"]) if isinstance(x[1]["timestamp"], str) else x[1]["timestamp"])
            after_idx, after_point = min(after_points, key=lambda x: datetime.fromisoformat(x[1]["timestamp"]) if isinstance(x[1]["timestamp"], str) else x[1]["timestamp"])
            
            # Convert timestamps to seconds for interpolation
            before_time = datetime.fromisoformat(before_point["timestamp"]) if isinstance(before_point["timestamp"], str) else before_point["timestamp"]
            after_time = datetime.fromisoformat(after_point["timestamp"]) if isinstance(after_point["timestamp"], str) else after_point["timestamp"]
            
            before_seconds = (before_time - start_time).total_seconds()
            after_seconds = (after_time - start_time).total_seconds()
            point_seconds = (time - start_time).total_seconds()
            
            # Calculate interpolation factor
            if after_seconds == before_seconds:
                factor = 0.0
            else:
                factor = (point_seconds - before_seconds) / (after_seconds - before_seconds)
            
            # Interpolate position and velocity
            position = [
                before_point["position"][i] + factor * (after_point["position"][i] - before_point["position"][i])
                for i in range(3)
            ]
            
            velocity = [
                before_point["velocity"][i] + factor * (after_point["velocity"][i] - before_point["velocity"][i])
                for i in range(3)
            ]
            
            interpolated_reference.append({
                "timestamp": point["timestamp"],
                "position": position,
                "velocity": velocity
            })
        
        return {
            "actual": filtered_trajectory,
            "reference": interpolated_reference
        }
    
    def _calculate_deviations(self, aligned_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate deviations between actual and reference trajectories.
        
        Args:
            aligned_data: Dictionary with aligned trajectory data
            
        Returns:
            Dictionary with deviation metrics
        """
        actual = aligned_data["actual"]
        reference = aligned_data["reference"]
        
        # Calculate position and velocity deviations
        position_deviations = []
        velocity_deviations = []
        
        for actual_point, reference_point in zip(actual, reference):
            # Position deviation
            pos_dev = np.linalg.norm(
                np.array(actual_point["position"]) - np.array(reference_point["position"])
            )
            position_deviations.append(pos_dev)
            
            # Velocity deviation
            vel_dev = np.linalg.norm(
                np.array(actual_point["velocity"]) - np.array(reference_point["velocity"])
            )
            velocity_deviations.append(vel_dev)
        
        # Calculate statistics
        avg_pos_dev = np.mean(position_deviations)
        max_pos_dev = np.max(position_deviations)
        std_pos_dev = np.std(position_deviations)
        
        avg_vel_dev = np.mean(velocity_deviations)
        max_vel_dev = np.max(velocity_deviations)
        std_vel_dev = np.std(velocity_deviations)
        
        return {
            "position": {
                "average": float(avg_pos_dev),
                "maximum": float(max_pos_dev),
                "std_dev": float(std_pos_dev),
                "values": [float(dev) for dev in position_deviations]
            },
            "velocity": {
                "average": float(avg_vel_dev),
                "maximum": float(max_vel_dev),
                "std_dev": float(std_vel_dev),
                "values": [float(dev) for dev in velocity_deviations]
            },
            "expected": {
                "position_km": 1.0,  # Expected deviation due to environmental factors
                "velocity_kms": 0.01  # Expected deviation due to environmental factors
            },
            "actual": {
                "position_km": float(avg_pos_dev),
                "velocity_kms": float(avg_vel_dev)
            }
        }
    
    def _analyze_deviations(self, deviations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze deviations to determine if they're environmental or intentional.
        
        Args:
            deviations: Dictionary with deviation metrics
            
        Returns:
            Dictionary with analysis results
        """
        # Get expected and actual deviations
        expected_pos_dev = deviations["expected"]["position_km"]
        actual_pos_dev = deviations["actual"]["position_km"]
        
        expected_vel_dev = deviations["expected"]["velocity_kms"]
        actual_vel_dev = deviations["actual"]["velocity_kms"]
        
        # Calculate deviation ratios
        pos_ratio = actual_pos_dev / expected_pos_dev if expected_pos_dev > 0 else float('inf')
        vel_ratio = actual_vel_dev / expected_vel_dev if expected_vel_dev > 0 else float('inf')
        
        # Determine if deviations are likely environmental or intentional
        # If the ratio is close to 1, it's likely environmental
        # If the ratio is much larger, it's likely intentional
        is_environmental_pos = pos_ratio < 2.0
        is_environmental_vel = vel_ratio < 2.0
        
        # Combined assessment
        is_environmental = is_environmental_pos and is_environmental_vel
        
        # Calculate confidence level
        if is_environmental:
            confidence_level = 1.0 - max(pos_ratio - 1.0, 0.0) / 1.0
            confidence_level = max(0.5, min(confidence_level, 0.95))
        else:
            confidence_level = 0.5 + min(pos_ratio - 2.0, 3.0) / 6.0
            confidence_level = max(0.5, min(confidence_level, 0.95))
        
        # Determine primary environmental factor
        # In a real implementation, this would be based on actual environmental data
        primary_factor = "atmospheric_drag"
        
        # Sample environmental factors
        environmental_factors = {
            "atmospheric_drag": 0.7,
            "solar_radiation": 0.2,
            "geomagnetic_activity": 0.1
        }
        
        return {
            "is_environmental": is_environmental,
            "confidence_level": float(confidence_level),
            "deviation_ratio": {
                "position": float(pos_ratio),
                "velocity": float(vel_ratio)
            },
            "primary_factor": primary_factor,
            "environmental_factors": environmental_factors
        } 