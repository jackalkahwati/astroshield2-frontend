"""RPO Shape Analysis Service for AstroShield.

This service provides machine learning-based analysis of Rendezvous and Proximity Operations (RPO)
shape patterns to classify maneuvers by behavior patterns and highlight suspicious activity.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from infrastructure.circuit_breaker import circuit_breaker
from infrastructure.monitoring import MonitoringService
from infrastructure.cache import CacheManager
from infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)
monitoring = MonitoringService()
cache = CacheManager()
event_bus = EventBus()

# RPO shape patterns and their descriptions
RPO_PATTERNS = {
    "inspection": "Circumnavigation or station-keeping near target",
    "rendezvous": "Gradual approach to target with decreasing relative velocity",
    "docking": "Final approach with very low relative velocity",
    "circumnavigation": "Orbit around target at relatively constant distance",
    "lead_trail": "Following target in similar orbit at constant distance",
    "fly_by": "Close approach followed by continued trajectory",
    "spiral": "Spiral approach or departure pattern",
    "zigzag": "Irregular approach with lateral movements",
    "hover": "Maintaining position relative to target with minimal movement",
    "intercept": "Direct, high-velocity approach toward target",
    "unknown": "Pattern does not match known RPO shapes"
}

# Suspicious patterns that may indicate hostile intent
SUSPICIOUS_PATTERNS = ["intercept", "zigzag", "spiral"]

class RPOShapeAnalysisService:
    """Service for analyzing RPO shape patterns using machine learning."""
    
    def __init__(self):
        """Initialize the RPO Shape Analysis service."""
        self.trajectory_buffer = {}  # Buffer to store recent trajectories
        self.buffer_ttl = timedelta(hours=24)  # Time to live for buffer entries
        self.min_trajectory_points = 10  # Minimum points needed for analysis
        self.pattern_models = {}  # ML models for pattern recognition
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize machine learning models for pattern recognition."""
        # In a real implementation, this would load trained models
        # For now, we'll use simple clustering for demonstration
        self.pattern_models = {
            "clustering": DBSCAN(eps=0.5, min_samples=5)
        }
        logger.info("RPO shape analysis models initialized")
    
    @circuit_breaker
    async def analyze_trajectory(self, spacecraft_id: str, trajectory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a spacecraft trajectory for RPO shape patterns.
        
        Args:
            spacecraft_id: ID of the spacecraft
            trajectory_data: List of trajectory points with position and timestamp
            
        Returns:
            Dictionary with analysis results
        """
        with monitoring.create_span("rpo_shape_analyze_trajectory") as span:
            try:
                span.set_attribute("spacecraft_id", spacecraft_id)
                span.set_attribute("trajectory_points", len(trajectory_data))
                
                # Validate trajectory data
                if len(trajectory_data) < self.min_trajectory_points:
                    return {
                        "spacecraft_id": spacecraft_id,
                        "status": "insufficient_data",
                        "message": f"Need at least {self.min_trajectory_points} trajectory points for analysis",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # Store trajectory in buffer
                buffer_key = f"{spacecraft_id}_{datetime.utcnow().isoformat()}"
                self.trajectory_buffer[buffer_key] = {
                    "spacecraft_id": spacecraft_id,
                    "trajectory": trajectory_data,
                    "timestamp": datetime.utcnow()
                }
                
                # Clean up old trajectories
                self._clean_buffer()
                
                # Extract features from trajectory
                features = self._extract_trajectory_features(trajectory_data)
                
                # Classify the trajectory shape
                shape_classification = self._classify_shape(features)
                
                # Determine if the pattern is suspicious
                is_suspicious = shape_classification["pattern"] in SUSPICIOUS_PATTERNS
                
                # Calculate additional metrics
                metrics = self._calculate_trajectory_metrics(trajectory_data)
                
                # Combine results
                result = {
                    "spacecraft_id": spacecraft_id,
                    "status": "analyzed",
                    "shape_classification": shape_classification,
                    "metrics": metrics,
                    "is_suspicious": is_suspicious,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Publish event
                event_bus.publish("rpo_shape_analyzed", {
                    "spacecraft_id": spacecraft_id,
                    "pattern": shape_classification["pattern"],
                    "confidence": shape_classification["confidence"],
                    "is_suspicious": is_suspicious,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # If suspicious, publish alert
                if is_suspicious and shape_classification["confidence"] > 0.7:
                    event_bus.publish("rpo_shape_alert", {
                        "spacecraft_id": spacecraft_id,
                        "pattern": shape_classification["pattern"],
                        "confidence": shape_classification["confidence"],
                        "alert_level": "high" if shape_classification["confidence"] > 0.85 else "medium",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"Error analyzing trajectory: {str(e)}")
                span.record_exception(e)
                return {
                    "spacecraft_id": spacecraft_id,
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    @circuit_breaker
    async def compare_trajectories(self, trajectory_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple trajectories to identify patterns across spacecraft.
        
        Args:
            trajectory_ids: List of trajectory IDs to compare
            
        Returns:
            Dictionary with comparison results
        """
        with monitoring.create_span("rpo_shape_compare_trajectories") as span:
            try:
                span.set_attribute("trajectory_count", len(trajectory_ids))
                
                # Get trajectories from buffer
                trajectories = []
                for traj_id in trajectory_ids:
                    if traj_id in self.trajectory_buffer:
                        trajectories.append(self.trajectory_buffer[traj_id])
                
                if len(trajectories) < 2:
                    return {
                        "status": "insufficient_data",
                        "message": "Need at least 2 valid trajectories for comparison",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # Extract features for each trajectory
                features_list = []
                for traj in trajectories:
                    features = self._extract_trajectory_features(traj["trajectory"])
                    features_list.append(features)
                
                # Calculate similarity matrix
                similarity_matrix = self._calculate_similarity_matrix(features_list)
                
                # Identify similar trajectories
                similar_pairs = self._identify_similar_trajectories(similarity_matrix, trajectories)
                
                # Determine if there's a coordinated pattern
                coordinated_pattern = len(similar_pairs) > 0
                
                result = {
                    "status": "analyzed",
                    "trajectory_count": len(trajectories),
                    "similarity_matrix": similarity_matrix.tolist(),
                    "similar_pairs": similar_pairs,
                    "coordinated_pattern": coordinated_pattern,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Publish event if coordinated pattern detected
                if coordinated_pattern:
                    event_bus.publish("rpo_coordinated_pattern_detected", {
                        "trajectory_count": len(trajectories),
                        "similar_pair_count": len(similar_pairs),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"Error comparing trajectories: {str(e)}")
                span.record_exception(e)
                return {
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    @circuit_breaker
    async def detect_anomalies(self, spacecraft_id: str, trajectory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect anomalies in a spacecraft trajectory.
        
        Args:
            spacecraft_id: ID of the spacecraft
            trajectory_data: List of trajectory points with position and timestamp
            
        Returns:
            Dictionary with anomaly detection results
        """
        with monitoring.create_span("rpo_shape_detect_anomalies") as span:
            try:
                span.set_attribute("spacecraft_id", spacecraft_id)
                span.set_attribute("trajectory_points", len(trajectory_data))
                
                # Validate trajectory data
                if len(trajectory_data) < self.min_trajectory_points:
                    return {
                        "spacecraft_id": spacecraft_id,
                        "status": "insufficient_data",
                        "message": f"Need at least {self.min_trajectory_points} trajectory points for analysis",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # Extract features
                features = self._extract_trajectory_features(trajectory_data)
                
                # Detect anomalies
                anomalies = self._detect_trajectory_anomalies(trajectory_data)
                
                # Calculate anomaly score
                anomaly_score = sum(a["severity"] for a in anomalies) / len(anomalies) if anomalies else 0.0
                
                result = {
                    "spacecraft_id": spacecraft_id,
                    "status": "analyzed",
                    "anomalies": anomalies,
                    "anomaly_score": anomaly_score,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Publish event if anomalies detected
                if anomalies:
                    event_bus.publish("rpo_anomalies_detected", {
                        "spacecraft_id": spacecraft_id,
                        "anomaly_count": len(anomalies),
                        "anomaly_score": anomaly_score,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"Error detecting anomalies: {str(e)}")
                span.record_exception(e)
                return {
                    "spacecraft_id": spacecraft_id,
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    def _clean_buffer(self) -> None:
        """Clean up old trajectories from the buffer."""
        current_time = datetime.utcnow()
        to_remove = []
        
        for traj_id, traj in self.trajectory_buffer.items():
            if current_time - traj["timestamp"] > self.buffer_ttl:
                to_remove.append(traj_id)
        
        for traj_id in to_remove:
            del self.trajectory_buffer[traj_id]
        
        if to_remove:
            logger.info(f"Removed {len(to_remove)} old trajectories from buffer")
    
    def _extract_trajectory_features(self, trajectory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from a trajectory for shape analysis.
        
        Args:
            trajectory_data: List of trajectory points
            
        Returns:
            Dictionary of trajectory features
        """
        # Extract positions and timestamps
        positions = np.array([point["position"] for point in trajectory_data if "position" in point])
        timestamps = np.array([
            (datetime.fromisoformat(point["timestamp"]) if isinstance(point["timestamp"], str) else point["timestamp"]).timestamp() 
            for point in trajectory_data if "timestamp" in point
        ])
        
        if len(positions) < 2:
            return {"valid": False}
        
        # Calculate basic features
        start_pos = positions[0]
        end_pos = positions[-1]
        total_distance = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
        straight_line_distance = np.linalg.norm(end_pos - start_pos)
        
        # Calculate path efficiency (ratio of straight line to actual path)
        path_efficiency = straight_line_distance / total_distance if total_distance > 0 else 0.0
        
        # Calculate velocity and acceleration
        time_diffs = np.diff(timestamps)
        velocities = np.diff(positions, axis=0) / time_diffs[:, np.newaxis]
        accelerations = np.diff(velocities, axis=0) / time_diffs[:-1, np.newaxis]
        
        # Calculate average velocity and acceleration magnitudes
        velocity_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        acceleration_magnitudes = np.sqrt(np.sum(accelerations**2, axis=1))
        avg_velocity = np.mean(velocity_magnitudes)
        avg_acceleration = np.mean(acceleration_magnitudes)
        
        # Calculate trajectory curvature
        curvature = self._calculate_trajectory_curvature(positions)
        
        # Calculate trajectory smoothness
        smoothness = self._calculate_trajectory_smoothness(velocities)
        
        # Calculate trajectory periodicity
        periodicity = self._calculate_trajectory_periodicity(positions)
        
        return {
            "valid": True,
            "path_efficiency": path_efficiency,
            "avg_velocity": avg_velocity,
            "avg_acceleration": avg_acceleration,
            "curvature": curvature,
            "smoothness": smoothness,
            "periodicity": periodicity,
            "total_distance": total_distance,
            "straight_line_distance": straight_line_distance,
            "duration": timestamps[-1] - timestamps[0]
        }
    
    def _calculate_trajectory_curvature(self, positions: np.ndarray) -> float:
        """Calculate the average curvature of a trajectory.
        
        Args:
            positions: Array of positions
            
        Returns:
            Average curvature value
        """
        if len(positions) < 3:
            return 0.0
        
        # Calculate vectors between consecutive points
        vectors = np.diff(positions, axis=0)
        
        # Calculate angles between consecutive vectors
        dot_products = np.sum(vectors[:-1] * vectors[1:], axis=1)
        norms = np.linalg.norm(vectors[:-1], axis=1) * np.linalg.norm(vectors[1:], axis=1)
        
        # Avoid division by zero
        norms = np.where(norms > 0, norms, 1e-10)
        
        # Calculate cosine of angles
        cos_angles = np.clip(dot_products / norms, -1.0, 1.0)
        
        # Calculate angles in radians
        angles = np.arccos(cos_angles)
        
        # Return average angle as curvature measure
        return np.mean(angles)
    
    def _calculate_trajectory_smoothness(self, velocities: np.ndarray) -> float:
        """Calculate the smoothness of a trajectory based on velocity changes.
        
        Args:
            velocities: Array of velocities
            
        Returns:
            Smoothness value between 0 and 1
        """
        if len(velocities) < 2:
            return 1.0
        
        # Calculate velocity changes
        velocity_changes = np.diff(velocities, axis=0)
        
        # Calculate magnitude of velocity changes
        change_magnitudes = np.sqrt(np.sum(velocity_changes**2, axis=1))
        
        # Calculate average change magnitude
        avg_change = np.mean(change_magnitudes)
        
        # Calculate average velocity magnitude
        avg_velocity = np.mean(np.sqrt(np.sum(velocities**2, axis=1)))
        
        # Calculate smoothness (lower change relative to velocity is smoother)
        if avg_velocity > 0:
            smoothness = 1.0 - min(avg_change / avg_velocity, 1.0)
        else:
            smoothness = 1.0
        
        return smoothness
    
    def _calculate_trajectory_periodicity(self, positions: np.ndarray) -> float:
        """Calculate the periodicity of a trajectory.
        
        Args:
            positions: Array of positions
            
        Returns:
            Periodicity value between 0 and 1
        """
        if len(positions) < 10:
            return 0.0
        
        # Calculate distances from start point
        distances_from_start = np.sqrt(np.sum((positions - positions[0])**2, axis=1))
        
        # Check if trajectory returns close to start point
        min_dist_after_midpoint = np.min(distances_from_start[len(distances_from_start)//2:])
        max_dist = np.max(distances_from_start)
        
        # Calculate periodicity (lower minimum distance indicates return to start)
        if max_dist > 0:
            periodicity = 1.0 - min(min_dist_after_midpoint / max_dist, 1.0)
        else:
            periodicity = 0.0
        
        return periodicity
    
    def _classify_shape(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the shape of a trajectory based on extracted features.
        
        Args:
            features: Dictionary of trajectory features
            
        Returns:
            Dictionary with classification results
        """
        if not features["valid"]:
            return {
                "pattern": "unknown",
                "confidence": 0.0,
                "description": RPO_PATTERNS["unknown"]
            }
        
        # Define feature thresholds for different patterns
        # In a real implementation, this would use ML models
        
        # Check for inspection pattern (high curvature, high periodicity)
        if features["curvature"] > 0.5 and features["periodicity"] > 0.6:
            return {
                "pattern": "inspection",
                "confidence": 0.7 + 0.3 * min(features["periodicity"], 1.0),
                "description": RPO_PATTERNS["inspection"]
            }
        
        # Check for rendezvous pattern (high path efficiency, decreasing velocity)
        if features["path_efficiency"] > 0.8 and features["avg_acceleration"] < 0:
            return {
                "pattern": "rendezvous",
                "confidence": 0.7 + 0.3 * features["path_efficiency"],
                "description": RPO_PATTERNS["rendezvous"]
            }
        
        # Check for circumnavigation (high curvature, high periodicity, consistent distance)
        if features["curvature"] > 0.4 and features["periodicity"] > 0.7 and features["smoothness"] > 0.7:
            return {
                "pattern": "circumnavigation",
                "confidence": 0.7 + 0.3 * features["periodicity"],
                "description": RPO_PATTERNS["circumnavigation"]
            }
        
        # Check for fly-by (high path efficiency, high velocity)
        if features["path_efficiency"] > 0.9 and features["avg_velocity"] > 1.0:
            return {
                "pattern": "fly_by",
                "confidence": 0.7 + 0.3 * features["path_efficiency"],
                "description": RPO_PATTERNS["fly_by"]
            }
        
        # Check for intercept (high path efficiency, high velocity, high acceleration)
        if features["path_efficiency"] > 0.9 and features["avg_velocity"] > 2.0 and features["avg_acceleration"] > 0.5:
            return {
                "pattern": "intercept",
                "confidence": 0.8 + 0.2 * features["path_efficiency"],
                "description": RPO_PATTERNS["intercept"]
            }
        
        # Check for hover (low total distance, low velocity)
        if features["total_distance"] < 10.0 and features["avg_velocity"] < 0.1:
            return {
                "pattern": "hover",
                "confidence": 0.7 + 0.3 * (1.0 - features["avg_velocity"]),
                "description": RPO_PATTERNS["hover"]
            }
        
        # Check for zigzag (low path efficiency, high curvature)
        if features["path_efficiency"] < 0.5 and features["curvature"] > 0.7:
            return {
                "pattern": "zigzag",
                "confidence": 0.7 + 0.3 * (1.0 - features["path_efficiency"]),
                "description": RPO_PATTERNS["zigzag"]
            }
        
        # Default to unknown
        return {
            "pattern": "unknown",
            "confidence": 0.5,
            "description": RPO_PATTERNS["unknown"]
        }
    
    def _calculate_trajectory_metrics(self, trajectory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate additional metrics for a trajectory.
        
        Args:
            trajectory_data: List of trajectory points
            
        Returns:
            Dictionary of metrics
        """
        # Extract positions and timestamps
        positions = np.array([point["position"] for point in trajectory_data if "position" in point])
        timestamps = np.array([
            (datetime.fromisoformat(point["timestamp"]) if isinstance(point["timestamp"], str) else point["timestamp"]).timestamp() 
            for point in trajectory_data if "timestamp" in point
        ])
        
        if len(positions) < 2:
            return {}
        
        # Calculate time-based metrics
        duration = timestamps[-1] - timestamps[0]
        
        # Calculate distance-based metrics
        total_distance = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
        max_distance_from_start = np.max(np.sqrt(np.sum((positions - positions[0])**2, axis=1)))
        
        # Calculate velocity-based metrics
        time_diffs = np.diff(timestamps)
        velocities = np.diff(positions, axis=0) / time_diffs[:, np.newaxis]
        velocity_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        max_velocity = np.max(velocity_magnitudes)
        min_velocity = np.min(velocity_magnitudes)
        
        return {
            "duration_seconds": duration,
            "total_distance_km": float(total_distance),
            "max_distance_from_start_km": float(max_distance_from_start),
            "max_velocity_kmps": float(max_velocity),
            "min_velocity_kmps": float(min_velocity),
            "avg_velocity_kmps": float(np.mean(velocity_magnitudes))
        }
    
    def _detect_trajectory_anomalies(self, trajectory_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in a trajectory.
        
        Args:
            trajectory_data: List of trajectory points
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Extract positions and timestamps
        positions = np.array([point["position"] for point in trajectory_data if "position" in point])
        timestamps = np.array([
            (datetime.fromisoformat(point["timestamp"]) if isinstance(point["timestamp"], str) else point["timestamp"]).timestamp() 
            for point in trajectory_data if "timestamp" in point
        ])
        
        if len(positions) < 3:
            return anomalies
        
        # Calculate velocities
        time_diffs = np.diff(timestamps)
        velocities = np.diff(positions, axis=0) / time_diffs[:, np.newaxis]
        velocity_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        
        # Calculate accelerations
        accelerations = np.diff(velocities, axis=0) / time_diffs[:-1, np.newaxis]
        acceleration_magnitudes = np.sqrt(np.sum(accelerations**2, axis=1))
        
        # Detect sudden velocity changes
        velocity_changes = np.diff(velocity_magnitudes)
        for i, change in enumerate(velocity_changes):
            if abs(change) > 0.5:  # Threshold for sudden change
                anomalies.append({
                    "type": "sudden_velocity_change",
                    "index": i + 1,
                    "timestamp": datetime.fromtimestamp(timestamps[i + 1]).isoformat(),
                    "value": float(change),
                    "severity": min(abs(change) / 2.0, 1.0)
                })
        
        # Detect sudden direction changes
        for i in range(1, len(velocities) - 1):
            v1 = velocities[i-1]
            v2 = velocities[i]
            
            # Calculate angle between velocity vectors
            dot_product = np.sum(v1 * v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norm_product > 0:
                cos_angle = dot_product / norm_product
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                if angle > 0.5:  # Threshold for sudden direction change (in radians)
                    anomalies.append({
                        "type": "sudden_direction_change",
                        "index": i,
                        "timestamp": datetime.fromtimestamp(timestamps[i]).isoformat(),
                        "value": float(angle),
                        "severity": min(angle / np.pi, 1.0)
                    })
        
        # Detect unusual accelerations
        for i, acc_mag in enumerate(acceleration_magnitudes):
            if acc_mag > 1.0:  # Threshold for unusual acceleration
                anomalies.append({
                    "type": "unusual_acceleration",
                    "index": i + 1,
                    "timestamp": datetime.fromtimestamp(timestamps[i + 1]).isoformat(),
                    "value": float(acc_mag),
                    "severity": min(acc_mag / 5.0, 1.0)
                })
        
        return anomalies
    
    def _calculate_similarity_matrix(self, features_list: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate similarity matrix between multiple trajectory features.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Similarity matrix
        """
        n = len(features_list)
        similarity_matrix = np.zeros((n, n))
        
        # Extract feature vectors
        feature_vectors = []
        for features in features_list:
            if not features.get("valid", False):
                feature_vectors.append(np.zeros(6))
                continue
                
            vector = np.array([
                features.get("path_efficiency", 0.0),
                features.get("avg_velocity", 0.0),
                features.get("avg_acceleration", 0.0),
                features.get("curvature", 0.0),
                features.get("smoothness", 0.0),
                features.get("periodicity", 0.0)
            ])
            feature_vectors.append(vector)
        
        # Calculate pairwise distances
        feature_vectors = np.array(feature_vectors)
        distance_matrix = cdist(feature_vectors, feature_vectors, 'euclidean')
        
        # Convert distances to similarities
        max_distance = np.max(distance_matrix) if np.max(distance_matrix) > 0 else 1.0
        similarity_matrix = 1.0 - distance_matrix / max_distance
        
        return similarity_matrix
    
    def _identify_similar_trajectories(self, similarity_matrix: np.ndarray, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify pairs of similar trajectories.
        
        Args:
            similarity_matrix: Matrix of similarity scores
            trajectories: List of trajectory data
            
        Returns:
            List of similar trajectory pairs
        """
        similar_pairs = []
        
        # Find pairs with high similarity
        threshold = 0.8
        n = similarity_matrix.shape[0]
        
        for i in range(n):
            for j in range(i+1, n):
                if similarity_matrix[i, j] > threshold:
                    similar_pairs.append({
                        "spacecraft_1": trajectories[i]["spacecraft_id"],
                        "spacecraft_2": trajectories[j]["spacecraft_id"],
                        "similarity_score": float(similarity_matrix[i, j])
                    })
        
        return similar_pairs 