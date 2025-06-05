"""
Subsystem 2: State Estimation
UCT processing, orbit determination, and state vector management
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import logging
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize

from ..kafka.kafka_client import (
    WeldersArcKafkaClient, 
    WeldersArcMessage, 
    KafkaTopics, 
    SubsystemID
)

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Observation track data"""
    track_id: str
    sensor_id: str
    timestamp: datetime
    position: np.ndarray  # [x, y, z] in ECI
    velocity: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


@dataclass
class StateVector:
    """Orbital state vector"""
    object_id: str
    epoch: datetime
    position: np.ndarray  # [x, y, z] km
    velocity: np.ndarray  # [vx, vy, vz] km/s
    covariance: np.ndarray  # 6x6 covariance matrix
    quality_score: float
    source: str


class UCTProcessor:
    """Uncorrelated Track Processing Engine"""
    
    def __init__(self):
        self.pending_tracks: List[Track] = []
        self.correlation_threshold = 0.95
        self.min_tracks_for_od = 3
        
    async def process_track(self, track: Track) -> Optional[str]:
        """Process a single uncorrelated track"""
        # Try to correlate with existing tracks
        correlation = await self._correlate_track(track)
        
        if correlation:
            return correlation
            
        # Add to pending tracks
        self.pending_tracks.append(track)
        
        # Try to form new object
        if len(self.pending_tracks) >= self.min_tracks_for_od:
            return await self._attempt_orbit_determination()
            
        return None
        
    async def _correlate_track(self, track: Track) -> Optional[str]:
        """Attempt to correlate track with known objects"""
        # Simple distance-based correlation
        best_match = None
        best_score = 0
        
        # This would query the catalog
        # For now, return None (no correlation found)
        return None
        
    async def _attempt_orbit_determination(self) -> Optional[str]:
        """Attempt orbit determination from pending tracks"""
        # Use DBSCAN to cluster tracks
        positions = np.array([t.position for t in self.pending_tracks])
        times = np.array([(t.timestamp - self.pending_tracks[0].timestamp).total_seconds() 
                         for t in self.pending_tracks])
        
        # Normalize for clustering
        data = np.column_stack([positions, times.reshape(-1, 1)])
        clustering = DBSCAN(eps=100, min_samples=self.min_tracks_for_od).fit(data)
        
        # Process each cluster
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise
                continue
                
            cluster_tracks = [self.pending_tracks[i] 
                            for i, label in enumerate(clustering.labels_) 
                            if label == cluster_id]
            
            if len(cluster_tracks) >= self.min_tracks_for_od:
                state_vector = await self._fit_orbit(cluster_tracks)
                if state_vector:
                    # Remove used tracks
                    self.pending_tracks = [t for t in self.pending_tracks 
                                         if t not in cluster_tracks]
                    return state_vector.object_id
                    
        return None
        
    async def _fit_orbit(self, tracks: List[Track]) -> Optional[StateVector]:
        """Fit orbit to track observations"""
        # Initial orbit determination using Gibbs method
        if len(tracks) >= 3:
            r1 = tracks[0].position
            r2 = tracks[1].position
            r3 = tracks[2].position
            
            # Gibbs method implementation
            state = self._gibbs_method(r1, r2, r3)
            
            if state:
                # Refine with least squares
                refined_state = await self._refine_orbit(state, tracks)
                
                return StateVector(
                    object_id=f"UCT-{datetime.utcnow().timestamp()}",
                    epoch=tracks[0].timestamp,
                    position=refined_state[:3],
                    velocity=refined_state[3:],
                    covariance=np.eye(6) * 0.1,  # Placeholder
                    quality_score=0.8,
                    source="UCT_PROCESSOR"
                )
                
        return None
        
    def _gibbs_method(self, r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> Optional[np.ndarray]:
        """Gibbs method for initial orbit determination"""
        # Constants
        mu = 398600.4418  # Earth gravitational parameter km^3/s^2
        
        # Cross products
        c12 = np.cross(r1, r2)
        c23 = np.cross(r2, r3)
        c31 = np.cross(r3, r1)
        
        # Check coplanarity
        if np.abs(np.dot(r1, c23)) > 1e-6:
            # Velocity at r2
            n = np.linalg.norm(r2)
            d = c12 + c23 + c31
            s = r1 * (np.linalg.norm(r2) - np.linalg.norm(r3)) + \
                r2 * (np.linalg.norm(r3) - np.linalg.norm(r1)) + \
                r3 * (np.linalg.norm(r1) - np.linalg.norm(r2))
            
            v2 = np.sqrt(mu / (n * np.linalg.norm(d))) * \
                 (np.cross(d, r2) / n + s)
            
            return np.concatenate([r2, v2])
            
        return None
        
    async def _refine_orbit(self, initial_state: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Refine orbit using least squares"""
        def residuals(state):
            total_residual = 0
            for track in tracks:
                # Propagate state to track time
                propagated = self._propagate_state(state, track.timestamp)
                residual = np.linalg.norm(propagated[:3] - track.position)
                total_residual += residual ** 2
            return total_residual
            
        result = minimize(residuals, initial_state, method='L-BFGS-B')
        return result.x
        
    def _propagate_state(self, state: np.ndarray, target_time: datetime) -> np.ndarray:
        """Two-body propagation with J2 perturbation"""
        # Constants
        mu = 398600.4418  # Earth gravitational parameter km^3/s^2
        J2 = 1.08263e-3   # Earth's J2 coefficient
        Re = 6378.137     # Earth radius km
        
        # Extract state
        r0 = state[:3]
        v0 = state[3:]
        
        # Time difference
        dt = (target_time - datetime.utcnow()).total_seconds()
        
        # Calculate orbital elements
        r0_mag = np.linalg.norm(r0)
        v0_mag = np.linalg.norm(v0)
        
        # Specific angular momentum
        h = np.cross(r0, v0)
        h_mag = np.linalg.norm(h)
        
        # Eccentricity vector
        e_vec = ((v0_mag**2 - mu/r0_mag) * r0 - np.dot(r0, v0) * v0) / mu
        e = np.linalg.norm(e_vec)
        
        # Semi-major axis
        a = 1 / (2/r0_mag - v0_mag**2/mu)
        
        # Mean motion with J2 correction
        n = np.sqrt(mu / a**3)
        i = np.arccos(h[2] / h_mag)
        
        # J2 perturbation on mean motion
        n_J2 = n * (1 + 1.5 * J2 * (Re/a)**2 * (1 - e**2)**(-1.5) * (1 - 1.5 * np.sin(i)**2))
        
        # Propagate mean anomaly
        M0 = self._true_to_mean_anomaly(r0, v0, e, a)
        M = M0 + n_J2 * dt
        
        # Solve Kepler's equation
        E = self._solve_kepler(M, e)
        
        # True anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
        
        # Position and velocity in perifocal frame
        p = a * (1 - e**2)
        r_pf = p / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0])
        v_pf = np.sqrt(mu/p) * np.array([-np.sin(nu), e + np.cos(nu), 0])
        
        # Rotation matrices
        omega = np.arctan2(e_vec[1], e_vec[0])
        Omega = np.arctan2(h[0], -h[1])
        
        # Transform back to ECI
        r_new = self._perifocal_to_eci(r_pf, omega, Omega, i)
        v_new = self._perifocal_to_eci(v_pf, omega, Omega, i)
        
        return np.concatenate([r_new, v_new])
        
    def _true_to_mean_anomaly(self, r: np.ndarray, v: np.ndarray, e: float, a: float) -> float:
        """Convert position/velocity to mean anomaly"""
        mu = 398600.4418
        
        # Eccentric anomaly
        E = np.arctan2(np.dot(r, v) / np.sqrt(mu * a), 1 - np.linalg.norm(r) / a)
        
        # Mean anomaly
        M = E - e * np.sin(E)
        return M
        
    def _solve_kepler(self, M: float, e: float, tol: float = 1e-8) -> float:
        """Solve Kepler's equation using Newton-Raphson"""
        E = M  # Initial guess
        
        for _ in range(10):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            E_new = E - f / f_prime
            
            if abs(E_new - E) < tol:
                return E_new
            E = E_new
            
        return E
        
    def _perifocal_to_eci(self, vec_pf: np.ndarray, omega: float, Omega: float, i: float) -> np.ndarray:
        """Transform from perifocal to ECI frame"""
        # Rotation matrices
        R3_omega = np.array([
            [np.cos(omega), -np.sin(omega), 0],
            [np.sin(omega), np.cos(omega), 0],
            [0, 0, 1]
        ])
        
        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i), np.cos(i)]
        ])
        
        R3_Omega = np.array([
            [np.cos(Omega), -np.sin(Omega), 0],
            [np.sin(Omega), np.cos(Omega), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = R3_Omega @ R1_i @ R3_omega
        
        return R @ vec_pf


class StateEstimator:
    """Main state estimation subsystem"""
    
    def __init__(self, kafka_client: WeldersArcKafkaClient):
        self.kafka_client = kafka_client
        self.uct_processor = UCTProcessor()
        self.state_catalog: Dict[str, StateVector] = {}
        self.ensemble_processors: List[UCTProcessor] = []
        
    async def initialize(self):
        """Initialize state estimation subsystem"""
        # Subscribe to relevant topics
        self.kafka_client.subscribe(
            KafkaTopics.UCT_TRACKS,
            self._handle_uct_track
        )
        
        self.kafka_client.subscribe(
            KafkaTopics.SENSOR_OBSERVATIONS,
            self._handle_observation
        )
        
        # Initialize ensemble of UCT processors
        for i in range(5):
            processor = UCTProcessor()
            # Vary parameters for ensemble diversity
            processor.correlation_threshold = 0.9 + i * 0.02
            processor.min_tracks_for_od = 3 + (i % 2)
            self.ensemble_processors.append(processor)
            
        logger.info("State estimation subsystem initialized")
        
    async def _handle_uct_track(self, message: WeldersArcMessage):
        """Handle incoming UCT track"""
        track_data = message.data
        track = Track(
            track_id=track_data["track_id"],
            sensor_id=track_data["sensor_id"],
            timestamp=datetime.fromisoformat(track_data["timestamp"]),
            position=np.array(track_data["position"]),
            velocity=np.array(track_data.get("velocity")) if "velocity" in track_data else None,
            metadata=track_data.get("metadata", {})
        )
        
        # Process through ensemble
        results = await asyncio.gather(*[
            processor.process_track(track) 
            for processor in self.ensemble_processors
        ])
        
        # Aggregate results
        valid_results = [r for r in results if r is not None]
        if valid_results:
            # Majority voting or consensus
            await self._handle_new_object(valid_results[0])
            
    async def _handle_observation(self, message: WeldersArcMessage):
        """Handle correlated observation for state update"""
        obs_data = message.data
        object_id = obs_data.get("object_id")
        
        if object_id and object_id in self.state_catalog:
            # Update state vector
            await self._update_state_vector(object_id, obs_data)
            
    async def _update_state_vector(self, object_id: str, observation: Dict[str, Any]):
        """Update object state vector with new observation using Extended Kalman Filter"""
        current_state = self.state_catalog[object_id]
        
        # Extract observation data
        obs_time = datetime.fromisoformat(observation["timestamp"])
        obs_position = np.array(observation["position"])
        obs_covariance = np.array(observation.get("covariance", np.eye(3) * 0.1))
        
        # Propagate current state to observation time
        propagated_state = self._propagate_state(
            np.concatenate([current_state.position, current_state.velocity]),
            obs_time
        )
        
        # Extended Kalman Filter update
        # State transition matrix (simplified)
        dt = (obs_time - current_state.epoch).total_seconds()
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Process noise
        Q = np.eye(6) * 1e-6
        Q[3:6, 3:6] *= 100  # Higher uncertainty in velocity
        
        # Measurement matrix (position only)
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)
        
        # Predict
        x_pred = propagated_state
        P_pred = F @ current_state.covariance @ F.T + Q
        
        # Update
        y = obs_position - H @ x_pred  # Innovation
        S = H @ P_pred @ H.T + obs_covariance  # Innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        # Updated state and covariance
        x_updated = x_pred + K @ y
        P_updated = (np.eye(6) - K @ H) @ P_pred
        
        # Create updated state vector
        updated_state = StateVector(
            object_id=object_id,
            epoch=obs_time,
            position=x_updated[:3],
            velocity=x_updated[3:],
            covariance=P_updated,
            quality_score=self._calculate_quality_score(P_updated, y, S),
            source="EKF_UPDATE"
        )
        
        # Update catalog
        self.state_catalog[object_id] = updated_state
        
        # Publish updated state
        message = WeldersArcMessage(
            message_id=f"state-update-{object_id}-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS2_STATE_ESTIMATION,
            event_type="state_update",
            data={
                "object_id": object_id,
                "state_vector": {
                    "position": updated_state.position.tolist(),
                    "velocity": updated_state.velocity.tolist(),
                    "epoch": updated_state.epoch.isoformat(),
                    "quality_score": updated_state.quality_score,
                    "innovation": y.tolist(),
                    "innovation_covariance": S.tolist()
                }
            }
        )
        
        await self.kafka_client.publish(KafkaTopics.STATE_VECTORS, message)
        
    def _calculate_quality_score(self, covariance: np.ndarray, innovation: np.ndarray, 
                                 innovation_cov: np.ndarray) -> float:
        """Calculate state quality score based on covariance and innovation"""
        # Normalized innovation squared
        nis = innovation.T @ np.linalg.inv(innovation_cov) @ innovation
        
        # Chi-squared test (3 DOF, 95% confidence)
        chi2_threshold = 7.815
        
        # Covariance trace (uncertainty measure)
        pos_uncertainty = np.sqrt(np.trace(covariance[:3, :3]))
        
        # Quality score
        quality = 1.0
        
        # Penalize high innovation
        if nis > chi2_threshold:
            quality *= np.exp(-0.1 * (nis - chi2_threshold))
            
        # Penalize high uncertainty
        if pos_uncertainty > 1.0:  # 1 km threshold
            quality *= np.exp(-0.1 * (pos_uncertainty - 1.0))
            
        return min(1.0, max(0.0, quality))
        
    async def _handle_new_object(self, object_id: str):
        """Handle newly identified object"""
        logger.info(f"New object identified: {object_id}")
        
        # Publish to catalog correlation topic
        message = WeldersArcMessage(
            message_id=f"new-object-{object_id}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS2_STATE_ESTIMATION,
            event_type="new_object",
            data={
                "object_id": object_id,
                "detection_time": datetime.utcnow().isoformat(),
                "source": "UCT_PROCESSING"
            }
        )
        
        await self.kafka_client.publish(KafkaTopics.CATALOG_CORRELATION, message)
        
    async def get_state_prediction(
        self, 
        object_id: str, 
        target_time: datetime
    ) -> Optional[StateVector]:
        """Get predicted state at target time"""
        if object_id not in self.state_catalog:
            return None
            
        current_state = self.state_catalog[object_id]
        
        # Propagate to target time
        # Placeholder - would use proper propagator
        return current_state
        
    async def process_maneuver_hypothesis(
        self,
        object_id: str,
        maneuver_time: datetime,
        observations_pre: List[Dict],
        observations_post: List[Dict]
    ) -> Dict[str, Any]:
        """Process potential maneuver detection"""
        # Fit orbits before and after
        if len(observations_pre) < 3 or len(observations_post) < 3:
            return {
                "maneuver_detected": False,
                "reason": "Insufficient observations",
                "confidence": 0.0
            }
            
        # Convert observations to Track objects
        tracks_pre = []
        for obs in observations_pre:
            track = Track(
                track_id=obs["id"],
                sensor_id=obs["sensor_id"],
                timestamp=datetime.fromisoformat(obs["timestamp"]),
                position=np.array(obs["position"])
            )
            tracks_pre.append(track)
            
        tracks_post = []
        for obs in observations_post:
            track = Track(
                track_id=obs["id"],
                sensor_id=obs["sensor_id"],
                timestamp=datetime.fromisoformat(obs["timestamp"]),
                position=np.array(obs["position"])
            )
            tracks_post.append(track)
            
        # Fit orbits
        orbit_pre = await self.uct_processor._fit_orbit(tracks_pre)
        orbit_post = await self.uct_processor._fit_orbit(tracks_post)
        
        if not orbit_pre or not orbit_post:
            return {
                "maneuver_detected": False,
                "reason": "Unable to fit orbits",
                "confidence": 0.0
            }
            
        # Calculate delta-V
        # Propagate pre-maneuver orbit to maneuver time
        state_pre_at_maneuver = self._propagate_state(
            np.concatenate([orbit_pre.position, orbit_pre.velocity]),
            maneuver_time
        )
        
        # Propagate post-maneuver orbit back to maneuver time
        state_post_at_maneuver = self._propagate_state(
            np.concatenate([orbit_post.position, orbit_post.velocity]),
            maneuver_time
        )
        
        # Delta-V calculation
        delta_v = state_post_at_maneuver[3:] - state_pre_at_maneuver[3:]
        delta_v_magnitude = np.linalg.norm(delta_v)
        
        # Maneuver detection threshold (10 m/s)
        maneuver_threshold = 0.01  # km/s
        
        if delta_v_magnitude > maneuver_threshold:
            # Calculate confidence based on orbit quality and delta-V magnitude
            confidence = min(0.95, 0.5 + 0.5 * (delta_v_magnitude / 0.1))
            
            # Classify maneuver type
            maneuver_type = self._classify_maneuver(delta_v, state_pre_at_maneuver)
            
            return {
                "maneuver_detected": True,
                "delta_v": delta_v.tolist(),
                "delta_v_magnitude": delta_v_magnitude,
                "maneuver_type": maneuver_type,
                "maneuver_time": maneuver_time.isoformat(),
                "confidence": confidence,
                "orbit_quality_pre": orbit_pre.quality_score,
                "orbit_quality_post": orbit_post.quality_score
            }
        else:
            return {
                "maneuver_detected": False,
                "delta_v_magnitude": delta_v_magnitude,
                "reason": "Delta-V below threshold",
                "confidence": 0.85
            }
            
    def _classify_maneuver(self, delta_v: np.ndarray, state: np.ndarray) -> str:
        """Classify maneuver type based on delta-V direction"""
        # Get orbital frame vectors
        r = state[:3]
        v = state[3:]
        
        # Radial, along-track, cross-track frame
        r_hat = r / np.linalg.norm(r)
        h = np.cross(r, v)
        h_hat = h / np.linalg.norm(h)
        s_hat = np.cross(h_hat, r_hat)
        
        # Project delta-V onto RSW frame
        dv_r = np.dot(delta_v, r_hat)
        dv_s = np.dot(delta_v, s_hat)
        dv_w = np.dot(delta_v, h_hat)
        
        # Classify based on dominant component
        if abs(dv_s) > abs(dv_r) and abs(dv_s) > abs(dv_w):
            if dv_s > 0:
                return "altitude_raise"
            else:
                return "altitude_lower"
        elif abs(dv_r) > abs(dv_w):
            return "phasing_maneuver"
        else:
            return "inclination_change" 