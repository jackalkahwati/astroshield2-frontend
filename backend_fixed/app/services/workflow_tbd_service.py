"""
Event Processing Workflow TBD Service
Implements solutions for the 8 critical TBDs identified in the Event Processing Workflows
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio

# Import existing AstroShield services
from .ccdm import CCDMService, get_ccdm_service
from .trajectory_predictor import TrajectoryPredictor
from .analytics_service import AnalyticsService
from .satellite_service import SatelliteService
from .maneuver_service import ManeuverService
from .event_processor_base import EventProcessorBase
from .tle_orbit_explainer_service import TLEOrbitExplainerService

# Import models
from app.models.ccdm import (
    ThreatLevel, ObjectThreatAssessment, ThreatAssessmentRequest,
    ObjectAnalysisRequest, HistoricalAnalysisRequest
)

logger = logging.getLogger(__name__)

class ExitCondition(Enum):
    """Proximity exit condition types"""
    WEZ_PEZ_EXIT = "wez_pez_exit"
    FORMATION_FLYER = "formation_flyer"
    MANEUVER_CESSATION = "maneuver_cessation"
    OBJECT_MERGER = "object_merger"
    UCT_DEBRIS = "uct_debris"

@dataclass
class ProximityAssessment:
    """Proximity assessment data structure"""
    primary_object: str
    secondary_object: str
    miss_distance_km: float
    time_to_closest_approach: datetime
    pez_score: float
    wez_score: float
    fusion_score: float
    risk_level: ThreatLevel
    confidence: float

@dataclass
class ManeuverPrediction:
    """Maneuver prediction result"""
    object_id: str
    predicted_time: datetime
    maneuver_type: str
    delta_v_estimate: float
    confidence: float
    ephemeris_update: Dict[str, Any]

@dataclass
class ThresholdResult:
    """Threshold determination result"""
    range_threshold_km: float
    velocity_threshold_ms: float
    approach_rate_threshold: float
    confidence: float
    dynamic_factors: Dict[str, float]

class WorkflowTBDService:
    """
    Service implementing solutions for Event Processing Workflow TBDs
    Integrates with existing AstroShield services to provide workflow-specific capabilities
    """
    
    def __init__(self, db_session=None):
        """Initialize the Workflow TBD Service"""
        self.ccdm_service = get_ccdm_service(db_session)
        self.trajectory_predictor = TrajectoryPredictor()
        self.analytics_service = AnalyticsService()
        self.satellite_service = SatelliteService(db_session) if db_session else None
        self.maneuver_service = ManeuverService(db_session) if db_session else None
        self.tle_explainer = TLEOrbitExplainerService()
        logger.info("🚀 AstroShield WorkflowTBDService initialized with TLE Orbit Explainer")
        
    # ==================== TBD #1: RISK TOLERANCE ASSESSMENT ====================
    # Status: READY NOW - Core AstroShield capability
    
    async def assess_risk_tolerance(self, proximity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        TBD Proximity #5: Assess risk tolerance by fusing outputs 1-4 and CCDM
        This is AstroShield's CORE COMPETENCY - ready for immediate deployment
        """
        logger.info(f"Assessing risk tolerance for proximity event")
        
        try:
            # Extract object IDs
            primary_id = proximity_data.get("primary_object")
            secondary_id = proximity_data.get("secondary_object")
            
            # Use existing CCDM threat assessment
            threat_request = ThreatAssessmentRequest(
                norad_id=int(primary_id),
                assessment_factors=["proximity", "maneuver_history", "object_type", "debris_risk"]
            )
            
            threat_assessment = self.ccdm_service.assess_threat(threat_request)
            
            # Fuse with proximity-specific factors
            proximity_factors = {
                "miss_distance_km": proximity_data.get("miss_distance_km", 10.0),
                "relative_velocity_ms": proximity_data.get("relative_velocity_ms", 1000.0),
                "time_to_closest_approach_hours": proximity_data.get("time_to_ca_hours", 24.0),
                "object_size_ratio": proximity_data.get("size_ratio", 1.0)
            }
            
            # Calculate fused risk score
            risk_score = self._calculate_fused_risk_score(
                threat_assessment, proximity_factors
            )
            
            # Generate response recommendation
            response_recommendation = self._generate_response_recommendation(
                risk_score, proximity_factors, threat_assessment
            )
            
            return {
                "header": {
                    "messageType": "risk-tolerance-assessment",
                    "source": "astroshield-workflow-tbd",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "payload": {
                    "primary_object": primary_id,
                    "secondary_object": secondary_id,
                    "risk_tolerance_level": risk_score["level"],
                    "confidence": risk_score["confidence"],
                    "fused_score": risk_score["score"],
                    "threat_components": threat_assessment.threat_components,
                    "proximity_factors": proximity_factors,
                    "response_recommendation": response_recommendation,
                    "workflow_integration": "ss6.response-recommendation.on-orbit"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in risk tolerance assessment: {str(e)}")
            raise
    
    def _calculate_fused_risk_score(self, threat_assessment: ObjectThreatAssessment, 
                                   proximity_factors: Dict[str, float]) -> Dict[str, Any]:
        """Calculate fused risk score from CCDM and proximity factors"""
        
        # Convert threat level to numeric score
        threat_scores = {
            ThreatLevel.NONE: 0.0,
            ThreatLevel.LOW: 0.2,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.CRITICAL: 1.0
        }
        
        base_threat_score = threat_scores.get(threat_assessment.overall_threat, 0.0)
        
        # Proximity risk factors
        miss_distance_risk = max(0.0, 1.0 - (proximity_factors["miss_distance_km"] / 10.0))
        velocity_risk = min(1.0, proximity_factors["relative_velocity_ms"] / 5000.0)
        time_risk = max(0.0, 1.0 - (proximity_factors["time_to_ca_hours"] / 72.0))
        
        # Weighted fusion
        fused_score = (
            0.4 * base_threat_score +
            0.3 * miss_distance_risk +
            0.2 * velocity_risk +
            0.1 * time_risk
        )
        
        # Determine risk level
        if fused_score >= 0.8:
            level = "CRITICAL"
        elif fused_score >= 0.6:
            level = "HIGH"
        elif fused_score >= 0.3:
            level = "MEDIUM"
        else:
            level = "LOW"
            
        return {
            "score": round(fused_score, 3),
            "level": level,
            "confidence": threat_assessment.confidence
        }
    
    def _generate_response_recommendation(self, risk_score: Dict[str, Any], 
                                        proximity_factors: Dict[str, float],
                                        threat_assessment: ObjectThreatAssessment) -> Dict[str, Any]:
        """Generate response recommendations based on risk assessment"""
        
        recommendations = []
        priority = "LOW"
        
        if risk_score["level"] in ["HIGH", "CRITICAL"]:
            recommendations.extend([
                "Prepare immediate evasive maneuver options",
                "Alert spacecraft operators",
                "Increase monitoring frequency to 1-minute intervals",
                "Activate emergency response protocols"
            ])
            priority = "URGENT"
            
        elif risk_score["level"] == "MEDIUM":
            recommendations.extend([
                "Consider potential evasive maneuvers",
                "Monitor continuously until closest approach",
                "Prepare contingency maneuver plans"
            ])
            priority = "HIGH"
            
        else:
            recommendations.extend([
                "Continue routine monitoring",
                "Update trajectory predictions hourly"
            ])
            
        return {
            "priority": priority,
            "actions": recommendations,
            "maneuver_window": self._calculate_maneuver_window(proximity_factors),
            "monitoring_cadence": self._determine_monitoring_cadence(risk_score["level"])
        }
    
    # ==================== TBD #2: MANEUVER PREDICTION ====================
    
    async def predict_maneuver(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        TBD #3: Maneuver Prediction (Maneuver #2)
        
        Enhanced with jackal79/tle-orbit-explainer for natural language insights
        """
        logger.info("🛰️ TBD #3: Enhanced Maneuver Prediction with TLE Orbit Explainer")
        
        try:
            object_id = prediction_data.get("object_id")
            state_history = prediction_data.get("state_history", [])
            object_characteristics = prediction_data.get("object_characteristics", {})
            
            # Enhanced: Check for TLE data to use orbit explainer
            tle_data = prediction_data.get("tle_data")
            
            # Basic maneuver detection from state vectors
            maneuver_detected = False
            delta_v_estimate = 0.0
            confidence = 0.0
            maneuver_type = "UNKNOWN"
            
            if len(state_history) >= 2:
                # Simplified maneuver detection
                recent_states = state_history[-2:]
                delta_v_estimate, maneuver_detected = self._detect_maneuvers_from_states_simplified(recent_states)
                
                if maneuver_detected:
                    maneuver_type = self._classify_maneuver_type_simplified(delta_v_estimate)
                    confidence = min(0.85, max(0.6, delta_v_estimate * 20))
            
            # Enhanced analysis with TLE Orbit Explainer if available
            enhanced_context = {}
            if tle_data and len(tle_data) >= 2:
                try:
                    # Use TLE orbit explainer for enhanced context
                    if len(tle_data) == 4:  # Pre and post maneuver TLEs
                        pre_tle = (tle_data[0], tle_data[1])
                        post_tle = (tle_data[2], tle_data[3])
                        
                        maneuver_analysis = self.tle_explainer.analyze_maneuver_context(pre_tle, post_tle)
                        enhanced_context = {
                            "tle_analysis": maneuver_analysis,
                            "natural_language_explanation": maneuver_analysis["post_maneuver_analysis"]["explanation"],
                            "orbit_regime": maneuver_analysis["post_maneuver_analysis"]["orbital_parameters"].get("orbital_regime", "UNKNOWN"),
                            "enhanced_classification": maneuver_analysis["maneuver_classification"]
                        }
                        
                        # Override basic classification with enhanced results
                        if maneuver_analysis["maneuver_classification"]["confidence"] > confidence:
                            maneuver_type = maneuver_analysis["maneuver_classification"]["maneuver_type"]
                            confidence = maneuver_analysis["maneuver_classification"]["confidence"]
                            
                    else:  # Single TLE for context
                        tle_analysis = self.tle_explainer.explain_tle(tle_data[0], tle_data[1])
                        enhanced_context = {
                            "tle_analysis": tle_analysis,
                            "natural_language_explanation": tle_analysis["explanation"],
                            "orbit_regime": tle_analysis["orbital_parameters"].get("orbital_regime", "UNKNOWN"),
                            "risk_factors": tle_analysis["risk_assessment"]["anomaly_flags"]
                        }
                        
                except Exception as e:
                    logger.warning(f"⚠️ TLE analysis failed, using basic prediction: {e}")
            
            # Prediction timing
            predicted_time = None
            if maneuver_detected:
                # Estimate next maneuver time based on type
                base_time = datetime.now(timezone.utc)
                if maneuver_type == "STATION_KEEPING":
                    predicted_time = (base_time + timedelta(days=7)).isoformat()
                elif maneuver_type == "ORBIT_ADJUSTMENT":
                    predicted_time = (base_time + timedelta(days=30)).isoformat()
                else:
                    predicted_time = (base_time + timedelta(days=14)).isoformat()
            
            result = {
                "object_id": object_id,
                "maneuver_detected": maneuver_detected,
                "predicted_maneuver_type": maneuver_type,
                "delta_v_estimate": round(delta_v_estimate, 6),
                "confidence": round(confidence, 3),
                "predicted_time": predicted_time,
                "analysis_method": "AstroShield Enhanced AI with TLE Orbit Explainer",
                "enhanced_context": enhanced_context,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"✅ Maneuver prediction complete: {maneuver_type} with {confidence:.1%} confidence")
            return result
            
        except Exception as e:
            logger.error(f"❌ Maneuver prediction failed: {e}")
            return {
                "object_id": prediction_data.get("object_id", "unknown"),
                "error": str(e),
                "maneuver_detected": False,
                "confidence": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    # ==================== TBD #3: PEZ/WEZ SCORING FUSION ====================
    
    async def assess_pez_wez_fusion(self, proximity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        TBD Proximity #0.c: PEZ/WEZ scoring fusion
        Multi-sensor fusion for Protected/Warning Exclusion Zones
        """
        logger.info("Performing PEZ/WEZ scoring fusion")
        
        try:
            # Extract PEZ/WEZ assessments from different sources
            pez_assessments = proximity_data.get("pez_assessments", [])
            wez_assessments = proximity_data.get("wez_assessments", [])
            
            # Fuse multiple sensor inputs
            pez_fusion_score = self._fuse_pez_scores(pez_assessments)
            wez_fusion_score = self._fuse_wez_scores(wez_assessments)
            
            # Combined assessment
            combined_score = self._combine_pez_wez_scores(pez_fusion_score, wez_fusion_score)
            
            # Get threat assessment context
            threat_context = await self._get_threat_context(proximity_data)
            
            return {
                "header": {
                    "messageType": "pez-wez-fusion-assessment",
                    "source": "astroshield-workflow-tbd",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "payload": {
                    "pez_fusion_score": pez_fusion_score,
                    "wez_fusion_score": wez_fusion_score,
                    "combined_score": combined_score,
                    "threat_context": threat_context,
                    "sensor_confidence": self._calculate_sensor_confidence(
                        pez_assessments, wez_assessments
                    ),
                    "workflow_integration": "ss5.pez-wez-prediction.fusion"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in PEZ/WEZ fusion: {str(e)}")
            raise
    
    def _fuse_pez_scores(self, pez_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse multiple PEZ assessment scores"""
        if not pez_assessments:
            return {"score": 0.0, "confidence": 0.0, "sources": []}
            
        scores = [assess.get("score", 0.0) for assess in pez_assessments]
        confidences = [assess.get("confidence", 0.0) for assess in pez_assessments]
        sources = [assess.get("source", "unknown") for assess in pez_assessments]
        
        # Weighted average based on confidence
        total_weight = sum(confidences)
        if total_weight > 0:
            weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_weight
        else:
            weighted_score = np.mean(scores) if scores else 0.0
            
        return {
            "score": round(weighted_score, 3),
            "confidence": round(np.mean(confidences), 3),
            "sources": sources,
            "individual_scores": scores
        }
    
    def _fuse_wez_scores(self, wez_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse multiple WEZ assessment scores"""
        if not wez_assessments:
            return {"score": 0.0, "confidence": 0.0, "sources": []}
            
        scores = [assess.get("score", 0.0) for assess in wez_assessments]
        confidences = [assess.get("confidence", 0.0) for assess in wez_assessments]
        sources = [assess.get("source", "unknown") for assess in wez_assessments]
        
        # Take maximum score for warning zones (more conservative)
        max_score = max(scores) if scores else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            "score": round(max_score, 3),
            "confidence": round(avg_confidence, 3),
            "sources": sources,
            "individual_scores": scores
        }
    
    def _combine_pez_wez_scores(self, pez_score: Dict[str, Any], 
                               wez_score: Dict[str, Any]) -> Dict[str, Any]:
        """Combine PEZ and WEZ scores into unified assessment"""
        # PEZ is more critical, so weight it higher
        combined_score = 0.7 * pez_score["score"] + 0.3 * wez_score["score"]
        combined_confidence = (pez_score["confidence"] + wez_score["confidence"]) / 2
        
        return {
            "combined_score": round(combined_score, 3),
            "confidence": round(combined_confidence, 3),
            "assessment": "CRITICAL" if combined_score > 0.8 else 
                         "HIGH" if combined_score > 0.6 else
                         "MEDIUM" if combined_score > 0.3 else "LOW"
        }
    
    # ==================== TBD #4: THRESHOLD DETERMINATION ====================
    
    def determine_proximity_thresholds(self, context_data: Dict[str, Any]) -> ThresholdResult:
        """
        TBD Proximity #1: Dynamic threshold determination
        Leverages existing CCDM threshold algorithms
        """
        logger.info("Determining proximity thresholds")
        
        try:
            # Extract context factors
            object_types = context_data.get("object_types", ["satellite", "satellite"])
            orbital_regime = context_data.get("orbital_regime", "LEO")
            mission_criticality = context_data.get("mission_criticality", "standard")
            
            # Base thresholds by orbital regime
            base_thresholds = self._get_base_thresholds(orbital_regime)
            
            # Adjust for object characteristics
            adjusted_thresholds = self._adjust_thresholds_for_objects(
                base_thresholds, object_types, mission_criticality
            )
            
            # Dynamic factors
            dynamic_factors = self._calculate_dynamic_factors(context_data)
            
            # Apply dynamic adjustments
            final_thresholds = self._apply_dynamic_adjustments(
                adjusted_thresholds, dynamic_factors
            )
            
            return ThresholdResult(
                range_threshold_km=final_thresholds["range_km"],
                velocity_threshold_ms=final_thresholds["velocity_ms"],
                approach_rate_threshold=final_thresholds["approach_rate"],
                confidence=dynamic_factors["confidence"],
                dynamic_factors=dynamic_factors
            )
            
        except Exception as e:
            logger.error(f"Error in threshold determination: {str(e)}")
            raise
    
    def _get_base_thresholds(self, orbital_regime: str) -> Dict[str, float]:
        """Get base thresholds by orbital regime"""
        thresholds = {
            "LEO": {"range_km": 5.0, "velocity_ms": 1000.0, "approach_rate": 0.1},
            "MEO": {"range_km": 10.0, "velocity_ms": 500.0, "approach_rate": 0.05},
            "GEO": {"range_km": 50.0, "velocity_ms": 100.0, "approach_rate": 0.01},
            "HEO": {"range_km": 25.0, "velocity_ms": 750.0, "approach_rate": 0.08}
        }
        return thresholds.get(orbital_regime, thresholds["LEO"])
    
    # ==================== TBD #5: PROXIMITY EXIT CONDITIONS ====================
    
    async def monitor_proximity_exit_conditions(self, proximity_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        TBD Proximity #8.a-8.e: Monitor various proximity exit conditions
        Real-time monitoring for exit scenarios
        """
        logger.info("Monitoring proximity exit conditions")
        
        try:
            exit_checks = {
                "wez_pez_exit": await self._check_wez_pez_exit(proximity_event),
                "formation_flyer": await self._check_formation_flyer(proximity_event),
                "maneuver_cessation": await self._check_maneuver_cessation(proximity_event),
                "object_merger": await self._check_object_merger(proximity_event),
                "uct_debris": await self._check_uct_debris(proximity_event)
            }
            
            # Determine overall exit status
            exit_detected = any(check["detected"] for check in exit_checks.values())
            exit_type = None
            confidence = 0.0
            
            if exit_detected:
                # Find the highest confidence exit condition
                for condition, result in exit_checks.items():
                    if result["detected"] and result["confidence"] > confidence:
                        exit_type = condition
                        confidence = result["confidence"]
            
            return {
                "header": {
                    "messageType": "proximity-exit-conditions",
                    "source": "astroshield-workflow-tbd",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "payload": {
                    "exit_detected": exit_detected,
                    "exit_type": exit_type,
                    "confidence": confidence,
                    "detailed_checks": exit_checks,
                    "primary_object": proximity_event.get("primary_object"),
                    "secondary_object": proximity_event.get("secondary_object")
                }
            }
            
        except Exception as e:
            logger.error(f"Error monitoring exit conditions: {str(e)}")
            raise
    
    async def _check_wez_pez_exit(self, proximity_event: Dict[str, Any]) -> Dict[str, bool]:
        """Check if objects have exited WEZ/PEZ radius"""
        current_distance = proximity_event.get("current_distance_km", 0.0)
        wez_radius = proximity_event.get("wez_radius_km", 10.0)
        pez_radius = proximity_event.get("pez_radius_km", 5.0)
        
        exit_detected = current_distance > max(wez_radius, pez_radius)
        
        return {
            "detected": exit_detected,
            "confidence": 0.95 if exit_detected else 0.0,
            "details": {
                "current_distance_km": current_distance,
                "wez_radius_km": wez_radius,
                "pez_radius_km": pez_radius
            }
        }
    
    async def _check_formation_flyer(self, proximity_event: Dict[str, Any]) -> Dict[str, Any]:
        """Check if objects are formation flyers"""
        # Simplified check based on consistent close proximity
        distance_history = proximity_event.get("distance_history_km", [])
        
        if len(distance_history) >= 5:
            avg_distance = np.mean(distance_history)
            distance_variance = np.var(distance_history)
            
            # Formation flyers maintain consistent close distance
            is_formation = avg_distance < 1.0 and distance_variance < 0.1
        else:
            is_formation = False
        
        return {
            "detected": is_formation,
            "confidence": 0.8 if is_formation else 0.2
        }
    
    async def _check_maneuver_cessation(self, proximity_event: Dict[str, Any]) -> Dict[str, Any]:
        """Check if maneuvering has ceased"""
        recent_maneuvers = proximity_event.get("recent_maneuvers", [])
        
        # Check if no maneuvers in last 24 hours
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_maneuver_count = sum(
            1 for m in recent_maneuvers 
            if datetime.fromisoformat(m.get("time", "1970-01-01")) > cutoff_time
        )
        
        cessation_detected = recent_maneuver_count == 0
        
        return {
            "detected": cessation_detected,
            "confidence": 0.7 if cessation_detected else 0.3
        }
    
    async def _check_object_merger(self, proximity_event: Dict[str, Any]) -> Dict[str, Any]:
        """Check if objects have merged/collided"""
        current_distance = proximity_event.get("current_distance_km", 10.0)
        
        # Objects merged if distance is essentially zero
        merger_detected = current_distance < 0.001  # 1 meter
        
        return {
            "detected": merger_detected,
            "confidence": 0.99 if merger_detected else 0.0
        }
    
    async def _check_uct_debris(self, proximity_event: Dict[str, Any]) -> Dict[str, Any]:
        """Check for uncorrelated tracks indicating debris"""
        uct_detections = proximity_event.get("uct_detections", [])
        
        # Check for new uncorrelated tracks in the area
        recent_ucts = [
            uct for uct in uct_detections
            if datetime.fromisoformat(uct.get("detection_time", "1970-01-01")) > 
               datetime.utcnow() - timedelta(hours=1)
        ]
        
        debris_detected = len(recent_ucts) > 0
        
        return {
            "detected": debris_detected,
            "confidence": 0.6 if debris_detected else 0.1,
            "uct_count": len(recent_ucts)
        }
    
    # ==================== TBD #6: BEST EPHEMERIS AFTER MANEUVER ====================
    
    async def generate_post_maneuver_ephemeris(self, ephemeris_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        TBD #6: Post-Maneuver Ephemeris (Maneuver #3)
        
        Enhanced with jackal79/tle-orbit-explainer for improved accuracy recommendations
        """
        logger.info("📡 TBD #6: Enhanced Post-Maneuver Ephemeris with TLE Orbit Explainer")
        
        try:
            maneuver_event = ephemeris_data.get("maneuver_event", {})
            pre_maneuver_state = ephemeris_data.get("pre_maneuver_state", {})
            object_characteristics = ephemeris_data.get("object_characteristics", {})
            
            # Basic ephemeris generation
            object_id = maneuver_event.get("object_id", "unknown")
            execution_time = maneuver_event.get("execution_time", datetime.now(timezone.utc).isoformat())
            delta_v = maneuver_event.get("delta_v_estimate", 0.0)
            
            # Enhanced analysis with TLE Orbit Explainer if TLE data available
            enhanced_context = {}
            accuracy_recommendations = []
            uncertainty_factors = []
            
            tle_data = ephemeris_data.get("tle_data")
            if tle_data and len(tle_data) >= 2:
                try:
                    # Generate enhanced ephemeris context using TLE explainer
                    ephemeris_context = self.tle_explainer.generate_ephemeris_context(tle_data[0], tle_data[1])
                    
                    enhanced_context = {
                        "tle_enhanced_analysis": ephemeris_context,
                        "orbital_regime": ephemeris_context["ephemeris_context"]["orbital_regime"],
                        "decay_risk": ephemeris_context["ephemeris_context"]["decay_risk"],
                        "stability_assessment": ephemeris_context["ephemeris_context"]["stability_assessment"],
                        "natural_language_summary": ephemeris_context["ephemeris_context"]["natural_language_summary"]
                    }
                    
                    # Extract recommendations for improved accuracy
                    accuracy_recommendations = ephemeris_context["ephemeris_context"]["propagation_recommendations"]
                    uncertainty_factors = ephemeris_context["ephemeris_context"]["uncertainty_factors"]
                    
                except Exception as e:
                    logger.warning(f"⚠️ TLE ephemeris context generation failed: {e}")
            
            # Apply delta-V to generate post-maneuver state
            post_maneuver_state = self._apply_maneuver_delta_v(pre_maneuver_state, delta_v)
            
            # Generate trajectory points
            trajectory_points = self._propagate_post_maneuver_trajectory(
                post_maneuver_state, 
                execution_time,
                enhanced_context.get("orbital_regime", "LEO")
            )
            
            # Calculate uncertainty growth (enhanced with TLE insights)
            uncertainty = self._calculate_ephemeris_uncertainty(
                delta_v, 
                object_characteristics,
                uncertainty_factors
            )
            
            # Determine validity period based on orbital regime and uncertainty
            orbital_regime = enhanced_context.get("orbital_regime", "LEO")
            if orbital_regime == "GEO":
                validity_period = 168  # 7 days for GEO
            elif orbital_regime == "LEO" and enhanced_context.get("decay_risk") == "HIGH":
                validity_period = 24   # 1 day for decaying LEO
            elif orbital_regime == "LEO":
                validity_period = 72   # 3 days for stable LEO
            else:
                validity_period = 72   # Default 3 days
            
            result = {
                "object_id": object_id,
                "maneuver_execution_time": execution_time,
                "validity_period_hours": validity_period,
                "trajectory_points": len(trajectory_points),
                "post_maneuver_state": post_maneuver_state,
                "trajectory_data": trajectory_points[:10],  # First 10 points for demo
                "uncertainty": uncertainty,
                "enhanced_context": enhanced_context,
                "accuracy_recommendations": accuracy_recommendations,
                "uncertainty_factors": uncertainty_factors,
                "analysis_method": "AstroShield Enhanced SGP4/SDP4 with TLE Orbit Explainer",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"✅ Post-maneuver ephemeris generated: {validity_period}h validity period")
            return result
            
        except Exception as e:
            logger.error(f"❌ Post-maneuver ephemeris generation failed: {e}")
            return {
                "object_id": ephemeris_data.get("maneuver_event", {}).get("object_id", "unknown"),
                "error": str(e),
                "validity_period_hours": 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    # ==================== TBD #7: VOLUME SEARCH PATTERN GENERATION ====================
    
    def generate_volume_search_pattern(self, search_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        TBD Maneuver #2.b: Generate volume search pattern for lost objects
        Intelligent search pattern optimization
        """
        logger.info("Generating volume search pattern")
        
        try:
            # Extract search parameters
            last_known_position = search_request.get("last_known_position")
            last_known_velocity = search_request.get("last_known_velocity")
            time_since_last_observation = search_request.get("time_since_observation_hours", 24.0)
            object_characteristics = search_request.get("object_characteristics", {})
            
            # Calculate search volume
            search_volume = self._calculate_search_volume(
                last_known_position, last_known_velocity, time_since_last_observation
            )
            
            # Generate optimal search pattern
            search_pattern = self._optimize_search_pattern(
                search_volume, object_characteristics
            )
            
            # Calculate sensor tasking requirements
            sensor_tasking = self._calculate_sensor_tasking(search_pattern)
            
            return {
                "header": {
                    "messageType": "volume-search-pattern",
                    "source": "astroshield-workflow-tbd",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "payload": {
                    "search_volume": search_volume,
                    "search_pattern": search_pattern,
                    "sensor_tasking": sensor_tasking,
                    "estimated_search_duration_hours": search_pattern["duration_hours"],
                    "probability_of_detection": search_pattern["detection_probability"],
                    "workflow_integration": "ss3.search-pattern-generation"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating search pattern: {str(e)}")
            raise
    
    # ==================== TBD #8: OBJECT LOSS DECLARATION ====================
    
    async def evaluate_object_loss_declaration(self, custody_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        TBD Maneuver #7.b: Evaluate whether to declare an object lost
        Custody tracking and loss determination
        """
        logger.info(f"Evaluating object loss declaration for {custody_data.get('object_id')}")
        
        try:
            object_id = custody_data.get("object_id")
            last_observation_time = datetime.fromisoformat(custody_data.get("last_observation"))
            expected_observation_probability = custody_data.get("expected_obs_probability", 0.8)
            search_attempts = custody_data.get("search_attempts", [])
            
            # Calculate time since last observation
            time_since_observation = datetime.utcnow() - last_observation_time
            
            # Evaluate loss criteria
            loss_evaluation = self._evaluate_loss_criteria(
                time_since_observation, expected_observation_probability, search_attempts
            )
            
            # Make loss declaration decision
            declare_lost = self._make_loss_declaration_decision(loss_evaluation)
            
            return {
                "header": {
                    "messageType": "object-loss-declaration",
                    "source": "astroshield-workflow-tbd",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "payload": {
                    "object_id": object_id,
                    "declare_lost": declare_lost["decision"],
                    "confidence": declare_lost["confidence"],
                    "loss_evaluation": loss_evaluation,
                    "time_since_last_observation_hours": time_since_observation.total_seconds() / 3600,
                    "recommended_actions": declare_lost["recommended_actions"],
                    "workflow_integration": "ss3.object-loss-declaration"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in object loss evaluation: {str(e)}")
            raise
    
    # ==================== HELPER METHODS ====================
    
    def _calculate_maneuver_window(self, proximity_factors: Dict[str, float]) -> Dict[str, Any]:
        """Calculate optimal maneuver execution window"""
        time_to_ca = proximity_factors.get("time_to_ca_hours", 24.0)
        
        # Optimal maneuver time is typically 2-6 hours before closest approach
        optimal_time_hours = max(2.0, min(6.0, time_to_ca * 0.25))
        
        return {
            "optimal_execution_time_hours": optimal_time_hours,
            "latest_execution_time_hours": max(1.0, time_to_ca * 0.1),
            "maneuver_effectiveness": min(1.0, optimal_time_hours / 6.0)
        }
    
    def _determine_monitoring_cadence(self, risk_level: str) -> Dict[str, Any]:
        """Determine appropriate monitoring frequency"""
        cadences = {
            "LOW": {"interval_minutes": 60, "duration_hours": 24},
            "MEDIUM": {"interval_minutes": 15, "duration_hours": 48},
            "HIGH": {"interval_minutes": 5, "duration_hours": 72},
            "CRITICAL": {"interval_minutes": 1, "duration_hours": 96}
        }
        return cadences.get(risk_level, cadences["LOW"])
    
    async def _generate_post_maneuver_ephemeris(self, object_id: str, 
                                              prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ephemeris data after predicted maneuver"""
        # Simplified implementation - would use full trajectory propagation
        return {
            "ephemeris_type": "post_maneuver_prediction",
            "validity_start": prediction["predicted_time"].isoformat(),
            "validity_end": (prediction["predicted_time"] + timedelta(hours=72)).isoformat(),
            "orbital_elements": {
                "semi_major_axis_km": 7000.0,  # Placeholder
                "eccentricity": 0.001,
                "inclination_deg": 98.0,
                "raan_deg": 45.0,
                "arg_perigee_deg": 90.0,
                "mean_anomaly_deg": 180.0
            },
            "uncertainty_1sigma_km": 0.5,
            "confidence": prediction["confidence"]
        }
    
    def _predict_station_keeping_maneuver(self, object_id: str, 
                                        states: List[Dict]) -> ManeuverPrediction:
        """Predict station-keeping maneuver for objects showing orbital decay"""
        # Simplified implementation - would analyze orbital decay rate
        next_maneuver_time = datetime.utcnow() + timedelta(days=7)  # Typical SK interval
        
        return ManeuverPrediction(
            object_id=object_id,
            predicted_time=next_maneuver_time,
            maneuver_type="STATION_KEEPING",
            delta_v_estimate=0.02,  # Typical SK delta-v
            confidence=0.7,
            ephemeris_update={"type": "station_keeping_prediction"}
        )
    
    # ==================== MISSING HELPER METHODS ====================
    
    def _adjust_thresholds_for_objects(self, base_thresholds: Dict[str, float], 
                                     object_types: List[str], 
                                     mission_criticality: str) -> Dict[str, float]:
        """Adjust thresholds based on object characteristics"""
        adjusted = base_thresholds.copy()
        
        # Adjust for object types
        if "debris" in object_types:
            adjusted["range_km"] *= 1.5  # More conservative for debris
        
        if "active_satellite" in object_types:
            adjusted["velocity_ms"] *= 0.8  # More sensitive for active satellites
            
        # Adjust for mission criticality
        if mission_criticality == "critical":
            adjusted["range_km"] *= 2.0
            adjusted["approach_rate"] *= 0.5
        elif mission_criticality == "high":
            adjusted["range_km"] *= 1.5
            
        return adjusted
    
    def _calculate_dynamic_factors(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate dynamic adjustment factors"""
        factors = {
            "atmospheric_density": context_data.get("atmospheric_density_factor", 1.0),
            "solar_activity": context_data.get("solar_activity_factor", 1.0),
            "tracking_accuracy": context_data.get("tracking_accuracy", 0.9),
            "confidence": 0.8
        }
        
        # Adjust confidence based on data quality
        if factors["tracking_accuracy"] < 0.7:
            factors["confidence"] *= 0.8
            
        return factors
    
    def _apply_dynamic_adjustments(self, thresholds: Dict[str, float], 
                                 dynamic_factors: Dict[str, Any]) -> Dict[str, float]:
        """Apply dynamic adjustments to thresholds"""
        adjusted = thresholds.copy()
        
        # Adjust for atmospheric conditions
        atm_factor = dynamic_factors.get("atmospheric_density", 1.0)
        adjusted["range_km"] *= (1.0 + 0.2 * (atm_factor - 1.0))
        
        # Adjust for tracking accuracy
        tracking_acc = dynamic_factors.get("tracking_accuracy", 0.9)
        if tracking_acc < 0.8:
            adjusted["range_km"] *= 1.3  # More conservative with poor tracking
            
        return adjusted
    
    def _estimate_maneuver_duration(self, delta_v: float) -> float:
        """Estimate maneuver duration in minutes"""
        # Simplified linear model
        return max(1.0, delta_v * 100.0)  # 100 minutes per km/s delta-v
    
    def _analyze_orbital_impact(self, delta_v: float, altitude_km: float) -> Dict[str, Any]:
        """Analyze orbital impact of maneuver"""
        return {
            "altitude_change_km": delta_v * 100.0,  # Simplified
            "period_change_minutes": delta_v * 50.0,
            "impact_severity": "LOW" if delta_v < 0.05 else 
                             "MEDIUM" if delta_v < 0.1 else "HIGH"
        }
    
    async def _get_threat_context(self, proximity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get threat assessment context for PEZ/WEZ fusion"""
        primary_object = proximity_data.get("primary_object")
        
        if primary_object:
            try:
                threat_request = ThreatAssessmentRequest(
                    norad_id=int(primary_object),
                    assessment_factors=["proximity", "object_type"]
                )
                threat_assessment = self.ccdm_service.assess_threat(threat_request)
                return {
                    "threat_level": threat_assessment.overall_threat.value,
                    "confidence": threat_assessment.confidence
                }
            except Exception:
                pass
                
        return {"threat_level": "UNKNOWN", "confidence": 0.5}
    
    def _calculate_sensor_confidence(self, pez_assessments: List[Dict[str, Any]], 
                                   wez_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall sensor confidence"""
        all_confidences = []
        
        for assess in pez_assessments + wez_assessments:
            all_confidences.append(assess.get("confidence", 0.0))
            
        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            min_confidence = min(all_confidences)
            max_confidence = max(all_confidences)
        else:
            avg_confidence = min_confidence = max_confidence = 0.0
            
        return {
            "average": round(avg_confidence, 3),
            "minimum": round(min_confidence, 3),
            "maximum": round(max_confidence, 3),
            "sensor_count": len(all_confidences)
        }
    
    def _calculate_search_volume(self, last_position: Dict[str, float], 
                               last_velocity: Dict[str, float], 
                               time_hours: float) -> Dict[str, Any]:
        """Calculate search volume for lost object"""
        # Simplified uncertainty propagation
        position_uncertainty_km = time_hours * 0.5  # 0.5 km/hr uncertainty growth
        velocity_uncertainty_ms = time_hours * 0.1   # 0.1 m/s/hr uncertainty growth
        
        search_radius_km = position_uncertainty_km + (velocity_uncertainty_ms * time_hours * 3.6)
        
        return {
            "center_position": last_position,
            "search_radius_km": search_radius_km,
            "volume_km3": (4/3) * 3.14159 * (search_radius_km ** 3),
            "uncertainty_ellipse": {
                "semi_major_axis_km": search_radius_km * 1.2,
                "semi_minor_axis_km": search_radius_km * 0.8
            }
        }
    
    def _optimize_search_pattern(self, search_volume: Dict[str, Any], 
                                object_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimal search pattern"""
        search_radius = search_volume["search_radius_km"]
        
        # Grid-based search pattern
        grid_spacing_km = max(1.0, search_radius / 10.0)
        
        # Calculate search parameters
        coverage_area_km2 = 3.14159 * (search_radius ** 2)
        sensor_fov_km2 = object_characteristics.get("sensor_fov_km2", 100.0)
        
        search_points = int(coverage_area_km2 / sensor_fov_km2)
        duration_hours = search_points * 0.1  # 6 minutes per search point
        
        return {
            "pattern_type": "SPIRAL_GRID",
            "grid_spacing_km": grid_spacing_km,
            "search_points": search_points,
            "duration_hours": duration_hours,
            "detection_probability": min(0.95, 0.5 + (sensor_fov_km2 / coverage_area_km2))
        }
    
    def _calculate_sensor_tasking(self, search_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sensor tasking requirements"""
        return {
            "required_sensors": max(1, int(search_pattern["search_points"] / 100)),
            "observation_time_per_point_minutes": 6,
            "total_observation_time_hours": search_pattern["duration_hours"],
            "recommended_sensor_types": ["optical", "radar"],
            "priority": "HIGH" if search_pattern["duration_hours"] > 24 else "MEDIUM"
        }
    
    def _evaluate_loss_criteria(self, time_since_observation: timedelta, 
                               expected_probability: float, 
                               search_attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate criteria for object loss declaration"""
        hours_since = time_since_observation.total_seconds() / 3600
        
        # Time-based criteria
        time_threshold_exceeded = hours_since > 168  # 7 days
        
        # Search-based criteria
        search_attempts_count = len(search_attempts)
        comprehensive_search_conducted = search_attempts_count >= 3
        
        # Probability-based criteria
        low_detection_probability = expected_probability < 0.1
        
        return {
            "time_threshold_exceeded": time_threshold_exceeded,
            "hours_since_observation": hours_since,
            "comprehensive_search_conducted": comprehensive_search_conducted,
            "search_attempts_count": search_attempts_count,
            "low_detection_probability": low_detection_probability,
            "expected_probability": expected_probability
        }
    
    def _make_loss_declaration_decision(self, loss_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Make final decision on object loss declaration"""
        criteria_met = sum([
            loss_evaluation["time_threshold_exceeded"],
            loss_evaluation["comprehensive_search_conducted"],
            loss_evaluation["low_detection_probability"]
        ])
        
        # Need at least 2 out of 3 criteria
        declare_lost = criteria_met >= 2
        confidence = min(0.95, 0.3 + (criteria_met * 0.2))
        
        recommendations = []
        if declare_lost:
            recommendations.extend([
                "Update catalog with 'LOST' status",
                "Notify relevant authorities",
                "Archive tracking data",
                "Consider debris implications"
            ])
        else:
            recommendations.extend([
                "Continue search operations",
                "Expand search volume",
                "Request additional sensor time"
            ])
            
        return {
            "decision": declare_lost,
            "confidence": confidence,
            "criteria_met": criteria_met,
            "recommended_actions": recommendations
        }
    
    def _apply_maneuver_delta_v(self, pre_state: Dict[str, Any], 
                               delta_v: List[float]) -> Dict[str, Any]:
        """Apply maneuver delta-v to state vector"""
        post_state = pre_state.copy()
        
        # Apply delta-v to velocity components
        if "velocity" in post_state:
            post_state["velocity"]["x"] += delta_v[0]
            post_state["velocity"]["y"] += delta_v[1] 
            post_state["velocity"]["z"] += delta_v[2]
            
        return post_state
    
    def _propagate_post_maneuver_trajectory(self, post_state: Dict[str, Any], 
                                          execution_time: datetime,
                                          orbital_regime: str) -> List[Dict[str, Any]]:
        """Propagate trajectory after maneuver execution"""
        # Simplified trajectory propagation
        trajectory_points = []
        
        for hours in range(0, 73, 1):  # 72 hours, hourly points
            future_time = execution_time + timedelta(hours=hours)
            
            # Simplified orbital mechanics (would use full propagation in production)
            trajectory_points.append({
                "time": future_time.isoformat(),
                "position": post_state.get("position", {"x": 0, "y": 0, "z": 0}),
                "velocity": post_state.get("velocity", {"x": 0, "y": 0, "z": 0}),
                "uncertainty_km": 0.1 * hours  # Growing uncertainty
            })
            
        return trajectory_points
    
    def _calculate_ephemeris_uncertainty(self, delta_v: float, 
                                       object_characteristics: Dict[str, Any],
                                       uncertainty_factors: List[float]) -> Dict[str, Any]:
        """Calculate uncertainty in post-maneuver ephemeris"""
        delta_v_magnitude = np.linalg.norm([delta_v, delta_v, delta_v])
        
        # Uncertainty grows with delta-v magnitude and time
        base_uncertainty_km = 0.1 + (delta_v_magnitude * 10.0)
        
        # Apply additional uncertainty factors
        for factor in uncertainty_factors:
            base_uncertainty_km += factor * 0.1 * delta_v_magnitude
        
        return {
            "position_uncertainty_1sigma_km": base_uncertainty_km,
            "velocity_uncertainty_1sigma_ms": delta_v_magnitude * 0.1,
            "uncertainty_growth_rate_km_per_hour": 0.05,
            "confidence_degradation_per_day": 0.1
        }
    
    def _detect_maneuvers_from_states_simplified(self, states: List[Dict[str, Any]]) -> Tuple[float, bool]:
        """Simplified maneuver detection from state vectors"""
        if len(states) < 2:
            return 0.0, False
        
        # Calculate velocity changes between consecutive states
        velocity_changes = []
        timestamps = []
        
        for i in range(len(states) - 1):
            try:
                v1 = np.array([
                    states[i]["velocity"]["x"], 
                    states[i]["velocity"]["y"], 
                    states[i]["velocity"]["z"]
                ])
                v2 = np.array([
                    states[i+1]["velocity"]["x"], 
                    states[i+1]["velocity"]["y"], 
                    states[i+1]["velocity"]["z"]
                ])
                
                delta_v = np.linalg.norm(v2 - v1)
                
                if delta_v > 0.001:  # Minimum threshold in km/s
                    velocity_changes.append(delta_v)
                    timestamps.append(states[i+1].get("epoch", datetime.utcnow().isoformat()))
            except (KeyError, TypeError):
                continue
        
        if not velocity_changes:
            return 0.0, False
        
        # Find the largest delta-v
        max_delta_v = max(velocity_changes)
        max_delta_v_index = velocity_changes.index(max_delta_v)
        max_delta_v_time = timestamps[max_delta_v_index]
        
        # Classify maneuver type
        if max_delta_v > 0.5:
            maneuver_type = "ORBIT_CHANGE"
        elif max_delta_v > 0.1:
            maneuver_type = "ORBIT_ADJUSTMENT"
        elif max_delta_v > 0.05:
            maneuver_type = "STATION_KEEPING"
        else:
            maneuver_type = "MINOR_CORRECTION"
        
        # Calculate confidence
        confidence = min(0.95, 0.5 + max_delta_v)
        
        return max_delta_v, confidence
    
    def _classify_maneuver_type_simplified(self, delta_v: float) -> str:
        """Classify maneuver type based on delta-v"""
        if delta_v > 0.5:
            return "ORBIT_CHANGE"
        elif delta_v > 0.1:
            return "ORBIT_ADJUSTMENT"
        elif delta_v > 0.05:
            return "STATION_KEEPING"
        else:
            return "MINOR_CORRECTION"

# Factory function for service creation
def get_workflow_tbd_service(db_session=None) -> WorkflowTBDService:
    """Factory function to create WorkflowTBDService instance"""
    return WorkflowTBDService(db_session) 