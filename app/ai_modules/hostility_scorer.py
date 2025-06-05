"""
Hostility Scorer Module for AstroShield

This module implements a multi-factor hostility scoring system that assesses threat levels
based on actor identity, maneuver patterns, proximity operations, and operational context.
It provides structured threat assessments with explainable scoring for decision support.

Key capabilities:
- Multi-factor threat scoring algorithm
- Actor classification and history analysis
- Pattern-of-life baseline establishment
- Proximity threat assessment
- Explainable threat reasoning
- Integration with intelligence feeds
- Real-time threat level updates
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import hashlib
from dataclasses import dataclass

from .models import (
    HostilityAssessment, IntentClassificationResult, ThreatLevel, 
    ActorType, ManeuverEvent, ProximityEvent, ModelConfig, AIAnalysisMessage
)
from app.common.logging import logger


@dataclass
class ThreatScoringFactors:
    """Configuration for threat scoring factors and weights."""
    actor_weight: float = 0.25        # Actor identity and history
    intent_weight: float = 0.30       # Classified intent from ML
    proximity_weight: float = 0.20    # Proximity operations
    pattern_weight: float = 0.15      # Pattern-of-life deviation
    capability_weight: float = 0.10   # Known capabilities


class ActorDatabase:
    """Database of known actors and their threat profiles."""
    
    def __init__(self):
        """Initialize with default actor classifications."""
        self.actors = {
            # US and Allied
            "USSF": {"type": ActorType.US, "base_threat": 0.0, "reliability": 0.95},
            "NASA": {"type": ActorType.US, "base_threat": 0.0, "reliability": 0.95},
            "NOAA": {"type": ActorType.US, "base_threat": 0.0, "reliability": 0.90},
            "ESA": {"type": ActorType.ALLY, "base_threat": 0.1, "reliability": 0.90},
            "JAXA": {"type": ActorType.ALLY, "base_threat": 0.1, "reliability": 0.85},
            "CSA": {"type": ActorType.ALLY, "base_threat": 0.1, "reliability": 0.85},
            
            # Commercial
            "SPACEX": {"type": ActorType.COMMERCIAL, "base_threat": 0.1, "reliability": 0.80},
            "PLANET": {"type": ActorType.COMMERCIAL, "base_threat": 0.1, "reliability": 0.75},
            "SKYBOX": {"type": ActorType.COMMERCIAL, "base_threat": 0.1, "reliability": 0.75},
            
            # Adversary nations (example patterns)
            "CNSA": {"type": ActorType.ADVERSARY, "base_threat": 0.6, "reliability": 0.60},
            "ROSCOSMOS": {"type": ActorType.ADVERSARY, "base_threat": 0.5, "reliability": 0.65},
            
            # Unknown
            "UNKNOWN": {"type": ActorType.UNKNOWN, "base_threat": 0.3, "reliability": 0.30}
        }
        
        # Historical threat events by actor
        self.threat_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Pattern-of-life baselines
        self.pol_baselines: Dict[str, Dict[str, Any]] = {}
    
    def get_actor_profile(self, actor_id: str) -> Dict[str, Any]:
        """Get actor profile or return unknown profile."""
        # Try exact match first
        if actor_id in self.actors:
            return self.actors[actor_id]
        
        # Try partial matches for organizations
        for known_actor, profile in self.actors.items():
            if known_actor.lower() in actor_id.lower():
                return profile
        
        # Default to unknown
        return self.actors["UNKNOWN"]
    
    def update_threat_history(self, actor_id: str, threat_event: Dict[str, Any]):
        """Update threat history for an actor."""
        if actor_id not in self.threat_history:
            self.threat_history[actor_id] = []
        
        self.threat_history[actor_id].append(threat_event)
        
        # Keep only last 100 events per actor
        if len(self.threat_history[actor_id]) > 100:
            self.threat_history[actor_id] = self.threat_history[actor_id][-100:]
    
    def get_historical_threat_score(self, actor_id: str, days: int = 30) -> float:
        """Calculate historical threat score based on past events."""
        if actor_id not in self.threat_history:
            return 0.0
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_events = [
            event for event in self.threat_history[actor_id]
            if datetime.fromisoformat(event['timestamp']) > cutoff_date
        ]
        
        if not recent_events:
            return 0.0
        
        # Weight recent events more heavily
        total_score = 0.0
        total_weight = 0.0
        
        for event in recent_events:
            age_days = (datetime.utcnow() - datetime.fromisoformat(event['timestamp'])).days
            weight = np.exp(-age_days / 10.0)  # Exponential decay
            total_score += event.get('threat_score', 0.0) * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


class PatternAnalyzer:
    """Analyzes behavioral patterns and deviations from normal operations."""
    
    def __init__(self):
        self.normal_patterns: Dict[str, Dict[str, Any]] = {}
        self.recent_activities: Dict[str, List[Dict[str, Any]]] = {}
    
    def establish_baseline(self, actor_id: str, activities: List[Dict[str, Any]]):
        """Establish pattern-of-life baseline for an actor."""
        if len(activities) < 10:  # Need sufficient data
            return
        
        # Analyze temporal patterns
        time_intervals = []
        for i in range(1, len(activities)):
            interval = (datetime.fromisoformat(activities[i]['timestamp']) - 
                       datetime.fromisoformat(activities[i-1]['timestamp'])).total_seconds() / 3600
            time_intervals.append(interval)
        
        # Analyze maneuver characteristics
        delta_vs = [activity.get('delta_v', 0.0) for activity in activities]
        
        baseline = {
            'mean_interval_hours': np.mean(time_intervals) if time_intervals else 0.0,
            'std_interval_hours': np.std(time_intervals) if time_intervals else 0.0,
            'mean_delta_v': np.mean(delta_vs),
            'std_delta_v': np.std(delta_vs),
            'activity_frequency': len(activities) / 30.0,  # per day average
            'last_updated': datetime.utcnow().isoformat()
        }
        
        self.normal_patterns[actor_id] = baseline
    
    def analyze_deviation(self, actor_id: str, recent_activity: Dict[str, Any]) -> float:
        """Analyze how much recent activity deviates from normal pattern."""
        if actor_id not in self.normal_patterns:
            return 0.0  # No baseline to compare against
        
        baseline = self.normal_patterns[actor_id]
        deviation_score = 0.0
        
        # Check delta-v deviation
        if 'delta_v' in recent_activity:
            delta_v = recent_activity['delta_v']
            expected_dv = baseline['mean_delta_v']
            dv_std = baseline['std_delta_v']
            
            if dv_std > 0:
                dv_zscore = abs(delta_v - expected_dv) / dv_std
                deviation_score += min(dv_zscore / 3.0, 1.0) * 0.4  # Cap at 1.0
        
        # Check timing deviation
        if actor_id in self.recent_activities and self.recent_activities[actor_id]:
            last_activity = self.recent_activities[actor_id][-1]
            time_gap = (datetime.fromisoformat(recent_activity['timestamp']) - 
                       datetime.fromisoformat(last_activity['timestamp'])).total_seconds() / 3600
            
            expected_gap = baseline['mean_interval_hours']
            gap_std = baseline['std_interval_hours']
            
            if gap_std > 0:
                gap_zscore = abs(time_gap - expected_gap) / gap_std
                deviation_score += min(gap_zscore / 3.0, 1.0) * 0.3
        
        # Check activity frequency anomaly
        if actor_id in self.recent_activities:
            recent_count = len([
                a for a in self.recent_activities[actor_id]
                if (datetime.utcnow() - datetime.fromisoformat(a['timestamp'])).days <= 7
            ])
            expected_weekly = baseline['activity_frequency'] * 7
            if expected_weekly > 0:
                freq_ratio = recent_count / expected_weekly
                if freq_ratio > 2.0 or freq_ratio < 0.5:  # 2x increase or 50% decrease
                    deviation_score += 0.3
        
        return min(deviation_score, 1.0)
    
    def update_activity(self, actor_id: str, activity: Dict[str, Any]):
        """Update recent activity tracking."""
        if actor_id not in self.recent_activities:
            self.recent_activities[actor_id] = []
        
        self.recent_activities[actor_id].append(activity)
        
        # Keep only last 30 days of activities
        cutoff = datetime.utcnow() - timedelta(days=30)
        self.recent_activities[actor_id] = [
            a for a in self.recent_activities[actor_id]
            if datetime.fromisoformat(a['timestamp']) > cutoff
        ]


class ProximityThreatAnalyzer:
    """Analyzes proximity operations for threat indicators."""
    
    def __init__(self):
        self.high_value_assets = {
            # Example high-value targets (NORAD IDs)
            "25544": {"name": "ISS", "criticality": 1.0},
            "43013": {"name": "GPS III SV01", "criticality": 0.9},
            "40128": {"name": "WGS-6", "criticality": 0.8},
        }
        
        self.proximity_thresholds = {
            "critical": 1000,    # meters
            "high": 5000,        # meters
            "medium": 20000,     # meters
            "low": 100000        # meters
        }
    
    def analyze_proximity_threat(self, proximity_event: ProximityEvent, 
                               intent_result: Optional[IntentClassificationResult] = None) -> float:
        """Analyze threat level of a proximity operation."""
        threat_score = 0.0
        
        # Check if target is high-value asset
        target_criticality = 0.0
        if proximity_event.secondary_norad_id in self.high_value_assets:
            target_criticality = self.high_value_assets[proximity_event.secondary_norad_id]["criticality"]
            threat_score += target_criticality * 0.4
        
        # Analyze proximity distance
        min_distance = proximity_event.minimum_distance
        if min_distance < self.proximity_thresholds["critical"]:
            threat_score += 0.8
        elif min_distance < self.proximity_thresholds["high"]:
            threat_score += 0.6
        elif min_distance < self.proximity_thresholds["medium"]:
            threat_score += 0.3
        elif min_distance < self.proximity_thresholds["low"]:
            threat_score += 0.1
        
        # Consider relative velocity
        rel_velocity = proximity_event.relative_velocity
        if rel_velocity > 1000:  # High speed approach
            threat_score += 0.3
        elif rel_velocity < 100:  # Slow, controlled approach
            threat_score += 0.2
        
        # Consider duration
        if proximity_event.duration_minutes > 60:  # Extended proximity
            threat_score += 0.2
        
        # Factor in intent classification if available
        if intent_result:
            if intent_result.intent_class.value in ["inspection", "shadowing"]:
                threat_score += 0.3
            elif intent_result.intent_class.value in ["collision_course"]:
                threat_score += 0.9
        
        return min(threat_score, 1.0)


class HostilityScorer:
    """Main hostility scoring service."""
    
    def __init__(self, config: Optional[ModelConfig] = None, kafka_adapter=None):
        """Initialize the hostility scorer."""
        self.config = config or ModelConfig(
            model_name="hostility_scorer",
            model_version="1.0.0",
            confidence_threshold=0.6
        )
        self.kafka_adapter = kafka_adapter
        
        # Initialize components
        self.scoring_factors = ThreatScoringFactors()
        self.actor_database = ActorDatabase()
        self.pattern_analyzer = PatternAnalyzer()
        self.proximity_analyzer = ProximityThreatAnalyzer()
        
        # Assessment cache and history
        self.assessment_cache: Dict[str, HostilityAssessment] = {}
        self.scoring_history: Dict[str, List[float]] = {}
        
        logger.info(f"HostilityScorer initialized: {self.config.model_name}")
    
    async def assess_hostility(self, 
                             event: ManeuverEvent,
                             intent_result: Optional[IntentClassificationResult] = None,
                             proximity_event: Optional[ProximityEvent] = None,
                             actor_id: Optional[str] = None) -> HostilityAssessment:
        """Perform comprehensive hostility assessment."""
        
        try:
            target_norad_id = event.primary_norad_id
            actor_id = actor_id or self._infer_actor_from_norad(target_norad_id)
            
            # Get actor profile
            actor_profile = self.actor_database.get_actor_profile(actor_id)
            actor_type = actor_profile["type"]
            
            # Calculate individual factor scores
            factor_scores = await self._calculate_factor_scores(
                event, intent_result, proximity_event, actor_id, actor_profile
            )
            
            # Calculate weighted hostility score
            hostility_score = self._calculate_weighted_score(factor_scores)
            
            # Determine threat level
            threat_level = self._determine_threat_level(hostility_score, actor_type)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                hostility_score, threat_level, factor_scores, actor_type
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(factor_scores, actor_profile)
            
            # Create assessment
            assessment = HostilityAssessment(
                event_id=event.event_id,
                timestamp=datetime.utcnow(),
                target_norad_id=target_norad_id,
                actor_type=actor_type,
                threat_level=threat_level,
                hostility_score=hostility_score,
                contributing_factors=factor_scores,
                pattern_analysis={
                    "baseline_deviation": factor_scores.get("pattern_deviation", 0.0),
                    "activity_frequency": self.pattern_analyzer.recent_activities.get(actor_id, []),
                    "historical_threat": factor_scores.get("historical_threat", 0.0)
                },
                recommendations=recommendations,
                confidence=confidence,
                model_version=self.config.model_version
            )
            
            # Update history and patterns
            await self._update_records(actor_id, assessment, event)
            
            # Cache assessment
            cache_key = self._generate_cache_key(event)
            self.assessment_cache[cache_key] = assessment
            
            # Publish to Kafka if adapter available
            if self.kafka_adapter:
                await self._publish_assessment(assessment)
            
            logger.info(f"Hostility assessment completed: {threat_level} (score: {hostility_score:.3f})")
            return assessment
            
        except Exception as e:
            logger.error(f"Hostility assessment failed for event {event.event_id}: {str(e)}")
            raise
    
    async def _calculate_factor_scores(self, 
                                     event: ManeuverEvent,
                                     intent_result: Optional[IntentClassificationResult],
                                     proximity_event: Optional[ProximityEvent],
                                     actor_id: str,
                                     actor_profile: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual factor scores."""
        
        factor_scores = {}
        
        # Actor factor score
        base_threat = actor_profile["base_threat"]
        historical_threat = self.actor_database.get_historical_threat_score(actor_id)
        factor_scores["actor_base"] = base_threat
        factor_scores["historical_threat"] = historical_threat
        factor_scores["actor_total"] = (base_threat + historical_threat) / 2.0
        
        # Intent factor score
        if intent_result:
            intent_threat_mapping = {
                "inspection": 0.3,
                "shadowing": 0.7,
                "collision_course": 0.95,
                "evasion": 0.4,
                "imaging_pass": 0.2,
                "station_keeping": 0.1,
                "debris_avoidance": 0.05,
                "rendezvous": 0.5,
                "routine_maintenance": 0.05,
                "unknown": 0.2
            }
            # Handle both enum and string values
            intent_value = intent_result.intent_class.value if hasattr(intent_result.intent_class, 'value') else str(intent_result.intent_class)
            intent_score = intent_threat_mapping.get(intent_value, 0.2)
            # Weight by confidence
            factor_scores["intent"] = intent_score * intent_result.confidence_score
        else:
            factor_scores["intent"] = 0.2  # Default uncertainty
        
        # Proximity factor score
        if proximity_event:
            factor_scores["proximity"] = self.proximity_analyzer.analyze_proximity_threat(
                proximity_event, intent_result
            )
        else:
            factor_scores["proximity"] = 0.0
        
        # Pattern deviation score
        # Handle both enum and string values for maneuver_type
        maneuver_type_value = event.maneuver_type.value if hasattr(event.maneuver_type, 'value') else str(event.maneuver_type)
        activity_data = {
            "timestamp": event.timestamp.isoformat(),
            "delta_v": event.delta_v,
            "maneuver_type": maneuver_type_value
        }
        factor_scores["pattern_deviation"] = self.pattern_analyzer.analyze_deviation(
            actor_id, activity_data
        )
        
        # Capability factor (placeholder - would integrate with intel feeds)
        factor_scores["capability"] = self._assess_capability_threat(actor_id, event)
        
        return factor_scores
    
    def _calculate_weighted_score(self, factor_scores: Dict[str, float]) -> float:
        """Calculate weighted hostility score."""
        weighted_score = (
            factor_scores.get("actor_total", 0.0) * self.scoring_factors.actor_weight +
            factor_scores.get("intent", 0.0) * self.scoring_factors.intent_weight +
            factor_scores.get("proximity", 0.0) * self.scoring_factors.proximity_weight +
            factor_scores.get("pattern_deviation", 0.0) * self.scoring_factors.pattern_weight +
            factor_scores.get("capability", 0.0) * self.scoring_factors.capability_weight
        )
        
        return min(weighted_score, 1.0)
    
    def _determine_threat_level(self, hostility_score: float, actor_type: ActorType) -> ThreatLevel:
        """Determine threat level based on score and actor type."""
        # Adjust thresholds based on actor type
        if actor_type == ActorType.US:
            # Very high threshold for US assets
            if hostility_score > 0.9:
                return ThreatLevel.SUSPECT
            else:
                return ThreatLevel.BENIGN
        elif actor_type == ActorType.ALLY:
            # High threshold for allies
            if hostility_score > 0.8:
                return ThreatLevel.SUSPECT
            else:
                return ThreatLevel.BENIGN
        elif actor_type == ActorType.ADVERSARY:
            # Lower thresholds for adversaries
            if hostility_score > 0.8:
                return ThreatLevel.CRITICAL
            elif hostility_score > 0.6:
                return ThreatLevel.HOSTILE
            elif hostility_score > 0.3:
                return ThreatLevel.SUSPECT
            else:
                return ThreatLevel.BENIGN
        else:  # COMMERCIAL or UNKNOWN
            # Standard thresholds
            if hostility_score > 0.85:
                return ThreatLevel.CRITICAL
            elif hostility_score > 0.7:
                return ThreatLevel.HOSTILE
            elif hostility_score > 0.4:
                return ThreatLevel.SUSPECT
            else:
                return ThreatLevel.BENIGN
    
    def _generate_recommendations(self, hostility_score: float, threat_level: ThreatLevel,
                                factor_scores: Dict[str, float], actor_type: ActorType) -> List[str]:
        """Generate actionable recommendations based on assessment."""
        recommendations = []
        
        if threat_level in [ThreatLevel.HOSTILE, ThreatLevel.CRITICAL]:
            recommendations.append("Immediate enhanced tracking and monitoring required")
            recommendations.append("Alert operational commanders and intelligence analysts")
            
            if factor_scores.get("proximity", 0.0) > 0.5:
                recommendations.append("Implement collision avoidance maneuvers if necessary")
            
            if factor_scores.get("intent", 0.0) > 0.7:
                recommendations.append("Analyze intent classification for specific threat patterns")
        
        elif threat_level == ThreatLevel.SUSPECT:
            recommendations.append("Increase monitoring frequency and data collection")
            recommendations.append("Review pattern-of-life analysis for anomalies")
            
            if factor_scores.get("pattern_deviation", 0.0) > 0.5:
                recommendations.append("Investigate cause of behavioral pattern deviation")
        
        # Actor-specific recommendations
        if actor_type == ActorType.UNKNOWN:
            recommendations.append("Prioritize actor identification and intelligence collection")
        
        if actor_type == ActorType.ADVERSARY and hostility_score > 0.3:
            recommendations.append("Cross-reference with known adversary capabilities and patterns")
        
        # Capability-based recommendations
        if factor_scores.get("capability", 0.0) > 0.5:
            recommendations.append("Assess potential for hostile capability employment")
        
        return recommendations
    
    def _calculate_confidence(self, factor_scores: Dict[str, float], 
                            actor_profile: Dict[str, Any]) -> float:
        """Calculate confidence in the assessment."""
        confidence_factors = []
        
        # Actor reliability
        confidence_factors.append(actor_profile.get("reliability", 0.5))
        
        # Data completeness
        available_factors = sum(1 for score in factor_scores.values() if score > 0)
        total_factors = 5  # actor, intent, proximity, pattern, capability
        completeness = available_factors / total_factors
        confidence_factors.append(completeness)
        
        # Recent data availability
        confidence_factors.append(0.8)  # Placeholder
        
        return np.mean(confidence_factors)
    
    def _assess_capability_threat(self, actor_id: str, event: ManeuverEvent) -> float:
        """Assess threat based on demonstrated capabilities."""
        # Placeholder implementation - would integrate with intelligence feeds
        
        # High delta-v suggests advanced propulsion
        if event.delta_v > 50:
            return 0.7
        elif event.delta_v > 10:
            return 0.4
        else:
            return 0.1
    
    def _infer_actor_from_norad(self, norad_id: str) -> str:
        """Infer actor from NORAD catalog ID."""
        # Placeholder implementation - would use actual database
        # In practice, this would query satellite database for owner/operator
        return "UNKNOWN"
    
    def _generate_cache_key(self, event: ManeuverEvent) -> str:
        """Generate cache key for assessment."""
        key_data = f"{event.event_id}_{event.timestamp}_{event.primary_norad_id}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _update_records(self, actor_id: str, assessment: HostilityAssessment, 
                            event: ManeuverEvent):
        """Update historical records and patterns."""
        
        # Update threat history
        threat_event = {
            "timestamp": assessment.timestamp.isoformat(),
            "threat_score": assessment.hostility_score,
            "threat_level": assessment.threat_level.value if hasattr(assessment.threat_level, 'value') else str(assessment.threat_level),
            "event_type": "maneuver"
        }
        self.actor_database.update_threat_history(actor_id, threat_event)
        
        # Update pattern analysis
        # Handle both enum and string values for maneuver_type
        maneuver_type_value = event.maneuver_type.value if hasattr(event.maneuver_type, 'value') else str(event.maneuver_type)
        activity_data = {
            "timestamp": event.timestamp.isoformat(),
            "delta_v": event.delta_v,
            "maneuver_type": maneuver_type_value
        }
        self.pattern_analyzer.update_activity(actor_id, activity_data)
        
        # Update scoring history
        if actor_id not in self.scoring_history:
            self.scoring_history[actor_id] = []
        self.scoring_history[actor_id].append(assessment.hostility_score)
        
        # Keep only last 100 scores
        if len(self.scoring_history[actor_id]) > 100:
            self.scoring_history[actor_id] = self.scoring_history[actor_id][-100:]
    
    async def _publish_assessment(self, assessment: HostilityAssessment):
        """Publish hostility assessment to Kafka."""
        try:
            message = AIAnalysisMessage(
                message_type="hostility_assessment_result",
                analysis_type="hostility_assessment",
                payload=assessment.dict(),
                correlation_id=assessment.event_id
            )
            
            await self.kafka_adapter.publish("astroshield.ai.hostility_assessment", message.dict())
            logger.debug(f"Published hostility assessment for event {assessment.event_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish assessment: {str(e)}")
    
    async def batch_assess(self, events: List[ManeuverEvent]) -> List[HostilityAssessment]:
        """Assess hostility for multiple events in batch."""
        assessments = []
        
        # Process events concurrently
        tasks = [self.assess_hostility(event) for event in events]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch assessment failed for event {events[i].event_id}: {str(result)}")
                else:
                    assessments.append(result)
            
            return assessments
            
        except Exception as e:
            logger.error(f"Batch assessment failed: {str(e)}")
            return []
    
    def get_actor_threat_summary(self, actor_id: str) -> Dict[str, Any]:
        """Get threat summary for an actor."""
        profile = self.actor_database.get_actor_profile(actor_id)
        historical_score = self.actor_database.get_historical_threat_score(actor_id)
        recent_scores = self.scoring_history.get(actor_id, [])
        
        # Handle both enum and string values for actor type
        actor_type_value = profile["type"].value if hasattr(profile["type"], 'value') else str(profile["type"])
        
        return {
            "actor_id": actor_id,
            "actor_type": actor_type_value,
            "base_threat": profile["base_threat"],
            "historical_threat": historical_score,
            "recent_average": np.mean(recent_scores) if recent_scores else 0.0,
            "assessment_count": len(recent_scores),
            "reliability": profile["reliability"]
        }
    
    def update_scoring_weights(self, new_factors: ThreatScoringFactors):
        """Update threat scoring factor weights."""
        self.scoring_factors = new_factors
        logger.info("Updated threat scoring weights")
        
        # Clear cache to force re-scoring with new weights
        self.assessment_cache.clear()


# Factory function for easy instantiation
def create_hostility_scorer(config: Optional[ModelConfig] = None, 
                          kafka_adapter=None) -> HostilityScorer:
    """Create and return a HostilityScorer instance."""
    return HostilityScorer(config=config, kafka_adapter=kafka_adapter) 