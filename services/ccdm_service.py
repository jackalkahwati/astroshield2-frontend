"""CCDM Service with enhanced ML capabilities."""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from analysis.ml_evaluators import MLManeuverEvaluator, MLSignatureEvaluator, MLAMREvaluator
import random

logger = logging.getLogger(__name__)

class CCDMService:
    def __init__(self, db=None):
        self.maneuver_evaluator = MLManeuverEvaluator()
        self.signature_evaluator = MLSignatureEvaluator()
        self.amr_evaluator = MLAMREvaluator()
        self.db = db  # SQLAlchemy session
        
    def analyze_conjunction(self, spacecraft_id: str, other_spacecraft_id: str) -> Dict[str, Any]:
        """Analyze potential conjunction between two spacecraft using ML models."""
        try:
            # Get trajectory data
            trajectory_data = self._get_trajectory_data(spacecraft_id, other_spacecraft_id)
            
            # Analyze maneuvers
            maneuver_indicators = self.maneuver_evaluator.analyze_maneuvers(trajectory_data)
            
            # Get signature data
            optical_data = self._get_optical_data(spacecraft_id)
            radar_data = self._get_radar_data(spacecraft_id)
            
            # Analyze signatures
            signature_indicators = self.signature_evaluator.analyze_signatures(optical_data, radar_data)
            
            # Get AMR data
            amr_data = self._get_amr_data(spacecraft_id)
            
            # Analyze AMR
            amr_indicators = self.amr_evaluator.analyze_amr(amr_data)
            
            # Combine all indicators
            all_indicators = maneuver_indicators + signature_indicators + amr_indicators
            
            # Store results in database if available
            if self.db:
                self._store_analysis_results(spacecraft_id, all_indicators)
            
            return {
                'status': 'operational',
                'indicators': [indicator.dict() for indicator in all_indicators],
                'analysis_timestamp': datetime.utcnow(),
                'risk_assessment': self._calculate_risk(all_indicators)
            }
            
        except Exception as e:
            logger.error(f"Error in conjunction analysis: {str(e)}")
            return {
                'status': 'error',
                'message': f'Analysis failed: {str(e)}'
            }

    def get_active_conjunctions(self, spacecraft_id: str) -> List[Dict[str, Any]]:
        """Get list of active conjunctions with ML-enhanced risk assessment."""
        try:
            # Check if we should use the database
            if self.db:
                return self._get_conjunctions_from_db(spacecraft_id)
            
            # Fallback to simulated data
            # Get nearby spacecraft
            nearby_spacecraft = self._get_nearby_spacecraft(spacecraft_id)
            
            conjunctions = []
            for other_id in nearby_spacecraft:
                analysis = self.analyze_conjunction(spacecraft_id, other_id)
                if analysis['status'] == 'operational':
                    conjunctions.append({
                        'spacecraft_id': other_id,
                        'analysis': analysis
                    })
                    
            return conjunctions
            
        except Exception as e:
            logger.error(f"Error getting active conjunctions: {str(e)}")
            return []

    def analyze_conjunction_trends(self, spacecraft_id: str, hours: int = 24) -> Dict[str, Any]:
        """Analyze conjunction trends using ML models."""
        try:
            # Get historical conjunction data
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Check if we should use database
            if self.db:
                historical_data = self._get_historical_conjunctions_from_db(spacecraft_id, start_time)
            else:
                historical_data = self._get_historical_conjunctions(spacecraft_id, start_time)
            
            # Analyze trends
            return {
                'total_conjunctions': len(historical_data),
                'risk_levels': self._analyze_risk_levels(historical_data),
                'temporal_metrics': self._analyze_temporal_trends(historical_data),
                'velocity_metrics': self._analyze_velocity_trends(historical_data),
                'ml_insights': self._get_ml_insights(historical_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conjunction trends: {str(e)}")
            return {
                'status': 'error',
                'message': f'Trend analysis failed: {str(e)}'
            }

    def _calculate_risk(self, indicators: List[Any]) -> Dict[str, Any]:
        """Calculate overall risk based on ML indicators."""
        risk_scores = {
            'maneuver': 0.0,
            'signature': 0.0,
            'amr': 0.0
        }
        
        for indicator in indicators:
            if 'maneuver' in indicator.indicator_name:
                risk_scores['maneuver'] = max(risk_scores['maneuver'], indicator.confidence_level)
            elif 'signature' in indicator.indicator_name:
                risk_scores['signature'] = max(risk_scores['signature'], indicator.confidence_level)
            elif 'amr' in indicator.indicator_name:
                risk_scores['amr'] = max(risk_scores['amr'], indicator.confidence_level)
        
        overall_risk = max(risk_scores.values())
        
        return {
            'overall_risk': overall_risk,
            'risk_factors': risk_scores,
            'risk_level': self._get_risk_level(overall_risk)
        }

    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level."""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'moderate'
        else:
            return 'low'

    # Helper methods to get data (to be implemented based on data source)
    def _get_trajectory_data(self, spacecraft_id: str, other_spacecraft_id: str) -> List[Dict[str, Any]]:
        """Get trajectory data for spacecraft."""
        # Implementation needed
        return []

    def _get_optical_data(self, spacecraft_id: str) -> Dict[str, Any]:
        """Get optical signature data."""
        # Implementation needed
        return {}

    def _get_radar_data(self, spacecraft_id: str) -> Dict[str, Any]:
        """Get radar signature data."""
        # Implementation needed
        return {}

    def _get_amr_data(self, spacecraft_id: str) -> Dict[str, Any]:
        """Get AMR data."""
        # Implementation needed
        return {}

    def _get_nearby_spacecraft(self, spacecraft_id: str) -> List[str]:
        """Get list of nearby spacecraft IDs."""
        # Implementation needed
        return []

    def _get_historical_conjunctions(self, spacecraft_id: str, start_time: datetime) -> List[Dict[str, Any]]:
        """Get historical conjunction data."""
        # Implementation needed
        return []

    def _analyze_risk_levels(self, historical_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze risk levels in historical data."""
        risk_levels = {
            'critical': 0,
            'high': 0,
            'moderate': 0,
            'low': 0
        }
        # Implementation needed
        return risk_levels

    def _analyze_temporal_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal trends in historical data."""
        return {
            'hourly_rate': 0.0,
            'peak_hour': None,
            'trend_direction': 'stable'
        }

    def _analyze_velocity_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze velocity trends in historical data."""
        return {
            'average_velocity': 0.0,
            'max_velocity': 0.0,
            'velocity_trend': 'stable'
        }

    def _get_ml_insights(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get ML-based insights from historical data."""
        return {
            'pattern_detected': False,
            'anomaly_score': 0.0,
            'confidence': 0.0
        }

    def get_historical_analysis(self, norad_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get historical analysis data for a specific spacecraft and date range.
        
        Args:
            norad_id: The NORAD ID of the spacecraft
            start_date: The start date in ISO format
            end_date: The end date in ISO format
            
        Returns:
            Dictionary containing historical analysis data
        """
        try:
            # Convert string dates to datetime objects
            start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            # Try to get data from database if available
            if self.db:
                analysis_points = self._get_historical_analysis_from_db(norad_id, start_datetime, end_datetime)
                
                # If no data in database, fall back to generated data
                if not analysis_points:
                    analysis_points = self._generate_historical_data_points(norad_id, start_datetime, end_datetime)
            else:
                # Generate simulated data
                analysis_points = self._generate_historical_data_points(norad_id, start_datetime, end_datetime)
            
            # Calculate overall trend summary
            trend_summary = self._calculate_trend_summary(analysis_points)
            
            return {
                "norad_id": int(norad_id) if norad_id.isdigit() else norad_id,
                "start_date": start_date,
                "end_date": end_date,
                "trend_summary": trend_summary,
                "analysis_points": analysis_points,
                "metadata": {
                    "data_source": "Database" if self.db and analysis_points else "Simulated Data",
                    "processing_version": "1.0.0",
                    "confidence_threshold": 0.7
                }
            }
            
        except Exception as e:
            logger.error(f"Error in historical analysis for spacecraft {norad_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Historical analysis failed: {str(e)}"
            }
    
    def _generate_historical_data_points(self, norad_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Generate simulated historical data points for testing."""
        # Calculate number of data points based on date range
        days_diff = (end_date - start_date).days
        num_points = max(days_diff * 2, 7)  # At least 7 points
        
        # Threat level possibilities with weights (higher weight = more common)
        threat_levels = {
            "NONE": 20,
            "LOW": 35, 
            "MEDIUM": 25,
            "HIGH": 15,
            "CRITICAL": 5
        }
        
        # Generate weighted threat levels list for random selection
        weighted_threats = []
        for level, weight in threat_levels.items():
            weighted_threats.extend([level] * weight)
        
        # Initialize simulation parameters
        points = []
        current_index = 0
        
        # Tendency to maintain similar threat levels (state machine-like)
        current_level_index = 1  # Start at LOW
        level_keys = list(threat_levels.keys())
        
        # Time interval between points
        interval = (end_date - start_date) / num_points
        
        for i in range(num_points):
            # Calculate timestamp for this point
            timestamp = start_date + interval * i
            
            # For realism, add some randomness to move up or down in threat level
            change = 0
            rand_val = random.random()
            if rand_val < 0.15:  # 15% chance to move up
                change = 1
            elif rand_val < 0.30:  # 15% chance to move down
                change = -1
                
            # Apply change but stay within bounds
            current_level_index = max(0, min(len(level_keys) - 1, current_level_index + change))
            threat_level = level_keys[current_level_index]
            
            # Generate confidence value (higher for extreme levels)
            base_confidence = 0.7 + random.random() * 0.25
            if threat_level in ["NONE", "CRITICAL"]:
                confidence = min(0.98, base_confidence + 0.1)
            else:
                confidence = base_confidence
                
            # Generate some random details appropriate for the threat level
            details = self._generate_details_for_threat_level(threat_level)
            
            # Add data point
            points.append({
                "timestamp": timestamp.isoformat(),
                "threat_level": threat_level,
                "confidence": round(confidence, 2),
                "details": details
            })
            
        return points
    
    def _generate_details_for_threat_level(self, threat_level: str) -> Dict[str, Any]:
        """Generate appropriate details for a threat level."""
        details = {}
        
        # Base anomaly score based on threat level
        anomaly_scores = {
            "NONE": 0.0,
            "LOW": 0.2,
            "MEDIUM": 0.5,
            "HIGH": 0.7,
            "CRITICAL": 0.9
        }
        
        # Add some randomness to the anomaly score
        base_score = anomaly_scores.get(threat_level, 0.0)
        anomaly_score = base_score + (random.random() * 0.2 - 0.1)  # ±0.1 randomness
        details["anomaly_score"] = round(max(0.0, min(1.0, anomaly_score)), 2)
        
        # Determine if velocity change occurred (more likely with higher threat levels)
        velocity_threshold = {
            "NONE": 0.05,
            "LOW": 0.2, 
            "MEDIUM": 0.5,
            "HIGH": 0.8,
            "CRITICAL": 0.95
        }
        
        details["velocity_change"] = random.random() < velocity_threshold.get(threat_level, 0.0)
        
        # For higher threat levels, add additional details
        if threat_level in ["HIGH", "CRITICAL"]:
            details["maneuver_detected"] = random.random() < 0.8
            details["signal_strength_change"] = round(random.random() * 0.5 + 0.3, 2)
            
        if threat_level == "CRITICAL":
            details["recommendation"] = "Immediate monitoring recommended"
            details["priority_level"] = 1
        
        return details
    
    def _calculate_trend_summary(self, analysis_points: List[Dict[str, Any]]) -> str:
        """Calculate a summary of the trend in threat levels."""
        if not analysis_points:
            return "No data available for trend analysis."
            
        # Count occurrences of each threat level
        threat_counts = {}
        for point in analysis_points:
            level = point["threat_level"]
            threat_counts[level] = threat_counts.get(level, 0) + 1
            
        # Find most common threat level
        most_common = max(threat_counts.items(), key=lambda x: x[1])
        most_common_level = most_common[0]
        
        # Determine if trend is increasing, decreasing, or stable
        threat_levels = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        level_values = {level: i for i, level in enumerate(threat_levels)}
        
        # Only look at first and last few points to determine trend
        sample_size = min(3, len(analysis_points) // 3)
        if sample_size == 0:
            sample_size = 1
            
        early_points = analysis_points[:sample_size]
        late_points = analysis_points[-sample_size:]
        
        early_avg = sum(level_values[p["threat_level"]] for p in early_points) / sample_size
        late_avg = sum(level_values[p["threat_level"]] for p in late_points) / sample_size
        
        trend_direction = "stable"
        if late_avg - early_avg > 0.5:
            trend_direction = "escalating"
        elif early_avg - late_avg > 0.5:
            trend_direction = "decreasing"
            
        # Generate summary text
        summary = f"The object has shown {trend_direction} behavior"
        
        if trend_direction == "escalating":
            early_level = threat_levels[min(4, int(early_avg))]
            late_level = threat_levels[min(4, int(late_avg))]
            summary += f" with threat levels increasing from {early_level} to {late_level}."
        elif trend_direction == "decreasing":
            early_level = threat_levels[min(4, int(early_avg))]
            late_level = threat_levels[min(4, int(late_avg))]
            summary += f" with threat levels decreasing from {early_level} to {late_level}."
        else:
            summary += f" with consistent threat levels around {most_common_level}."
            
        # Add note about specific threats if critical points exist
        if "CRITICAL" in threat_counts and threat_counts["CRITICAL"] > 0:
            summary += f" {threat_counts['CRITICAL']} critical threat incidents detected."
            
        return summary

    def get_assessment(self, object_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive CCDM assessment for a specific object.
        
        Args:
            object_id: The ID of the object to assess
            
        Returns:
            Dictionary containing the assessment data
        """
        try:
            # Try to get assessment from database if available
            if self.db:
                db_assessment = self._get_assessment_from_db(object_id)
                if db_assessment:
                    return db_assessment
            
            # Get the current time
            current_time = datetime.utcnow()
            
            # Create assessment types based on object_id to ensure consistent behavior
            # This is a simplified simulation - would use actual data in production
            object_id_hash = sum(ord(c) for c in object_id)
            assessment_type_options = [
                "maneuver_assessment", 
                "signature_analysis", 
                "conjunction_risk", 
                "anomaly_detection"
            ]
            assessment_type = assessment_type_options[object_id_hash % len(assessment_type_options)]
            
            # Generate confidence based on object_id (for consistency in testing)
            # Again, this would use real ML model confidence in production
            confidence = 0.65 + ((object_id_hash % 30) / 100)
            
            # Generate threat level
            threat_level = self._get_risk_level(confidence)
            
            # Generate assessment-type specific results
            results = self._generate_assessment_results(assessment_type, object_id, threat_level)
            
            # Generate recommendations based on assessment type and threat level
            recommendations = self._generate_recommendations(assessment_type, threat_level)
            
            # Create the assessment
            assessment = {
                "object_id": object_id,
                "assessment_type": assessment_type,
                "timestamp": current_time.isoformat(),
                "threat_level": threat_level,
                "results": results,
                "confidence_level": round(confidence, 2),
                "recommendations": recommendations
            }
            
            # Store in database if available
            if self.db:
                self._store_assessment(assessment)
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error generating assessment for object {object_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Assessment generation failed: {str(e)}"
            }
    
    def _generate_assessment_results(self, assessment_type: str, object_id: str, threat_level: str) -> Dict[str, Any]:
        """Generate appropriate results based on assessment type."""
        results = {
            "object_id": object_id,
            "threat_level": threat_level,
            "assessment_timestamp": datetime.utcnow().isoformat()
        }
        
        # Add assessment-specific metrics
        if assessment_type == "maneuver_assessment":
            results.update({
                "maneuver_detected": threat_level in ["high", "critical"],
                "delta_v_estimate": random.uniform(0.01, 0.5) if threat_level in ["high", "critical"] else 0.0,
                "trajectory_change": random.uniform(0.1, 5.0) if threat_level in ["high", "critical"] else 0.0,
                "propulsion_type": "chemical" if threat_level == "critical" else "unknown"
            })
            
        elif assessment_type == "signature_analysis":
            results.update({
                "signature_change_detected": threat_level in ["moderate", "high", "critical"],
                "optical_magnitude_change": random.uniform(0.05, 0.4) if threat_level != "low" else 0.01,
                "radar_cross_section_change": random.uniform(0.1, 0.8) if threat_level in ["high", "critical"] else 0.02,
                "thermal_emission_anomaly": threat_level == "critical"
            })
            
        elif assessment_type == "conjunction_risk":
            results.update({
                "conjunction_objects": random.randint(1, 3) if threat_level != "low" else 0,
                "minimum_distance_km": random.uniform(1.0, 20.0),
                "time_to_closest_approach_hours": random.uniform(12.0, 72.0),
                "collision_probability": self._get_collision_probability(threat_level),
                "evasive_action_recommended": threat_level in ["high", "critical"]
            })
            
        elif assessment_type == "anomaly_detection":
            results.update({
                "anomaly_score": self._get_anomaly_score(threat_level),
                "behavior_pattern": "irregular" if threat_level in ["high", "critical"] else "regular",
                "unusual_operations": random.randint(1, 4) if threat_level in ["moderate", "high", "critical"] else 0,
                "confidence_intervals_exceeded": random.randint(1, 3) if threat_level == "critical" else 0
            })
            
        return results
    
    def _generate_recommendations(self, assessment_type: str, threat_level: str) -> List[str]:
        """Generate appropriate recommendations based on assessment type and threat level."""
        recommendations = []
        
        # Common recommendations based on threat level
        if threat_level == "low":
            recommendations.append("Continue routine monitoring.")
        
        elif threat_level == "moderate":
            recommendations.append("Increase monitoring frequency.")
            recommendations.append("Review recent telemetry for potential anomalies.")
        
        elif threat_level == "high":
            recommendations.append("Immediate enhanced monitoring required.")
            recommendations.append("Notify relevant stakeholders of increased threat level.")
            recommendations.append("Prepare contingency responses.")
        
        elif threat_level == "critical":
            recommendations.append("URGENT: Immediate action required.")
            recommendations.append("Continuous monitoring at highest resolution.")
            recommendations.append("Activate emergency response protocols.")
        
        # Add assessment-specific recommendations
        if assessment_type == "maneuver_assessment" and threat_level in ["high", "critical"]:
            recommendations.append("Track trajectory changes and predict new orbit.")
            recommendations.append("Analyze propulsion signatures to determine capabilities.")
            
        elif assessment_type == "signature_analysis" and threat_level in ["moderate", "high", "critical"]:
            recommendations.append("Deploy additional sensors for signature characterization.")
            recommendations.append("Compare with known signature databases for identification.")
            
        elif assessment_type == "conjunction_risk" and threat_level in ["high", "critical"]:
            recommendations.append("Calculate potential evasive maneuvers for protected assets.")
            recommendations.append("Assess secondary conjunction risks after potential maneuvers.")
            
        elif assessment_type == "anomaly_detection" and threat_level in ["high", "critical"]:
            recommendations.append("Analyze pattern of anomalies for potential intent.")
            recommendations.append("Correlate with other objects for coordinated behavior.")
            
        return recommendations
    
    def _get_collision_probability(self, threat_level: str) -> float:
        """Get appropriate collision probability based on threat level."""
        probabilities = {
            "low": 0.0001,
            "moderate": 0.001,
            "high": 0.01,
            "critical": 0.1
        }
        base_probability = probabilities.get(threat_level, 0.0001)
        # Add some randomness
        return round(base_probability * (0.5 + random.random()), 6)
    
    def _get_anomaly_score(self, threat_level: str) -> float:
        """Get appropriate anomaly score based on threat level."""
        scores = {
            "low": 0.2,
            "moderate": 0.5,
            "high": 0.7,
            "critical": 0.9
        }
        base_score = scores.get(threat_level, 0.1)
        # Add some randomness but keep within range
        return round(min(0.99, max(0.01, base_score * (0.8 + random.random() * 0.4))), 2)

    # Database-related methods
    
    def _get_historical_analysis_from_db(self, norad_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get historical analysis data from database."""
        if not self.db:
            return []
            
        try:
            # Import models here to avoid circular imports
            from backend.models.ccdm import Spacecraft, CCDMIndicator, CCDMAssessment, ThreatLevel
            
            # Find spacecraft by NORAD ID
            spacecraft = self.db.query(Spacecraft).filter(Spacecraft.norad_id == norad_id).first()
            if not spacecraft:
                return []
                
            # Get assessments within date range
            assessments = (
                self.db.query(CCDMAssessment)
                .filter(
                    CCDMAssessment.spacecraft_id == spacecraft.id,
                    CCDMAssessment.timestamp >= start_date,
                    CCDMAssessment.timestamp <= end_date
                )
                .order_by(CCDMAssessment.timestamp)
                .all()
            )
            
            # Convert to analysis points
            analysis_points = []
            for assessment in assessments:
                analysis_points.append({
                    "timestamp": assessment.timestamp.isoformat(),
                    "threat_level": assessment.threat_level.value,
                    "confidence": assessment.confidence_level,
                    "details": assessment.results or {}
                })
                
            return analysis_points
        except Exception as e:
            logger.error(f"Error retrieving historical analysis from database: {str(e)}")
            return []
            
    def _get_conjunctions_from_db(self, spacecraft_id: str) -> List[Dict[str, Any]]:
        """Get active conjunctions from database."""
        if not self.db:
            return []
            
        try:
            # Import models here to avoid circular imports
            from backend.models.ccdm import Spacecraft, CCDMIndicator
            
            # Find spacecraft by ID or NORAD ID
            spacecraft = None
            if spacecraft_id.isdigit():
                spacecraft = self.db.query(Spacecraft).filter(
                    (Spacecraft.id == int(spacecraft_id)) | 
                    (Spacecraft.norad_id == spacecraft_id)
                ).first()
            else:
                spacecraft = self.db.query(Spacecraft).filter(Spacecraft.norad_id == spacecraft_id).first()
                
            if not spacecraft:
                return []
                
            # Get recent indicators (last 24 hours)
            recent_time = datetime.utcnow() - timedelta(hours=24)
            indicators = (
                self.db.query(CCDMIndicator)
                .filter(
                    CCDMIndicator.spacecraft_id == spacecraft.id,
                    CCDMIndicator.timestamp >= recent_time,
                    CCDMIndicator.conjunction_type.ilike("%CONJUNCTION%") | 
                    CCDMIndicator.conjunction_type.ilike("%APPROACH%")
                )
                .order_by(CCDMIndicator.timestamp.desc())
                .all()
            )
            
            # Convert to conjunction data
            conjunctions = []
            for indicator in indicators:
                conjunctions.append({
                    'spacecraft_id': str(indicator.spacecraft_id),
                    'analysis': {
                        'status': 'operational',
                        'indicators': [indicator.to_dict()],
                        'analysis_timestamp': indicator.timestamp.isoformat(),
                        'risk_assessment': {
                            'overall_risk': indicator.probability_of_collision or 0.0,
                            'risk_level': self._get_risk_level(indicator.probability_of_collision or 0.0)
                        }
                    }
                })
                
            return conjunctions
        except Exception as e:
            logger.error(f"Error retrieving conjunctions from database: {str(e)}")
            return []
    
    def _get_historical_conjunctions_from_db(self, spacecraft_id: str, start_time: datetime) -> List[Dict[str, Any]]:
        """Get historical conjunction data from database."""
        if not self.db:
            return []
            
        try:
            # Import models here to avoid circular imports
            from backend.models.ccdm import Spacecraft, CCDMIndicator
            
            # Find spacecraft by ID or NORAD ID
            spacecraft = None
            if spacecraft_id.isdigit():
                spacecraft = self.db.query(Spacecraft).filter(
                    (Spacecraft.id == int(spacecraft_id)) | 
                    (Spacecraft.norad_id == spacecraft_id)
                ).first()
            else:
                spacecraft = self.db.query(Spacecraft).filter(Spacecraft.norad_id == spacecraft_id).first()
                
            if not spacecraft:
                return []
                
            # Get indicators since start_time
            indicators = (
                self.db.query(CCDMIndicator)
                .filter(
                    CCDMIndicator.spacecraft_id == spacecraft.id,
                    CCDMIndicator.timestamp >= start_time
                )
                .order_by(CCDMIndicator.timestamp)
                .all()
            )
            
            # Convert to historical data
            historical_data = []
            for indicator in indicators:
                historical_data.append(indicator.to_dict())
                
            return historical_data
        except Exception as e:
            logger.error(f"Error retrieving historical conjunctions from database: {str(e)}")
            return []
            
    def _get_assessment_from_db(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get assessment from database."""
        if not self.db:
            return None
            
        try:
            # Import models here to avoid circular imports
            from backend.models.ccdm import Spacecraft, CCDMAssessment
            
            # Find spacecraft by ID or NORAD ID
            spacecraft = None
            if object_id.isdigit():
                spacecraft = self.db.query(Spacecraft).filter(
                    (Spacecraft.id == int(object_id)) | 
                    (Spacecraft.norad_id == object_id)
                ).first()
            else:
                spacecraft = self.db.query(Spacecraft).filter(Spacecraft.norad_id == object_id).first()
                
            if not spacecraft:
                return None
                
            # Get most recent assessment
            assessment = (
                self.db.query(CCDMAssessment)
                .filter(CCDMAssessment.spacecraft_id == spacecraft.id)
                .order_by(CCDMAssessment.timestamp.desc())
                .first()
            )
            
            if not assessment:
                return None
                
            # Convert to API format
            return {
                "object_id": object_id,
                "assessment_type": assessment.assessment_type,
                "timestamp": assessment.timestamp.isoformat(),
                "threat_level": assessment.threat_level.value.lower(),
                "results": assessment.results or {},
                "confidence_level": assessment.confidence_level,
                "recommendations": assessment.recommendations or []
            }
        except Exception as e:
            logger.error(f"Error retrieving assessment from database: {str(e)}")
            return None
            
    def _store_assessment(self, assessment: Dict[str, Any]) -> None:
        """Store assessment in database."""
        if not self.db:
            return
            
        try:
            # Import models here to avoid circular imports
            from backend.models.ccdm import Spacecraft, CCDMAssessment, ThreatLevel
            
            # Find or create spacecraft
            object_id = assessment["object_id"]
            spacecraft = None
            
            if object_id.isdigit():
                spacecraft = self.db.query(Spacecraft).filter(
                    (Spacecraft.id == int(object_id)) | 
                    (Spacecraft.norad_id == object_id)
                ).first()
            else:
                spacecraft = self.db.query(Spacecraft).filter(Spacecraft.norad_id == object_id).first()
                
            if not spacecraft:
                # Create a new spacecraft record
                spacecraft = Spacecraft(
                    norad_id=object_id,
                    name=f"Object {object_id}"
                )
                self.db.add(spacecraft)
                self.db.flush()  # Generate ID
                
            # Create assessment record
            timestamp = datetime.fromisoformat(assessment["timestamp"].replace("Z", "+00:00"))
            threat_level = assessment["threat_level"].upper()
            
            db_assessment = CCDMAssessment(
                spacecraft_id=spacecraft.id,
                assessment_type=assessment["assessment_type"],
                threat_level=ThreatLevel[threat_level],
                confidence_level=assessment["confidence_level"],
                summary=assessment.get("summary"),
                results=assessment["results"],
                recommendations=assessment["recommendations"],
                timestamp=timestamp
            )
            
            self.db.add(db_assessment)
            self.db.commit()
        except Exception as e:
            logger.error(f"Error storing assessment in database: {str(e)}")
            self.db.rollback()
            
    def _store_analysis_results(self, spacecraft_id: str, indicators: List[Any]) -> None:
        """Store analysis results in database."""
        if not self.db:
            return
            
        try:
            # Import models here to avoid circular imports
            from backend.models.ccdm import Spacecraft, CCDMIndicator
            
            # Find or create spacecraft
            spacecraft = None
            
            if spacecraft_id.isdigit():
                spacecraft = self.db.query(Spacecraft).filter(
                    (Spacecraft.id == int(spacecraft_id)) | 
                    (Spacecraft.norad_id == spacecraft_id)
                ).first()
            else:
                spacecraft = self.db.query(Spacecraft).filter(Spacecraft.norad_id == spacecraft_id).first()
                
            if not spacecraft:
                # Create a new spacecraft record
                spacecraft = Spacecraft(
                    norad_id=spacecraft_id,
                    name=f"Object {spacecraft_id}"
                )
                self.db.add(spacecraft)
                self.db.flush()  # Generate ID
                
            # Create indicator records
            for indicator in indicators:
                indicator_dict = indicator.dict()
                
                # Extract data from indicator
                conjunction_type = "MANEUVER_DETECTED"
                if "signature_" in indicator_dict["indicator_name"]:
                    conjunction_type = "SIGNATURE_CHANGE"
                elif "amr_" in indicator_dict["indicator_name"]:
                    conjunction_type = "AMR_ANOMALY"
                    
                db_indicator = CCDMIndicator(
                    spacecraft_id=spacecraft.id,
                    conjunction_type=conjunction_type,
                    relative_velocity=indicator_dict["details"].get("velocity_change", 0.0),
                    probability_of_collision=indicator_dict["confidence_level"],
                    details=indicator_dict,
                    timestamp=datetime.utcnow()
                )
                
                self.db.add(db_indicator)
                
            self.db.commit()
        except Exception as e:
            logger.error(f"Error storing analysis results in database: {str(e)}")
            self.db.rollback()
