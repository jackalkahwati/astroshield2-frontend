from flask import Blueprint, request, jsonify
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from infrastructure.circuit_breaker import circuit_breaker
from infrastructure.monitoring import MonitoringService
from infrastructure.bulkhead import BulkheadManager
from services.rpo_shape_analysis_service import RPOShapeAnalysisService

logger = logging.getLogger(__name__)
rpo_shape_bp = Blueprint('rpo_shape', __name__)
monitoring = MonitoringService()
bulkhead = BulkheadManager()

# Initialize the RPO Shape Analysis service
rpo_shape_service = RPOShapeAnalysisService()

@rpo_shape_bp.route('/analyze_trajectory', methods=['POST'])
@circuit_breaker
@bulkhead.limit('analysis')
async def analyze_trajectory():
    """Analyze a spacecraft trajectory for RPO shape patterns"""
    with monitoring.create_span("api_rpo_shape_analyze_trajectory") as span:
        try:
            data = request.get_json()
            spacecraft_id = data.get('spacecraft_id')
            trajectory_data = data.get('trajectory_data', [])
            
            span.set_attribute("spacecraft_id", spacecraft_id)
            span.set_attribute("trajectory_points", len(trajectory_data))
            
            # Call the service
            result = await rpo_shape_service.analyze_trajectory(spacecraft_id, trajectory_data)
            
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error analyzing trajectory: {str(e)}")
            return jsonify({'error': str(e)}), 500

@rpo_shape_bp.route('/compare_trajectories', methods=['POST'])
@circuit_breaker
@bulkhead.limit('analysis')
async def compare_trajectories():
    """Compare multiple trajectories to identify patterns across spacecraft"""
    with monitoring.create_span("api_rpo_shape_compare_trajectories") as span:
        try:
            data = request.get_json()
            trajectory_ids = data.get('trajectory_ids', [])
            
            span.set_attribute("trajectory_count", len(trajectory_ids))
            
            # Call the service
            result = await rpo_shape_service.compare_trajectories(trajectory_ids)
            
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error comparing trajectories: {str(e)}")
            return jsonify({'error': str(e)}), 500

@rpo_shape_bp.route('/detect_anomalies', methods=['POST'])
@circuit_breaker
@bulkhead.limit('analysis')
async def detect_anomalies():
    """Detect anomalies in a spacecraft trajectory"""
    with monitoring.create_span("api_rpo_shape_detect_anomalies") as span:
        try:
            data = request.get_json()
            spacecraft_id = data.get('spacecraft_id')
            trajectory_data = data.get('trajectory_data', [])
            
            span.set_attribute("spacecraft_id", spacecraft_id)
            span.set_attribute("trajectory_points", len(trajectory_data))
            
            # Call the service
            result = await rpo_shape_service.detect_anomalies(spacecraft_id, trajectory_data)
            
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return jsonify({'error': str(e)}), 500

@rpo_shape_bp.route('/patterns', methods=['GET'])
@circuit_breaker
async def get_patterns():
    """Get list of known RPO shape patterns"""
    with monitoring.create_span("api_rpo_shape_get_patterns") as span:
        try:
            from services.rpo_shape_analysis_service import RPO_PATTERNS, SUSPICIOUS_PATTERNS
            
            patterns = []
            for pattern_id, description in RPO_PATTERNS.items():
                patterns.append({
                    "pattern_id": pattern_id,
                    "description": description,
                    "is_suspicious": pattern_id in SUSPICIOUS_PATTERNS
                })
            
            result = {
                "patterns": patterns,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error retrieving patterns: {str(e)}")
            return jsonify({'error': str(e)}), 500

@rpo_shape_bp.route('/suspicious_activity', methods=['GET'])
@circuit_breaker
async def get_suspicious_activity():
    """Get list of recent suspicious RPO activities"""
    with monitoring.create_span("api_rpo_shape_get_suspicious_activity") as span:
        try:
            # In a real implementation, this would query a database
            # For now, return sample data
            
            current_time = datetime.utcnow()
            
            result = {
                "activities": [
                    {
                        "spacecraft_id": "SAT-12345",
                        "pattern": "intercept",
                        "confidence": 0.92,
                        "timestamp": (current_time.replace(hour=current_time.hour-2)).isoformat(),
                        "details": {
                            "target_spacecraft_id": "SAT-54321",
                            "min_distance_km": 5.2,
                            "max_velocity_kmps": 3.8
                        }
                    },
                    {
                        "spacecraft_id": "SAT-67890",
                        "pattern": "zigzag",
                        "confidence": 0.85,
                        "timestamp": (current_time.replace(hour=current_time.hour-5)).isoformat(),
                        "details": {
                            "target_spacecraft_id": "SAT-09876",
                            "min_distance_km": 12.7,
                            "max_velocity_kmps": 2.3
                        }
                    }
                ],
                "timestamp": current_time.isoformat()
            }
            
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error retrieving suspicious activity: {str(e)}")
            return jsonify({'error': str(e)}), 500 