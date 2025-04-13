from flask import Blueprint, request, jsonify
from app.models.ccdm import ThreatAssessmentRequest, ObjectThreatAssessment
from app.services.ccdm import CCDMService
import logging

# Import our mock circuit breaker instead of the one from tenacity
from infrastructure.circuit_breaker import circuit_breaker

logger = logging.getLogger(__name__)
ccdm_bp = Blueprint('ccdm', __name__, url_prefix='/api/v1/ccdm')
ccdm_service = CCDMService()

@ccdm_bp.route('/threat-assessment', methods=['POST'])
@circuit_breaker
def assess_threat():
    """Assess threat level of a space object"""
    try:
        data = request.get_json()
        request_obj = ThreatAssessmentRequest(**data)
        assessment = ccdm_service.assess_threat(request_obj)
        return jsonify(assessment.dict())
    except Exception as e:
        logger.error(f"Error assessing threat: {str(e)}")
        return jsonify({"error": str(e)}), 500

@ccdm_bp.route('/quick-assessment/<int:norad_id>', methods=['GET'])
@circuit_breaker
def quick_assess(norad_id):
    """Quick threat assessment by NORAD ID"""
    try:
        assessment = ccdm_service.quick_assess_norad_id(norad_id)
        return jsonify(assessment.dict())
    except Exception as e:
        logger.error(f"Error performing quick assessment: {str(e)}")
        return jsonify({"error": str(e)}), 500 