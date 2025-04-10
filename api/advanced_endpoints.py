from flask import Blueprint, request, jsonify
from asttroshield.analysis.indicator_models import (
    SystemInteraction, EclipsePeriod, TrackingData,
    UNRegistryEntry, OrbitOccupancyData, StimulationEvent,
    LaunchTrackingData
)
from asttroshield.analysis.advanced_indicators import (
    StimulationEvaluator, LaunchTrackingEvaluator,
    EclipseTrackingEvaluator, OrbitOccupancyEvaluator,
    UNRegistryEvaluator
)
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
advanced_blueprint = Blueprint('advanced', __name__)

# Initialize evaluators
stimulation_evaluator = StimulationEvaluator()
launch_tracking_evaluator = LaunchTrackingEvaluator()
eclipse_tracking_evaluator = EclipseTrackingEvaluator()
orbit_occupancy_evaluator = OrbitOccupancyEvaluator()
un_registry_evaluator = UNRegistryEvaluator()

@advanced_blueprint.route('/stimulation/<spacecraft_id>', methods=['POST'])
def analyze_stimulation(spacecraft_id: str):
    """Analyze system stimulation events for a spacecraft"""
    try:
        data = request.get_json()
        event = StimulationEvent(**data)
        
        # Analyze stimulation
        indicators = stimulation_evaluator.analyze_stimulation(
            {'spacecraft_id': spacecraft_id, **event.dict()},
            data.get('system_interactions', {})
        )
        
        return jsonify({
            'status': 'success',
            'spacecraft_id': spacecraft_id,
            'indicators': [i.dict() for i in indicators]
        })
    except Exception as e:
        logger.error(f"Stimulation analysis error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@advanced_blueprint.route('/tracking/launch/<launch_id>', methods=['POST'])
def analyze_launch_tracking(launch_id: str):
    """Analyze tracking data for a launch event"""
    try:
        data = request.get_json()
        tracking_data = LaunchTrackingData(**data)
        
        # Analyze launch tracking
        indicators = launch_tracking_evaluator.analyze_launch_tracking(
            {'launch_id': launch_id, **tracking_data.dict()},
            data.get('current_tracking', {})
        )
        
        return jsonify({
            'status': 'success',
            'launch_id': launch_id,
            'indicators': [i.dict() for i in indicators]
        })
    except Exception as e:
        logger.error(f"Launch tracking analysis error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@advanced_blueprint.route('/tracking/eclipse/<spacecraft_id>', methods=['POST'])
def analyze_eclipse_tracking(spacecraft_id: str):
    """Analyze tracking during eclipse periods"""
    try:
        data = request.get_json()
        eclipse_data = EclipsePeriod(**data['eclipse_data'])
        tracking_data = TrackingData(**data['tracking_data'])
        
        # Analyze eclipse tracking
        indicators = eclipse_tracking_evaluator.analyze_eclipse_tracking(
            tracking_data.dict(),
            {'eclipse_periods': [eclipse_data.dict()]}
        )
        
        return jsonify({
            'status': 'success',
            'spacecraft_id': spacecraft_id,
            'indicators': [i.dict() for i in indicators]
        })
    except Exception as e:
        logger.error(f"Eclipse tracking analysis error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@advanced_blueprint.route('/orbit/occupancy/<region_id>', methods=['POST'])
def analyze_orbit_occupancy(region_id: str):
    """Analyze orbit occupancy for a specific region"""
    try:
        data = request.get_json()
        occupancy_data = OrbitOccupancyData(**data)
        
        # Analyze orbit occupancy
        indicators = orbit_occupancy_evaluator.analyze_orbit_occupancy(
            occupancy_data.dict(),
            data.get('catalog_data', {})
        )
        
        return jsonify({
            'status': 'success',
            'region_id': region_id,
            'indicators': [i.dict() for i in indicators]
        })
    except Exception as e:
        logger.error(f"Orbit occupancy analysis error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@advanced_blueprint.route('/registry/verify/<spacecraft_id>', methods=['POST'])
def verify_un_registry(spacecraft_id: str):
    """Verify UN registry status for a spacecraft"""
    try:
        data = request.get_json()
        registry_data = UNRegistryEntry(**data['registry_data'])
        
        # Analyze registry status
        indicators = un_registry_evaluator.analyze_un_registry(
            {'spacecraft_id': spacecraft_id, **data.get('object_data', {})},
            {'entries': [registry_data.dict()]}
        )
        
        return jsonify({
            'status': 'success',
            'spacecraft_id': spacecraft_id,
            'indicators': [i.dict() for i in indicators]
        })
    except Exception as e:
        logger.error(f"UN registry verification error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Batch analysis endpoints
@advanced_blueprint.route('/batch/analyze', methods=['POST'])
def batch_analyze():
    """Perform batch analysis of multiple indicators"""
    try:
        data = request.get_json()
        results = {}
        
        # Process each analysis type if data is provided
        if 'stimulation' in data:
            results['stimulation'] = stimulation_evaluator.analyze_stimulation(
                data['stimulation'],
                data.get('system_interactions', {})
            )
        
        if 'launch_tracking' in data:
            results['launch_tracking'] = launch_tracking_evaluator.analyze_launch_tracking(
                data['launch_tracking'],
                data.get('current_tracking', {})
            )
        
        if 'eclipse_tracking' in data:
            results['eclipse_tracking'] = eclipse_tracking_evaluator.analyze_eclipse_tracking(
                data['tracking_data'],
                data['eclipse_data']
            )
        
        if 'orbit_occupancy' in data:
            results['orbit_occupancy'] = orbit_occupancy_evaluator.analyze_orbit_occupancy(
                data['occupancy_data'],
                data.get('catalog_data', {})
            )
        
        if 'un_registry' in data:
            results['un_registry'] = un_registry_evaluator.analyze_un_registry(
                data['object_data'],
                data['registry_data']
            )
        
        return jsonify({
            'status': 'success',
            'results': {k: [i.dict() for i in v] for k, v in results.items()}
        })
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
