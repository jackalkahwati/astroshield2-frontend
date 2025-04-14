from functools import wraps
from typing import Type, Optional
from flask import request, jsonify
from pydantic import BaseModel, ValidationError
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def validate_request(request_model: Optional[Type[BaseModel]] = None, 
                    response_model: Optional[Type[BaseModel]] = None):
    """
    Decorator for validating request and response data using Pydantic models.
    
    Args:
        request_model: Pydantic model for request validation
        response_model: Pydantic model for response validation
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate request data
            if request_model:
                try:
                    if request.is_json:
                        data = request.get_json()
                    else:
                        data = request.args.to_dict()
                    
                    validated_data = request_model(**data)
                    kwargs['validated_data'] = validated_data
                except ValidationError as e:
                    logger.error(f"Request validation error: {str(e)}")
                    return jsonify({
                        'error': 'Validation error',
                        'details': e.errors()
                    }), 400
                except Exception as e:
                    logger.error(f"Request processing error: {str(e)}")
                    return jsonify({
                        'error': 'Bad request',
                        'message': str(e)
                    }), 400

            # Execute the endpoint
            try:
                result = func(*args, **kwargs)
                
                # If response_model is specified and the response is successful
                if response_model and isinstance(result, tuple):
                    response_data, status_code = result
                    if 200 <= status_code < 300:
                        try:
                            validated_response = response_model(**response_data)
                            return jsonify(validated_response.dict()), status_code
                        except ValidationError as e:
                            logger.error(f"Response validation error: {str(e)}")
                            return jsonify({
                                'error': 'Internal validation error',
                                'message': 'Server response failed validation'
                            }), 500
                
                return result
                
            except Exception as e:
                logger.error(f"Endpoint execution error: {str(e)}")
                return jsonify({
                    'error': 'Internal server error',
                    'message': str(e)
                }), 500
                
        return wrapper
    return decorator

def validate_conjunction_parameters(params):
    """
    Validate conjunction event parameters.
    
    Args:
        params: Dict containing conjunction parameters
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not params:
        return False, "Conjunction parameters cannot be empty"
    
    # Required parameters
    required_fields = ['primary_object_id', 'secondary_object_id', 'conjunction_time']
    missing_fields = [field for field in required_fields if field not in params]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate object IDs
    if not is_valid_object_id(params['primary_object_id']):
        return False, f"Invalid primary object ID: {params['primary_object_id']}"
    
    if not is_valid_object_id(params['secondary_object_id']):
        return False, f"Invalid secondary object ID: {params['secondary_object_id']}"
    
    # Validate conjunction time
    try:
        conjunction_time = datetime.fromisoformat(params['conjunction_time'])
        
        # Time window validation (not in the past, not too far in the future)
        now = datetime.utcnow()
        max_future = now + timedelta(days=30)  # Max 30 days in the future
        
        if conjunction_time < now - timedelta(hours=1):  # Allow slight past for processing delays
            return False, "Conjunction time cannot be in the past"
        
        if conjunction_time > max_future:
            return False, f"Conjunction time cannot be more than 30 days in the future"
    except ValueError:
        return False, f"Invalid conjunction time format: {params['conjunction_time']}. Use ISO format."
    
    # Validate numeric parameters if present
    numeric_fields = {
        'miss_distance': {'min': 0, 'max': 100000},  # in meters
        'relative_velocity': {'min': 0, 'max': 20000},  # in m/s
        'collision_probability': {'min': 0, 'max': 1.0},
        'time_to_closest_approach': {'min': 0, 'max': 2592000}  # 30 days in seconds
    }
    
    for field, limits in numeric_fields.items():
        if field in params:
            try:
                value = float(params[field])
                if value < limits['min'] or value > limits['max']:
                    return False, f"{field} must be between {limits['min']} and {limits['max']}"
            except ValueError:
                return False, f"Invalid {field} value: {params[field]}"
    
    return True, ""

def is_valid_object_id(object_id):
    """
    Validate if the object ID is in a valid format.
    
    Args:
        object_id: Object ID to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check for NORAD ID format (numeric)
    if object_id.isdigit():
        return True
    
    # Check for COSPAR ID format (YYYY-NNNX)
    if len(object_id) >= 8:
        parts = object_id.split('-')
        if len(parts) == 2 and len(parts[0]) == 4 and parts[0].isdigit():
            return True
    
    return False

def validate_analysis_window(start_date, end_date):
    """
    Validate the analysis time window.
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        # Check if dates are valid
        if start > end:
            return False, "Start date cannot be after end date"
        
        # Check for maximum window size (e.g., 90 days)
        max_window = timedelta(days=90)
        if end - start > max_window:
            return False, f"Analysis window cannot exceed {max_window.days} days"
        
        # Check if dates are not too far in the past
        now = datetime.utcnow()
        max_past = now - timedelta(days=365 * 5)  # 5 years
        if start < max_past:
            return False, "Start date cannot be more than 5 years in the past"
        
        # Check if end date is not in the future
        if end > now + timedelta(days=30):
            return False, "End date cannot be more than 30 days in the future"
            
        return True, ""
        
    except ValueError:
        return False, "Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)."
