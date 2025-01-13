from functools import wraps
from typing import Type, Optional
from flask import request, jsonify
from pydantic import BaseModel, ValidationError
import logging

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
