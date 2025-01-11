from flask import jsonify, request
from functools import wraps
from models import Spacecraft
import traceback

class APIGateway:
    @staticmethod
    def validate_spacecraft(f):
        @wraps(f)
        def decorated_function(spacecraft_id, *args, **kwargs):
            try:
                spacecraft = Spacecraft.query.get(spacecraft_id)
                if not spacecraft:
                    return jsonify({
                        'error': 'Not Found',
                        'message': f'Spacecraft with ID {spacecraft_id} not found'
                    }), 404
                return f(spacecraft_id, *args, **kwargs)
            except Exception as e:
                return jsonify({
                    'error': 'Internal Server Error',
                    'message': str(e),
                    'details': traceback.format_exc()
                }), 500
        return decorated_function

    @staticmethod
    def handle_request(service_method):
        """Generic request handler with error handling"""
        try:
            result = service_method()
            return jsonify(result)
        except ValueError as e:
            return jsonify({
                'error': 'Validation Error',
                'message': str(e)
            }), 400
        except Exception as e:
            return jsonify({
                'error': 'Internal Server Error',
                'message': str(e),
                'details': traceback.format_exc()
            }), 500

    @staticmethod
    def validate_request_data(required_fields=None):
        """Validate request data middleware"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if required_fields:
                    data = request.get_json()
                    if not data:
                        return jsonify({
                            'error': 'Bad Request',
                            'message': 'No JSON data provided'
                        }), 400
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        return jsonify({
                            'error': 'Bad Request',
                            'message': f'Missing required fields: {", ".join(missing_fields)}'
                        }), 400
                return f(*args, **kwargs)
            return decorated_function
        return decorator
