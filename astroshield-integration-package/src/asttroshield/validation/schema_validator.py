"""
Schema validation for Kafka messages and other data structures.
"""
import json
import os
import logging
import jsonschema
from typing import Dict, Any, Optional

logger = logging.getLogger("astroshield.validation")

class SchemaValidator:
    """Schema validation for Kafka messages"""
    
    def __init__(self, schema_dir=None):
        """
        Initialize the validator
        
        Args:
            schema_dir: Directory containing schema files
        """
        self.schema_dir = schema_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "../../../schemas"
        )
        self.schemas = {}
        
    def load_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Load a schema from file
        
        Args:
            schema_name: Name of the schema file
            
        Returns:
            Schema as a dictionary
        """
        if schema_name not in self.schemas:
            schema_path = os.path.join(self.schema_dir, f"{schema_name}.schema.json")
            try:
                with open(schema_path, 'r') as f:
                    self.schemas[schema_name] = json.load(f)
            except FileNotFoundError:
                logger.error(f"Schema file not found: {schema_path}")
                raise ValueError(f"Schema '{schema_name}' not found")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in schema file: {schema_path}")
                raise ValueError(f"Invalid JSON in schema '{schema_name}'")
                
        return self.schemas[schema_name]
    
    def validate(self, message: Dict[str, Any], schema_name: str) -> bool:
        """
        Validate a message against a schema
        
        Args:
            message: Message to validate
            schema_name: Name of the schema
            
        Returns:
            True if valid, False otherwise
        """
        try:
            schema = self.load_schema(schema_name)
            jsonschema.validate(message, schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.warning(f"Validation error for schema {schema_name}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            return False
            
    def validate_by_message_type(self, message: Dict[str, Any]) -> bool:
        """
        Validate a message based on its message type
        
        Args:
            message: Message to validate
            
        Returns:
            True if valid, False otherwise
        """
        message_type = message.get("header", {}).get("messageType")
        if not message_type:
            logger.warning("Message has no messageType in header")
            return False
            
        schema_mapping = {
            "maneuver-detected": "maneuvers.detected",
            "dmd-object-update": "dmd.orbit.update",
            "weather-data-update": "weather.data",
            "observation-window-recommended": "observation.windows"
        }
        
        schema_name = schema_mapping.get(message_type)
        if not schema_name:
            logger.warning(f"No schema defined for message type: {message_type}")
            return False
            
        return self.validate(message, schema_name)
    
    def get_validation_errors(self, message: Dict[str, Any], schema_name: str) -> Optional[str]:
        """
        Get validation errors for a message against a schema
        
        Args:
            message: Message to validate
            schema_name: Name of the schema
            
        Returns:
            Error message or None if valid
        """
        try:
            schema = self.load_schema(schema_name)
            jsonschema.validate(message, schema)
            return None
        except jsonschema.exceptions.ValidationError as e:
            return str(e)
        except Exception as e:
            return str(e) 