import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import os
import threading
from uuid import uuid4
import inspect
from functools import wraps

# Import request context
from app.middlewares.request_id import get_request_id

# Configure audit logging
logger = logging.getLogger("audit")

# Environment settings
AUDIT_LOG_ENABLED = os.environ.get("AUDIT_LOG_ENABLED", "true").lower() == "true"
AUDIT_LOG_FILE = os.environ.get("AUDIT_LOG_FILE", "logs/audit.log")
AUDIT_LOG_LEVEL = os.environ.get("AUDIT_LOG_LEVEL", "INFO")

# Setup audit logger
def setup_audit_logger():
    """Configure audit logger with appropriate handlers"""
    if not AUDIT_LOG_ENABLED:
        return
        
    # Create directory for logs if it doesn't exist
    log_dir = os.path.dirname(AUDIT_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure audit logger
    audit_handler = logging.FileHandler(AUDIT_LOG_FILE)
    audit_handler.setLevel(getattr(logging, AUDIT_LOG_LEVEL))
    
    # Format audit logs as JSON
    formatter = logging.Formatter('%(message)s')
    audit_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(audit_handler)
    logger.setLevel(getattr(logging, AUDIT_LOG_LEVEL))
    
    # Set propagate to False to prevent audit logs from showing in main app logs
    logger.propagate = False
    
    logging.info(f"Audit logging configured to {AUDIT_LOG_FILE}")

# Thread-local storage for audit context
_audit_context = threading.local()

def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current user from context if available"""
    try:
        # This should be implemented according to your auth system
        # For example, using a request dependency to extract user from token
        from app.core.security import get_current_active_user
        return get_current_active_user()
    except:
        return None

class AuditLogger:
    """Audit logger for security-sensitive operations"""
    
    @staticmethod
    def set_context(user_id: str = None, session_id: str = None, 
                    operation_id: str = None, **kwargs):
        """Set context values for audit logging"""
        if not hasattr(_audit_context, "context"):
            _audit_context.context = {}
            
        if user_id:
            _audit_context.context["user_id"] = user_id
        if session_id:
            _audit_context.context["session_id"] = session_id
        if operation_id:
            _audit_context.context["operation_id"] = operation_id
            
        # Add any other context values
        for key, value in kwargs.items():
            _audit_context.context[key] = value
    
    @staticmethod
    def clear_context():
        """Clear audit context"""
        if hasattr(_audit_context, "context"):
            delattr(_audit_context, "context")
    
    @staticmethod
    def get_context() -> Dict[str, Any]:
        """Get current audit context"""
        if not hasattr(_audit_context, "context"):
            _audit_context.context = {}
        return _audit_context.context
    
    @staticmethod
    def log(action: str, resource_type: str = None, resource_id: str = None, 
            status: str = "success", details: Dict[str, Any] = None, 
            user_id: str = None, ip_address: str = None):
        """
        Log an audit event
        
        Args:
            action: The action being performed (e.g., "login", "access", "update")
            resource_type: Type of resource being accessed (e.g., "user", "satellite")
            resource_id: ID of the resource being accessed
            status: Outcome of the action (success, failure, error)
            details: Additional context-specific details
            user_id: ID of the user performing the action (if not in context)
            ip_address: IP address of the client (if not in context)
        """
        if not AUDIT_LOG_ENABLED:
            return
            
        # Get current context
        context = AuditLogger.get_context()
        
        # Create audit event
        audit_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "status": status,
            "request_id": get_request_id()
        }
        
        # Add user information - prefer parameter over context
        audit_event["user_id"] = user_id or context.get("user_id", "anonymous")
        
        # Add resource information if provided
        if resource_type:
            audit_event["resource_type"] = resource_type
        if resource_id:
            audit_event["resource_id"] = resource_id
            
        # Add IP address if available
        if ip_address:
            audit_event["ip_address"] = ip_address
        elif "ip_address" in context:
            audit_event["ip_address"] = context["ip_address"]
            
        # Add operation ID for correlation
        if "operation_id" in context:
            audit_event["operation_id"] = context["operation_id"]
        else:
            audit_event["operation_id"] = str(uuid4())
            
        # Add additional context
        if "session_id" in context:
            audit_event["session_id"] = context["session_id"]
            
        # Add details if provided
        if details:
            audit_event["details"] = details
            
        # Add source code information for traceability
        caller_frame = inspect.currentframe().f_back
        if caller_frame:
            frame_info = inspect.getframeinfo(caller_frame)
            audit_event["source"] = {
                "file": os.path.basename(frame_info.filename),
                "line": frame_info.lineno,
                "function": frame_info.function
            }
            
        # Log the audit event as JSON
        try:
            logger.info(json.dumps(audit_event))
        except Exception as e:
            logging.error(f"Failed to log audit event: {str(e)}")

def audit_log(action: str, resource_type: str = None):
    """
    Decorator for auditing sensitive operations
    
    Args:
        action: The action being performed
        resource_type: Type of resource being accessed
        
    Example:
        @audit_log("access", "satellite_data")
        def get_satellite_data(satellite_id):
            # Function implementation
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract resource_id from args or kwargs if possible
            resource_id = None
            
            # Try to get resource_id from function parameters
            param_names = inspect.signature(func).parameters.keys()
            param_list = list(param_names)
            
            # Check common parameter names for resource ID
            id_param_names = ["id", "resource_id", f"{resource_type}_id"]
            for name in id_param_names:
                if name in kwargs:
                    resource_id = kwargs[name]
                    break
                elif name in param_list and len(args) > param_list.index(name):
                    resource_id = args[param_list.index(name)]
                    break
            
            # Set operation ID for correlation
            operation_id = str(uuid4())
            AuditLogger.set_context(operation_id=operation_id)
            
            # Log before execution
            AuditLogger.log(
                action=f"{action}_attempt", 
                resource_type=resource_type,
                resource_id=str(resource_id) if resource_id else None,
                status="pending"
            )
            
            try:
                # Call the original function
                result = func(*args, **kwargs)
                
                # Log successful execution
                AuditLogger.log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id) if resource_id else None,
                    status="success"
                )
                
                return result
                
            except Exception as e:
                # Log execution failure
                AuditLogger.log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id) if resource_id else None,
                    status="failure",
                    details={"error": str(e), "error_type": type(e).__name__}
                )
                raise
                
        return wrapper
    return decorator

# Initialize audit logger
setup_audit_logger() 