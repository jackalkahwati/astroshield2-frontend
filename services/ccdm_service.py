"""CCDM Service with enhanced ML capabilities."""
from typing import Dict, Any, List, Optional, Union, Callable, Awaitable, TypeVar
from datetime import datetime, timedelta
import logging
import random
import time
import asyncio
import contextlib
import json
import socket
import os
import uuid
import threading
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
from functools import wraps, lru_cache
from collections import defaultdict
from analysis.ml_evaluators import MLManeuverEvaluator, MLSignatureEvaluator, MLAMREvaluator

# Configure enhanced structured logging
logger = logging.getLogger(__name__)

# Environment and configuration loading
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV", "development")
CONFIG_DIR = os.getenv("CONFIG_DIR", "./config")

# Load environment-specific configuration
def load_config():
    """
    Load configuration based on current environment.
    
    Returns:
        Dictionary with configuration values
    """
    # Default configuration
    default_config = {
        "service": {
            "name": "ccdm_service",
            "version": "1.0.0"
        },
        "logging": {
            "level": "INFO",
            "structured": True
        },
        "database": {
            "timeout": 30,
            "pool_size": 10,
            "max_overflow": 20
        },
        "rate_limiting": {
            "enabled": True,
            "default_limit": 100,
            "endpoints": {
                "get_historical_analysis": 30,
                "analyze_conjunction": 60,
                "get_assessment": 120
            }
        },
        "caching": {
            "enabled": True,
            "default_ttl": 300
        },
        "alerting": {
            "enabled": False,
            "email": {
                "enabled": False,
                "smtp_server": "",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_address": "",
                "to_addresses": []
            },
            "slack": {
                "enabled": False,
                "webhook_url": ""
            }
        }
    }
    
    # Try to load environment-specific config
    config_path = os.path.join(CONFIG_DIR, f"{DEPLOYMENT_ENV}.yaml")
    env_config = {}
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                env_config = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {str(e)}")
    
    # Merge configurations with environment config taking precedence
    def deep_merge(base, override):
        result = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    config = deep_merge(default_config, env_config)
    
    # Override with environment variables
    if os.getenv("DB_TIMEOUT_SECONDS"):
        config["database"]["timeout"] = int(os.getenv("DB_TIMEOUT_SECONDS"))
    
    if os.getenv("RATE_LIMIT_ENABLED"):
        config["rate_limiting"]["enabled"] = os.getenv("RATE_LIMIT_ENABLED").lower() in ("true", "1", "yes")
    
    if os.getenv("CACHE_ENABLED"):
        config["caching"]["enabled"] = os.getenv("CACHE_ENABLED").lower() in ("true", "1", "yes")
    
    if os.getenv("ALERTING_ENABLED"):
        config["alerting"]["enabled"] = os.getenv("ALERTING_ENABLED").lower() in ("true", "1", "yes")
    
    return config

# Load configuration
CONFIG = load_config()

# Host information for logging context
HOSTNAME = socket.gethostname()
SERVICE_NAME = CONFIG["service"]["name"]
SERVICE_VERSION = CONFIG["service"]["version"]

# Rate limiting configuration
RATE_LIMIT_ENABLED = CONFIG["rate_limiting"]["enabled"]
DEFAULT_RATE_LIMITS = CONFIG["rate_limiting"]["endpoints"]
DEFAULT_RATE_LIMITS["default"] = CONFIG["rate_limiting"]["default_limit"]

# Cache configuration
CACHE_ENABLED = CONFIG["caching"]["enabled"]
DEFAULT_CACHE_TTL = CONFIG["caching"]["default_ttl"]

# Alerting system
class AlertManager:
    """Alert system for critical errors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config["alerting"]["enabled"]
        self.email_config = config["alerting"]["email"]
        self.slack_config = config["alerting"]["slack"]
        
        # Maintain a set of recent alert IDs to prevent duplicates
        self.recent_alerts = set()
        self.lock = threading.RLock()
        
        # Start alert cleaner thread
        if self.enabled:
            self._start_alert_cleaner()
            
    def _start_alert_cleaner(self):
        """Start background thread to clean old alerts."""
        def clean_alerts():
            while True:
                time.sleep(3600)  # Run every hour
                with self.lock:
                    self.recent_alerts.clear()
                    
        thread = threading.Thread(target=clean_alerts, daemon=True)
        thread.start()
        
    def _is_duplicate(self, alert_id: str) -> bool:
        """Check if alert is a duplicate."""
        with self.lock:
            if alert_id in self.recent_alerts:
                return True
            self.recent_alerts.add(alert_id)
            return False
            
    def send_alert(self, level: str, title: str, message: str, context: Dict[str, Any] = None):
        """
        Send alert via configured channels.
        
        Args:
            level: Alert level (critical, error, warning)
            title: Alert title
            message: Alert message
            context: Additional context for the alert
        """
        if not self.enabled:
            return
            
        # Only alert for critical and error levels
        if level.lower() not in ("critical", "error"):
            return
            
        # Create alert ID to prevent duplicates
        if context:
            alert_id = f"{level}:{title}:{context.get('error_type', '')}"
        else:
            alert_id = f"{level}:{title}"
            
        if self._is_duplicate(alert_id):
            return
            
        # Prepare alert data
        alert_data = {
            "level": level,
            "title": title,
            "message": message,
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "environment": DEPLOYMENT_ENV,
            "hostname": HOSTNAME,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if context:
            alert_data["context"] = context
            
        # Send via configured channels
        if self.email_config["enabled"]:
            self._send_email_alert(alert_data)
            
        if self.slack_config["enabled"]:
            self._send_slack_alert(alert_data)
            
    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send alert via email."""
        try:
            config = self.email_config
            
            # Create message
            msg = MIMEMultipart()
            msg['Subject'] = f"[{alert_data['level'].upper()}] {alert_data['service']} - {alert_data['title']}"
            msg['From'] = config['from_address']
            msg['To'] = ", ".join(config['to_addresses'])
            
            # Create message body
            body = f"""
            <html>
            <body>
                <h2>{alert_data['title']}</h2>
                <p><strong>Level:</strong> {alert_data['level']}</p>
                <p><strong>Service:</strong> {alert_data['service']} v{alert_data['version']}</p>
                <p><strong>Environment:</strong> {alert_data['environment']}</p>
                <p><strong>Host:</strong> {alert_data['hostname']}</p>
                <p><strong>Time:</strong> {alert_data['timestamp']}</p>
                <h3>Message:</h3>
                <p>{alert_data['message']}</p>
            """
            
            if 'context' in alert_data:
                body += "<h3>Context:</h3><pre>" + json.dumps(alert_data['context'], indent=2) + "</pre>"
                
            body += """
            </body>
            </html>
            """
            
            # Attach body
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                if config['username'] and config['password']:
                    server.login(config['username'], config['password'])
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            
    def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Send alert via Slack webhook."""
        try:
            webhook_url = self.slack_config['webhook_url']
            if not webhook_url:
                return
                
            # Create color based on level
            color = "#ff0000" if alert_data['level'].lower() == "critical" else "#ffa500"
            
            # Create attachment
            attachment = {
                "fallback": f"{alert_data['level'].upper()}: {alert_data['title']}",
                "color": color,
                "title": f"{alert_data['level'].upper()}: {alert_data['title']}",
                "text": alert_data['message'],
                "fields": [
                    {
                        "title": "Service",
                        "value": f"{alert_data['service']} v{alert_data['version']}",
                        "short": True
                    },
                    {
                        "title": "Environment",
                        "value": alert_data['environment'],
                        "short": True
                    },
                    {
                        "title": "Host",
                        "value": alert_data['hostname'],
                        "short": True
                    },
                    {
                        "title": "Time",
                        "value": alert_data['timestamp'],
                        "short": True
                    }
                ],
                "footer": "CCDM Service Alerts"
            }
            
            # Add context if available
            if 'context' in alert_data:
                context_text = json.dumps(alert_data['context'], indent=2)
                attachment["fields"].append({
                    "title": "Context",
                    "value": f"```{context_text}```",
                    "short": False
                })
                
            # Create payload
            payload = {
                "text": f"Alert from {alert_data['service']} ({alert_data['environment']})",
                "attachments": [attachment]
            }
            
            # Send to Slack
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to send Slack alert: {response.status_code} {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")

# Initialize alert manager
alert_manager = AlertManager(CONFIG)

# Enhanced structured logger with alerts
class StructuredLogger:
    """Enhanced structured logger with consistent context and alerting."""
    
    @staticmethod
    def get_base_context():
        """Get base context for all logs."""
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "hostname": HOSTNAME,
            "environment": DEPLOYMENT_ENV
        }
    
    @classmethod
    def info(cls, message: str, **kwargs):
        """Log info message with structured context."""
        context = cls.get_base_context()
        context.update(kwargs)
        
        # Create structured log
        log_entry = {
            "message": message,
            "level": "INFO",
            "timestamp": datetime.utcnow().isoformat(),
            "context": context
        }
        
        logger.info(json.dumps(log_entry))
        
    @classmethod
    def warning(cls, message: str, **kwargs):
        """Log warning message with structured context."""
        context = cls.get_base_context()
        context.update(kwargs)
        
        # Create structured log
        log_entry = {
            "message": message,
            "level": "WARNING",
            "timestamp": datetime.utcnow().isoformat(),
            "context": context
        }
        
        logger.warning(json.dumps(log_entry))
        
    @classmethod
    def error(cls, message: str, error=None, alert=False, **kwargs):
        """Log error message with structured context and optional alert."""
        context = cls.get_base_context()
        context.update(kwargs)
        
        # Add error details if provided
        if error:
            context["error"] = {
                "type": type(error).__name__,
                "message": str(error)
            }
            
        # Create structured log
        log_entry = {
            "message": message,
            "level": "ERROR",
            "timestamp": datetime.utcnow().isoformat(),
            "context": context
        }
        
        logger.error(json.dumps(log_entry))
        
        # Send alert if requested
        if alert:
            alert_manager.send_alert(
                "error",
                message,
                str(error) if error else message,
                context
            )
        
    @classmethod
    def critical(cls, message: str, error=None, **kwargs):
        """Log critical error with structured context and automatic alert."""
        context = cls.get_base_context()
        context.update(kwargs)
        
        # Add error details if provided
        if error:
            context["error"] = {
                "type": type(error).__name__,
                "message": str(error)
            }
            
        # Create structured log
        log_entry = {
            "message": message,
            "level": "CRITICAL",
            "timestamp": datetime.utcnow().isoformat(),
            "context": context
        }
        
        logger.critical(json.dumps(log_entry))
        
        # Always send alerts for critical errors
        alert_manager.send_alert(
            "critical",
            message,
            str(error) if error else message,
            context
        )
        
    @classmethod
    def exception(cls, message: str, alert=True, **kwargs):
        """Log exception message with structured context, traceback, and alert."""
        context = cls.get_base_context()
        context.update(kwargs)
        
        # Create structured log
        log_entry = {
            "message": message,
            "level": "ERROR",
            "timestamp": datetime.utcnow().isoformat(),
            "context": context
        }
        
        logger.exception(json.dumps(log_entry))
        
        # Send alert for exceptions if enabled
        if alert:
            alert_manager.send_alert(
                "error",
                f"Exception: {message}",
                "See logs for stack trace",
                context
            )

# Rate limiter class
class RateLimiter:
    """Rate limiter to prevent API abuse."""
    
    def __init__(self):
        # Store request counts per user/IP and endpoint
        self.request_counts = defaultdict(lambda: defaultdict(list))
        self.rate_limits = DEFAULT_RATE_LIMITS.copy()
        self.lock = threading.RLock()
        
    def _clean_old_requests(self, user_id: str, endpoint: str):
        """Remove requests older than 1 minute."""
        with self.lock:
            now = time.time()
            # Keep only requests within the last minute
            self.request_counts[user_id][endpoint] = [
                timestamp for timestamp in self.request_counts[user_id][endpoint]
                if now - timestamp < 60
            ]
        
    def check_rate_limit(self, user_id: str, endpoint: str) -> bool:
        """
        Check if a request is allowed under rate limits.
        
        Args:
            user_id: User ID or IP address
            endpoint: API endpoint being accessed
            
        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        if not RATE_LIMIT_ENABLED:
            return True
            
        # Clean up old requests
        self._clean_old_requests(user_id, endpoint)
        
        # Get rate limit for endpoint
        rate_limit = self.rate_limits.get(endpoint, self.rate_limits["default"])
        
        with self.lock:
            # Check if user has exceeded rate limit
            if len(self.request_counts[user_id][endpoint]) >= rate_limit:
                return False
                
            # Record request
            self.request_counts[user_id][endpoint].append(time.time())
            return True
            
    def get_remaining_requests(self, user_id: str, endpoint: str) -> int:
        """Get remaining requests allowed for user and endpoint."""
        if not RATE_LIMIT_ENABLED:
            return 999
            
        # Clean up old requests
        self._clean_old_requests(user_id, endpoint)
        
        # Get rate limit for endpoint
        rate_limit = self.rate_limits.get(endpoint, self.rate_limits["default"])
        
        with self.lock:
            return max(0, rate_limit - len(self.request_counts[user_id][endpoint]))
            
    def update_rate_limit(self, endpoint: str, new_limit: int):
        """Update rate limit for an endpoint."""
        with self.lock:
            self.rate_limits[endpoint] = new_limit

# Singleton rate limiter instance
_rate_limiter = RateLimiter()

# Cache decorator with TTL
def cached(ttl_seconds: int = DEFAULT_CACHE_TTL):
    """
    Decorator to cache function results with time-to-live.
    
    Args:
        ttl_seconds: TTL in seconds for cached items
        
    Returns:
        Decorated function with caching
    """
    def decorator(func):
        # Cache to store results and timestamps
        cache = {}
        cache_lock = threading.RLock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not CACHE_ENABLED:
                return func(*args, **kwargs)
                
            # Create cache key from function name, args, and kwargs
            # Skip first arg (self) for instance methods
            cache_args = args[1:] if args and hasattr(args[0], '__dict__') else args
            key = f"{func.__name__}:{str(cache_args)}:{str(sorted(kwargs.items()))}"
            
            with cache_lock:
                # Check if result is in cache and not expired
                if key in cache:
                    result, timestamp = cache[key]
                    if time.time() - timestamp < ttl_seconds:
                        StructuredLogger.info(
                            f"Cache hit for {func.__name__}",
                            operation=func.__name__,
                            ttl_seconds=ttl_seconds
                        )
                        return result
                
                # Get fresh result
                result = func(*args, **kwargs)
                
                # Cache result with timestamp
                cache[key] = (result, time.time())
                
                # Clean up old cache entries if more than 1000 items
                if len(cache) > 1000:
                    # Remove oldest 20% of entries
                    oldest = sorted(
                        [(k, v[1]) for k, v in cache.items()],
                        key=lambda x: x[1]
                    )[:200]
                    for k, _ in oldest:
                        del cache[k]
                
                return result
                
        # Add method to clear cache
        def clear_cache():
            with cache_lock:
                cache.clear()
                
        wrapper.clear_cache = clear_cache
        return wrapper
        
    return decorator

# Singleton instance
_ccdm_service_instance = None

# Define result type for error handling decorator
T = TypeVar('T')

# Error codes
class ErrorCode:
    """Standard error codes for CCDM service."""
    INVALID_INPUT = "INVALID_INPUT"
    DATABASE_ERROR = "DATABASE_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    NOT_FOUND = "NOT_FOUND"
    SECURITY_ERROR = "SECURITY_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    
# Standardized error response
def error_response(code: str, message: str, object_id: str = None, request_id: str = None) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        code: Error code from ErrorCode class
        message: Error message
        object_id: Optional object ID
        request_id: Optional request ID
    
    Returns:
        Dictionary with error details
    """
    response = {
        "status": "error",
        "error_code": code,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if object_id:
        response["object_id"] = object_id
        
    if request_id:
        response["request_id"] = request_id
        
    return response

# Timeout context manager
@contextlib.contextmanager
def timeout(seconds: int, error_message: str = "Operation timed out"):
    """
    Context manager for operation timeout.
    
    Args:
        seconds: Timeout in seconds
        error_message: Error message for timeout exception
    
    Raises:
        TimeoutError: If operation exceeds timeout
    """
    def handle_timeout(signum, frame):
        raise TimeoutError(error_message)
        
    # Use signal for timeout in synchronous code
    import signal
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm

# Decorator for standardized error handling
def handle_errors(func):
    """
    Decorator for standardized error handling with structured logging.
    
    Args:
        func: Function to decorate
    
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate request ID and extract object ID
        request_id = str(uuid.uuid4())
        object_id = None
        
        # Try to get object_id from kwargs or args
        if 'norad_id' in kwargs:
            object_id = kwargs['norad_id']
        elif 'spacecraft_id' in kwargs:
            object_id = kwargs['spacecraft_id']
        elif 'object_id' in kwargs:
            object_id = kwargs['object_id']
        elif len(args) > 1 and isinstance(args[1], str):
            object_id = args[1]  # Assume second arg is ID in most methods
            
        # Log request start with context
        StructuredLogger.info(
            f"Starting {func.__name__}",
            request_id=request_id,
            operation=func.__name__,
            object_id=object_id,
            start_time=datetime.utcnow().isoformat()
        )
            
        try:
            with timeout(300, "Operation timed out"):  # 5-minute timeout
                result = func(*args, **kwargs)
                
                # Log successful completion
                StructuredLogger.info(
                    f"Completed {func.__name__}",
                    request_id=request_id,
                    operation=func.__name__,
                    object_id=object_id,
                    duration_ms=int((datetime.utcnow() - datetime.fromisoformat(StructuredLogger.get_base_context().get("start_time", datetime.utcnow().isoformat()))).total_seconds() * 1000),
                    status="success"
                )
                
                return result
        except ValueError as e:
            StructuredLogger.warning(
                f"Validation error in {func.__name__}",
                request_id=request_id,
                operation=func.__name__,
                object_id=object_id,
                error=e,
                status="failed"
            )
            return error_response(ErrorCode.INVALID_INPUT, str(e), object_id, request_id)
        except TimeoutError as e:
            StructuredLogger.error(
                f"Timeout in {func.__name__}",
                request_id=request_id,
                operation=func.__name__,
                object_id=object_id,
                error=e,
                status="timeout"
            )
            return error_response(ErrorCode.TIMEOUT_ERROR, str(e), object_id, request_id)
        except Exception as e:
            StructuredLogger.exception(
                f"Error in {func.__name__}",
                request_id=request_id,
                operation=func.__name__,
                object_id=object_id,
                error_type=type(e).__name__,
                status="error"
            )
            return error_response(ErrorCode.PROCESSING_ERROR, str(e), object_id, request_id)
            
    return wrapper

# Async version of the error handling decorator
def handle_errors_async(func):
    """
    Decorator for standardized error handling in async functions.
    
    Args:
        func: Async function to decorate
    
    Returns:
        Wrapped async function with error handling
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract object_id from arguments if available
        object_id = None
        request_id = f"req_{int(time.time())}"
        
        # Try to get object_id from kwargs or args
        if 'norad_id' in kwargs:
            object_id = kwargs['norad_id']
        elif 'spacecraft_id' in kwargs:
            object_id = kwargs['spacecraft_id']
        elif 'object_id' in kwargs:
            object_id = kwargs['object_id']
        elif len(args) > 1 and isinstance(args[1], str):
            object_id = args[1]  # Assume second arg is ID in most methods
            
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=300)  # 5-minute timeout
        except ValueError as e:
            logger.warning(f"[{request_id}] Validation error in {func.__name__}: {str(e)}")
            return error_response(ErrorCode.INVALID_INPUT, str(e), object_id, request_id)
        except asyncio.TimeoutError:
            logger.error(f"[{request_id}] Async timeout in {func.__name__}")
            return error_response(ErrorCode.TIMEOUT_ERROR, "Operation timed out", object_id, request_id)
        except Exception as e:
            logger.exception(f"[{request_id}] Error in {func.__name__}: {str(e)}")
            return error_response(ErrorCode.PROCESSING_ERROR, str(e), object_id, request_id)
            
    return wrapper

class CCDMService:
    def __init__(self, db=None):
        self.maneuver_evaluator = MLManeuverEvaluator()
        self.signature_evaluator = MLSignatureEvaluator()
        self.amr_evaluator = MLAMREvaluator()
        self.db = db  # SQLAlchemy session
        
        # Database configuration
        self.db_timeout = int(os.getenv("DB_TIMEOUT_SECONDS", "30"))
        
        # Load rate limiter
        self.rate_limiter = _rate_limiter
        
    @classmethod
    def get_instance(cls, db=None):
        """
        Get singleton instance of CCDMService.
        
        Args:
            db: Optional database session
            
        Returns:
            CCDMService instance
        """
        global _ccdm_service_instance
        
        if _ccdm_service_instance is None:
            logger.info("Initializing new CCDMService instance")
            _ccdm_service_instance = cls(db=db)
        elif db is not None:
            logger.debug("Updating database session in existing CCDMService instance")
            _ccdm_service_instance.db = db
            
        return _ccdm_service_instance

    @handle_errors
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

    # Apply rate limiting decorator
    def check_rate_limit(self, user_id: str, endpoint: str) -> Dict[str, Any]:
        """
        Check if request is allowed under rate limits.
        
        Args:
            user_id: User ID or IP address
            endpoint: API endpoint being accessed
            
        Returns:
            Dictionary with rate limit check result
        """
        allowed = self.rate_limiter.check_rate_limit(user_id, endpoint)
        remaining = self.rate_limiter.get_remaining_requests(user_id, endpoint)
        
        result = {
            "allowed": allowed,
            "remaining": remaining
        }
        
        # If rate limit exceeded, add retry-after header suggestion
        if not allowed:
            result["retry_after"] = 60  # Try again after 1 minute
            StructuredLogger.warning(
                f"Rate limit exceeded",
                user_id=user_id,
                endpoint=endpoint
            )
            
        return result
        
    # Add cache to relevant methods
    @cached(ttl_seconds=300)  # 5-minute cache
    def get_historical_analysis(self, norad_id: str, start_date: str, end_date: str, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        """
        Get historical analysis for a spacecraft.
        
        Args:
            norad_id: NORAD ID of the spacecraft
            start_date: Analysis start date (ISO format)
            end_date: Analysis end date (ISO format)
            page: Page number (starting from 1)
            page_size: Number of items per page
            
        Returns:
            Dict with historical analysis results
        """
        try:
            # Input validation
            if not self._is_valid_norad_id(norad_id):
                return error_response(
                    ErrorCode.INVALID_INPUT,
                    f"Invalid NORAD ID: {self._sanitize_input(norad_id)}"
                )
            
            try:
                start = datetime.fromisoformat(start_date)
                end = datetime.fromisoformat(end_date)
            except ValueError:
                return error_response(
                    ErrorCode.INVALID_INPUT,
                    "Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)."
                )
                
            # Validate analysis window
            valid, error_msg = validate_analysis_window(start_date, end_date)
            if not valid:
                return error_response(ErrorCode.INVALID_INPUT, error_msg)
                
            # Check if data exists in the database
            db_data = self._get_historical_analysis_from_db(norad_id, start, end)
            
            # If data exists in DB and covers the entire period, use it
            if db_data:
                logger.info(f"Retrieved historical analysis for {norad_id} from database",
                           extra={"norad_id": norad_id})
                
                # Apply pagination to the data
                total_items = len(db_data)
                total_pages = (total_items + page_size - 1) // page_size  # Ceiling division
                
                # Validate page number
                if page < 1:
                    page = 1
                elif page > total_pages and total_pages > 0:
                    page = total_pages
                    
                # Calculate slice indices
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, total_items)
                
                # Get paginated data
                paginated_data = db_data[start_idx:end_idx]
                
                # Construct response with pagination metadata
                response = {
                    "norad_id": norad_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "total_items": total_items,
                        "total_pages": total_pages
                    },
                    "data_points": paginated_data
                }
                
                # Add trend summary if we have enough data
                if total_items >= 3:
                    response["trend_summary"] = self._calculate_trend_summary(db_data)
                    
                return response
                
            # Generate new data if not in database
            logger.info(f"Generating historical analysis for {norad_id}",
                      extra={"norad_id": norad_id})
                      
            # Generate data points in batches to avoid memory issues with large date ranges
            all_data_points = self._generate_historical_data_points_optimized(norad_id, start, end)
            
            # Store in database asynchronously to avoid blocking the response
            threading.Thread(
                target=self._store_historical_analysis,
                args=(norad_id, all_data_points)
            ).start()
            
            # Apply pagination
            total_items = len(all_data_points)
            total_pages = (total_items + page_size - 1) // page_size
            
            # Validate page number
            if page < 1:
                page = 1
            elif page > total_pages and total_pages > 0:
                page = total_pages
                
            # Calculate slice indices
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_items)
            
            # Get paginated data
            paginated_data = all_data_points[start_idx:end_idx]
            
            # Construct response
            response = {
                "norad_id": norad_id,
                "start_date": start_date,
                "end_date": end_date,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_items": total_items,
                    "total_pages": total_pages
                },
                "data_points": paginated_data
            }
            
            # Add trend summary if we have enough data
            if total_items >= 3:
                response["trend_summary"] = self._calculate_trend_summary(all_data_points)
                
            return response
            
        except Exception as e:
            logger.exception(f"Error generating historical analysis: {str(e)}",
                           extra={"norad_id": norad_id})
            return error_response(
                ErrorCode.PROCESSING_ERROR,
                "Failed to generate historical analysis. Try again or contact support.",
                object_id=norad_id
            )

    def _generate_historical_data_points_optimized(self, norad_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Optimized version of historical data point generation.
        Processes data in chunks to reduce memory usage.
        
        Args:
            norad_id: NORAD ID of the spacecraft
            start_date: Start date
            end_date: End date
            
        Returns:
            List of historical data points
        """
        # Calculate total number of days
        total_days = (end_date - start_date).days + 1
        
        # If the period is large, process in chunks
        MAX_DAYS_PER_CHUNK = 30
        
        if total_days <= MAX_DAYS_PER_CHUNK:
            # For small ranges, process normally
            return self._generate_historical_data_points_batch(norad_id, start_date, end_date)
        
        # For large ranges, process in chunks
        all_data_points = []
        current_start = start_date
        
        while current_start <= end_date:
            # Calculate end of current chunk
            current_end = min(current_start + timedelta(days=MAX_DAYS_PER_CHUNK - 1), end_date)
            
            # Process chunk
            logger.info(f"Processing chunk from {current_start} to {current_end} for {norad_id}")
            chunk_data = self._generate_historical_data_points_batch(norad_id, current_start, current_end)
            all_data_points.extend(chunk_data)
            
            # Move to next chunk
            current_start = current_end + timedelta(days=1)
        
        return all_data_points

    def _generate_historical_data_points_batch(self, norad_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Generate historical data points for a specific date range.
        Optimized version that processes smaller batches.
        
        Args:
            norad_id: NORAD ID of the spacecraft
            start_date: Start date
            end_date: End date
            
        Returns:
            List of historical data points
        """
        data_points = []
        current_date = start_date
        
        # Create a reusable random number generator for consistency
        rng = random.Random(int(norad_id) if norad_id.isdigit() else hash(norad_id))
        
        # Precompute threat level probabilities based on norad_id
        # This creates a consistent pattern for each spacecraft
        threat_probs = {
            "none": 0.5 + (rng.random() * 0.2),
            "low": 0.2 + (rng.random() * 0.1),
            "medium": 0.05 + (rng.random() * 0.1),
            "high": 0.01 + (rng.random() * 0.04)
        }
        
        # Normalize probabilities to sum to 1
        total_prob = sum(threat_probs.values())
        for level in threat_probs:
            threat_probs[level] /= total_prob
        
        # Generate data points - one per day
        while current_date <= end_date:
            # Determine threat level based on weighted probabilities
            rand_val = rng.random()
            cumulative = 0
            threat_level = "none"  # Default
            
            for level, prob in threat_probs.items():
                cumulative += prob
                if rand_val <= cumulative:
                    threat_level = level
                    break
            
            # Generate analysis details based on threat level
            details = self._generate_details_for_threat_level(threat_level)
            
            # Create data point
            data_point = {
                "date": current_date.isoformat(),
                "threat_level": threat_level,
                "details": details
            }
            
            data_points.append(data_point)
            current_date += timedelta(days=1)
        
        return data_points

    def _store_historical_analysis(self, norad_id: str, data_points: List[Dict[str, Any]]) -> None:
        """
        Store historical analysis data in the database.
        
        Args:
            norad_id: NORAD ID of the spacecraft
            data_points: List of data points to store
        """
        try:
            # Batch inserts for better performance
            BATCH_SIZE = 100
            for i in range(0, len(data_points), BATCH_SIZE):
                batch = data_points[i:i+BATCH_SIZE]
                with self.db.session() as session:
                    for point in batch:
                        analysis = HistoricalAnalysis(
                            norad_id=norad_id,
                            analysis_date=datetime.fromisoformat(point["date"]),
                            threat_level=point["threat_level"],
                            analysis_type="historical",
                            data=point["details"]
                        )
                        session.add(analysis)
                    session.commit()
                    
            logger.info(f"Stored {len(data_points)} historical data points for {norad_id}",
                       extra={"norad_id": norad_id})
        except Exception as e:
            logger.exception(f"Error storing historical analysis: {str(e)}",
                           extra={"norad_id": norad_id})

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

    # Apply rate limiting to secure method
    def get_historical_analysis_secure(self, auth_token: str, norad_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Secure version of get_historical_analysis with authentication and rate limiting.
        
        Args:
            auth_token: Authentication token
            norad_id: The NORAD ID of the spacecraft
            start_date: The start date in ISO format
            end_date: The end date in ISO format
            
        Returns:
            Dictionary containing historical analysis data or error
        """
        request_id = str(uuid.uuid4())
        
        try:
            # Authenticate request
            auth_info = self.authenticate_request(auth_token)
            user_id = auth_info.get("user_id", "anonymous")
            
            # Check rate limit
            rate_limit_check = self.check_rate_limit(user_id, "get_historical_analysis")
            if not rate_limit_check["allowed"]:
                return error_response(
                    "RATE_LIMIT_EXCEEDED", 
                    f"Rate limit exceeded. Try again after {rate_limit_check['retry_after']} seconds.",
                    norad_id, 
                    request_id
                )
            
            # Authorize action
            if not self.authorize_action(auth_info, "read"):
                StructuredLogger.warning(
                    "Unauthorized access attempt", 
                    request_id=request_id,
                    user_id=user_id,
                    norad_id=norad_id
                )
                return error_response(
                    "UNAUTHORIZED", 
                    "You do not have permission to access this data", 
                    norad_id, 
                    request_id
                )
                
            # Log successful authentication
            StructuredLogger.info(
                "Authorized historical data access", 
                request_id=request_id,
                user_id=user_id,
                norad_id=norad_id
            )
                
            # Call the actual method
            result = self.get_historical_analysis(norad_id, start_date, end_date)
            
            # Add audit information to the result
            if isinstance(result, dict) and result.get("status") != "error":
                result["audit"] = {
                    "user_id": user_id,
                    "access_time": datetime.utcnow().isoformat(),
                    "request_id": request_id
                }
                
                # Add rate limit information
                result["rate_limit"] = {
                    "remaining": rate_limit_check["remaining"]
                }
                
            return result
            
        except ValueError as auth_error:
            # Handle authentication errors
            StructuredLogger.warning(
                "Authentication failed for historical data access", 
                request_id=request_id,
                norad_id=norad_id,
                error=auth_error
            )
            
            return error_response(
                "AUTHENTICATION_FAILED", 
                str(auth_error), 
                norad_id, 
                request_id
            )
            
    # Also add caching to assessment method
    @cached(ttl_seconds=180)  # 3-minute cache
    def get_assessment(self, object_id: str) -> Dict[str, Any]:
        """Get a comprehensive CCDM assessment for a specific object."""
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
        """
        Get historical analysis data from the database.
        Optimized for performance with proper query construction and result handling.
        
        Args:
            norad_id: NORAD ID of the spacecraft
            start_date: Start date for the analysis
            end_date: End date for the analysis
        
        Returns:
            List of historical analysis data points
        """
        if not self.db:
            logger.info(f"No database connection available for historical analysis of {norad_id}")
            return []
        
        start_time = datetime.utcnow()
        logger.info(f"Retrieving historical analysis for {norad_id} from {start_date} to {end_date}",
                   extra={"norad_id": norad_id})
        
        try:
            # Use session context manager to ensure proper session handling
            with self.db.session() as session:
                # Use optimized query with explicit join and column selection
                # This reduces data transfer and processing time
                query = (
                    session.query(
                        HistoricalAnalysis.norad_id,
                        HistoricalAnalysis.analysis_date,
                        HistoricalAnalysis.threat_level,
                        HistoricalAnalysis.data
                    )
                    .filter(
                        HistoricalAnalysis.norad_id == norad_id,
                        HistoricalAnalysis.analysis_date >= start_date,
                        HistoricalAnalysis.analysis_date <= end_date
                    )
                    # Use the composite index we created for this query pattern
                    .order_by(HistoricalAnalysis.analysis_date)
                    # Set timeout to prevent long-running queries
                    .execution_options(timeout=CONFIG["database"]["timeout"])
                )
                
                # Execute with custom timeout handler
                with timeout(seconds=CONFIG["database"]["timeout"]):
                    # Use chunks for large result sets to reduce memory usage
                    # Set a reasonable chunk size based on expected data volume
                    CHUNK_SIZE = 500
                    results = []
                    
                    # Stream results in chunks
                    for chunk in self._chunked_query_results(query, CHUNK_SIZE):
                        results.extend(chunk)
                    
                    # Convert to the format expected by the API
                    analysis_points = []
                    for record in results:
                        point = {
                            "date": record.analysis_date.isoformat(),
                            "threat_level": record.threat_level,
                            "details": record.data
                        }
                        analysis_points.append(point)
                    
                    logger.info(f"Retrieved {len(analysis_points)} historical data points for {norad_id}",
                               extra={"norad_id": norad_id, 
                                     "duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000})
                    
                    return analysis_points
        except Exception as e:
            logger.error(f"Error retrieving historical analysis from database: {str(e)}",
                       extra={"norad_id": norad_id, "error": str(e)})
            # Return empty list on error - the caller will generate simulated data
            return []

    def _chunked_query_results(self, query, chunk_size: int):
        """
        Helper method to process query results in chunks to reduce memory usage.
        
        Args:
            query: SQLAlchemy query object
            chunk_size: Number of records to process at once
        
        Yields:
            Chunks of query results
        """
        offset = 0
        while True:
            # Get a chunk of results
            chunk = query.limit(chunk_size).offset(offset).all()
            
            # If no results, we're done
            if not chunk:
                break
                
            # Yield this chunk
            yield chunk
            
            # Move to next chunk
            offset += chunk_size
            
            # If chunk is smaller than chunk_size, we're done
            if len(chunk) < chunk_size:
                break

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

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the CCDM service.
        
        Returns:
            Dictionary containing service health information
        """
        health_data = {
            "status": "operational",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "ml_evaluators": {
                    "maneuver": self._check_evaluator_health(self.maneuver_evaluator),
                    "signature": self._check_evaluator_health(self.signature_evaluator),
                    "amr": self._check_evaluator_health(self.amr_evaluator)
                },
                "database": self._check_database_health()
            }
        }
        
        # Determine overall health
        failed_services = [svc for svc, status in health_data["services"]["ml_evaluators"].items() 
                          if status["status"] != "operational"]
        
        if health_data["services"]["database"]["status"] != "operational":
            failed_services.append("database")
            
        if failed_services:
            health_data["status"] = "degraded"
            health_data["degraded_services"] = failed_services
            
        return health_data
        
    def _check_evaluator_health(self, evaluator) -> Dict[str, Any]:
        """Check health of an ML evaluator."""
        try:
            # Check if evaluator has required attributes
            if not hasattr(evaluator, 'model'):
                return {
                    "status": "degraded",
                    "error": "Evaluator missing model attribute"
                }
                
            return {
                "status": "operational",
                "model_loaded": True
            }
        except Exception as e:
            logger.error(f"Evaluator health check failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    def _check_database_health(self) -> Dict[str, Any]:
        """Check health of database connection."""
        if not self.db:
            return {
                "status": "not_configured",
                "message": "Database session not provided"
            }
            
        try:
            # Test database connection with simple query
            self.db.execute("SELECT 1")
            return {
                "status": "operational",
                "connection": "active"
            }
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _sanitize_input(self, input_str: str) -> str:
        """
        Sanitize input string to prevent injection attacks.
        
        Args:
            input_str: Input string to sanitize
            
        Returns:
            Sanitized string
        """
        if input_str is None:
            return ""
            
        # Remove any potentially dangerous characters
        # Keep only alphanumeric, dash, underscore, period
        import re
        return re.sub(r'[^\w\-\.]', '', input_str)
        
    def _is_valid_norad_id(self, norad_id: str) -> bool:
        """
        Validate if the string is a valid NORAD ID format.
        
        Args:
            norad_id: NORAD ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not norad_id:
            return False
            
        # Check if it's all digits (standard NORAD ID)
        if norad_id.isdigit():
            return True
            
        # Check for extended format (e.g. "STARLINK-1234")
        import re
        return bool(re.match(r'^[A-Za-z0-9\-]+$', norad_id))

    def authenticate_request(self, auth_token: str) -> Dict[str, Any]:
        """
        Authenticate API request using token authentication.
        
        Args:
            auth_token: Authentication token from request
            
        Returns:
            Dictionary with authentication result and user info if successful
        
        Raises:
            ValueError: If token is invalid or expired
        """
        if not auth_token:
            StructuredLogger.warning("Missing authentication token")
            raise ValueError("Authentication token is required")
            
        try:
            # For simple token authentication in development
            if DEPLOYMENT_ENV == "development" and auth_token == "dev-token":
                return {
                    "authenticated": True,
                    "user_id": "dev-user",
                    "roles": ["admin"],
                    "permissions": ["read", "write", "admin"]
                }
                
            # Import JWT library for token validation
            import jwt
            from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
            
            # Get JWT secret from environment
            jwt_secret = os.getenv("JWT_SECRET")
            if not jwt_secret:
                StructuredLogger.error("JWT secret not configured")
                raise ValueError("Authentication service not properly configured")
                
            # Decode and validate token
            try:
                decoded = jwt.decode(auth_token, jwt_secret, algorithms=["HS256"])
                
                # Validate required claims
                if "sub" not in decoded:
                    raise ValueError("Invalid token: missing required claims")
                    
                # Check if token is expired
                exp = decoded.get("exp", 0)
                if exp < time.time():
                    raise ExpiredSignatureError("Token has expired")
                    
                # Return authentication result
                return {
                    "authenticated": True,
                    "user_id": decoded["sub"],
                    "roles": decoded.get("roles", []),
                    "permissions": decoded.get("permissions", [])
                }
            except ExpiredSignatureError:
                StructuredLogger.warning("Expired authentication token", token_sub=decoded.get("sub", "unknown"))
                raise ValueError("Authentication token has expired")
            except InvalidTokenError:
                StructuredLogger.warning("Invalid authentication token")
                raise ValueError("Invalid authentication token")
                
        except Exception as e:
            StructuredLogger.error("Authentication error", error=e)
            
            if isinstance(e, ValueError):
                raise
                
            raise ValueError(f"Authentication failed: {str(e)}")
            
    def authorize_action(self, auth_info: Dict[str, Any], required_permission: str) -> bool:
        """
        Check if authenticated user has permission to perform an action.
        
        Args:
            auth_info: Authentication information from authenticate_request
            required_permission: Permission required for the action
            
        Returns:
            True if authorized, False otherwise
        """
        if not auth_info or not auth_info.get("authenticated"):
            return False
            
        # Check permissions
        permissions = auth_info.get("permissions", [])
        if "admin" in permissions or required_permission in permissions:
            return True
            
        # Check roles
        roles = auth_info.get("roles", [])
        if "admin" in roles:
            return True
            
        # Specific role-based checks
        if required_permission == "read" and any(role in ["analyst", "viewer"] for role in roles):
            return True
            
        if required_permission == "write" and any(role in ["analyst", "editor"] for role in roles):
            return True
            
        return False
        
    def get_liveness_probe(self) -> Dict[str, Any]:
        """
        Simple liveness probe to check if the service is running.
        
        Returns:
            Dictionary with service status
        """
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION
        }
        
    @handle_errors
    def get_readiness_probe(self) -> Dict[str, Any]:
        """
        Readiness probe to check if service is ready to handle requests.
        Checks all dependencies like database, ML models, etc.
        
        Returns:
            Dictionary with readiness status and details
        """
        ready = True
        dependencies = {
            "database": self._check_database_connection(),
            "ml_models": self._check_ml_models(),
            "memory": self._check_memory_status()
        }
        
        # Check if any dependency is not ready
        for dep_name, dep_status in dependencies.items():
            if not dep_status.get("ready", False):
                ready = False
                break
                
        return {
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat(),
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "dependencies": dependencies
        }
        
    def _check_database_connection(self) -> Dict[str, Any]:
        """Check database connection for readiness probe."""
        if not self.db:
            return {
                "ready": False,
                "status": "not_configured",
                "message": "Database session not provided"
            }
            
        try:
            # Test database connection with simple query and timeout
            with timeout(5, "Database connection check timed out"):
                self.db.execute("SELECT 1")
                
            return {
                "ready": True,
                "status": "connected"
            }
        except Exception as e:
            StructuredLogger.error("Database readiness check failed", error=e)
            return {
                "ready": False,
                "status": "error",
                "message": str(e)
            }
            
    def _check_ml_models(self) -> Dict[str, Any]:
        """Check ML models for readiness probe."""
        models_ready = True
        model_status = {}
        
        # Check all evaluators
        evaluators = {
            "maneuver": self.maneuver_evaluator,
            "signature": self.signature_evaluator,
            "amr": self.amr_evaluator
        }
        
        for name, evaluator in evaluators.items():
            try:
                model_loaded = hasattr(evaluator, 'model')
                model_status[name] = {
                    "ready": model_loaded,
                    "status": "loaded" if model_loaded else "not_loaded"
                }
                
                if not model_loaded:
                    models_ready = False
            except Exception as e:
                StructuredLogger.error(f"Error checking {name} model", error=e)
                model_status[name] = {
                    "ready": False,
                    "status": "error",
                    "message": str(e)
                }
                models_ready = False
                
        return {
            "ready": models_ready,
            "models": model_status
        }
        
    def _check_memory_status(self) -> Dict[str, Any]:
        """Check memory status for readiness probe."""
        try:
            import psutil
            
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Consider not ready if memory usage is above 95%
            ready = memory.percent < 95
            
            return {
                "ready": ready,
                "used_percent": memory.percent,
                "available_mb": memory.available / (1024 * 1024)
            }
        except ImportError:
            # psutil may not be available
            return {
                "ready": True,
                "status": "unknown",
                "message": "psutil not available"
            }
        except Exception as e:
            StructuredLogger.error("Error checking memory status", error=e)
            return {
                "ready": False,
                "status": "error",
                "message": str(e)
            }
            
    def graceful_shutdown(self) -> Dict[str, Any]:
        """
        Perform graceful shutdown of the service.
        Closes database connections, releases resources, etc.
        
        Returns:
            Dictionary with shutdown status
        """
        StructuredLogger.info("Starting graceful shutdown")
        shutdown_status = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "service": SERVICE_NAME,
            "components": {}
        }
        
        # Close database connection if available
        if self.db:
            try:
                StructuredLogger.info("Closing database connection")
                self.db.close()
                shutdown_status["components"]["database"] = {
                    "status": "closed",
                    "success": True
                }
            except Exception as e:
                StructuredLogger.error("Error closing database connection", error=e)
                shutdown_status["components"]["database"] = {
                    "status": "error",
                    "success": False,
                    "message": str(e)
                }
                shutdown_status["success"] = False
                
        # Close ML model resources
        try:
            StructuredLogger.info("Cleaning up ML resources")
            # Release any ML model resources here
            shutdown_status["components"]["ml_models"] = {
                "status": "released",
                "success": True
            }
        except Exception as e:
            StructuredLogger.error("Error releasing ML resources", error=e)
            shutdown_status["components"]["ml_models"] = {
                "status": "error",
                "success": False,
                "message": str(e)
            }
            shutdown_status["success"] = False
            
        # Log final shutdown status
        if shutdown_status["success"]:
            StructuredLogger.info("Graceful shutdown completed successfully")
        else:
            StructuredLogger.warning("Graceful shutdown completed with errors")
            
        return shutdown_status
    
    def __del__(self):
        """
        Destructor to ensure resources are released.
        """
        try:
            # Attempt to close database if it exists
            if hasattr(self, 'db') and self.db:
                try:
                    self.db.close()
                except:
                    pass
        except:
            # Ignore errors in destructor
            pass
