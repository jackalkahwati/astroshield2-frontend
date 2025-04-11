"""
UDL API Client

This module provides a client for interacting with the Unified Data Library (UDL) APIs.
"""

import json
import logging
import os
import time
import base64
import threading
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
from functools import wraps
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure module logger
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "base_url": "https://unifieddatalibrary.com",
    "timeout": 30,
    "max_retries": 3,
    "backoff_factor": 0.5,
    "retry_status_codes": [429, 500, 502, 503, 504],
    "rate_limit_requests": 3,  # UDL allows 3 requests per second
    "rate_limit_period": 1.0,  # 1 second period
    "circuit_breaker_threshold": 5,  # Number of failures before circuit opens
    "circuit_breaker_timeout": 60,  # Seconds before trying again after circuit opens
    "cache_ttl": 300,  # Default cache TTL in seconds (5 minutes)
    "api_version": "1.0",  # Current UDL API version
}

class RateLimiter:
    """Rate limiter implementation for API requests."""
    
    def __init__(self, max_calls: int, period: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.RLock()
        
    def __call__(self, func):
        """Decorator to rate limit function calls."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                # Remove calls older than the period
                self.calls = [t for t in self.calls if now - t < self.period]
                
                if len(self.calls) >= self.max_calls:
                    # We've hit the rate limit, calculate sleep time
                    sleep_time = self.period - (now - self.calls[0])
                    if sleep_time > 0:
                        logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                
                # Add the current call timestamp
                self.calls.append(time.time())
                
                # Execute the function
                return func(*args, **kwargs)
        return wrapper


class CircuitBreaker:
    """Circuit breaker implementation for API requests."""
    
    def __init__(self, threshold: int, timeout: int):
        """
        Initialize circuit breaker.
        
        Args:
            threshold: Number of failures before circuit opens
            timeout: Seconds before trying again after circuit opens
        """
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.state = "closed"  # closed, open, half-open
        self.last_failure_time = 0
        self.lock = threading.RLock()
        
    def __call__(self, func):
        """Decorator to apply circuit breaker pattern."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                
                # Check if circuit is open
                if self.state == "open":
                    if now - self.last_failure_time > self.timeout:
                        logger.info("Circuit half-open, allowing test request")
                        self.state = "half-open"
                    else:
                        raise Exception(f"Circuit open, request rejected (retry after {self.timeout - (now - self.last_failure_time):.1f}s)")
                
                try:
                    result = func(*args, **kwargs)
                    
                    # If successful and in half-open state, reset circuit
                    if self.state == "half-open":
                        logger.info("Circuit reset to closed after successful request")
                        self.failures = 0
                        self.state = "closed"
                    
                    return result
                    
                except Exception as e:
                    self.failures += 1
                    self.last_failure_time = now
                    
                    if self.failures >= self.threshold:
                        self.state = "open"
                        logger.warning(f"Circuit opened after {self.failures} failures")
                    
                    raise e
                    
        return wrapper


class Cache:
    """Simple cache implementation for API responses."""
    
    def __init__(self, ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            ttl: Default time-to-live for cache entries in seconds
        """
        self.cache = {}
        self.ttl = ttl
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if it exists and is not expired."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() < entry["expiry"]:
                    logger.debug(f"Cache hit for {key}")
                    return entry["value"]
                else:
                    # Remove expired entry
                    del self.cache[key]
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cache value with expiry time."""
        with self.lock:
            expiry = time.time() + (ttl if ttl is not None else self.ttl)
            self.cache[key] = {"value": value, "expiry": expiry}
            
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()


class UDLClient:
    """Client for interacting with the Unified Data Library (UDL) APIs."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        config_file: Optional[str] = None,
    ):
        """
        Initialize the UDL API client.

        Args:
            base_url: The base URL for the UDL API
            api_key: API key for authentication
            username: Username for authentication
            password: Password for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            config_file: Path to configuration file
        """
        # Load configuration
        self.config = DEFAULT_CONFIG.copy()
        
        # Override with config file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
        
        # Override with environment variables
        env_prefix = "UDL_"
        for key in self.config:
            env_key = f"{env_prefix}{key.upper()}"
            if env_key in os.environ:
                # Convert environment variable to appropriate type
                env_value = os.environ[env_key]
                if isinstance(self.config[key], bool):
                    self.config[key] = env_value.lower() in ('true', 'yes', '1')
                elif isinstance(self.config[key], int):
                    self.config[key] = int(env_value)
                elif isinstance(self.config[key], float):
                    self.config[key] = float(env_value)
                elif isinstance(self.config[key], list):
                    self.config[key] = json.loads(env_value)
                else:
                    self.config[key] = env_value
        
        # Override with constructor parameters
        self.base_url = base_url or self.config["base_url"]
        self.timeout = timeout or self.config["timeout"]
        
        # Authentication credentials
        self.api_key = api_key or os.environ.get("UDL_API_KEY")
        self.username = username or os.environ.get("UDL_USERNAME")
        self.password = password or os.environ.get("UDL_PASSWORD")
        self.token = None
        self.token_expiry = 0
        
        # Validate we have at least one authentication method
        if not (self.api_key or (self.username and self.password)):
            logger.warning("No authentication credentials provided. Some API calls may fail.")

        # Set up session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries or self.config["max_retries"],
            backoff_factor=self.config["backoff_factor"],
            status_forcelist=self.config["retry_status_codes"],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set up rate limiter, circuit breaker, and cache
        self.rate_limiter = RateLimiter(
            self.config["rate_limit_requests"],
            self.config["rate_limit_period"]
        )
        self.circuit_breaker = CircuitBreaker(
            self.config["circuit_breaker_threshold"],
            self.config["circuit_breaker_timeout"]
        )
        self.cache = Cache(self.config["cache_ttl"])
        
        # Version check
        self._check_api_version()
        
    def _check_api_version(self) -> None:
        """Check if the API version is compatible."""
        try:
            # Try to get API version information - this is a placeholder since
            # the actual UDL API might have a different version endpoint
            version_response = self.session.get(
                f"{self.base_url}/udl/version",
                timeout=self.timeout
            )
            
            if version_response.ok:
                server_version = version_response.json().get("version", "unknown")
                client_version = self.config["api_version"]
                
                # Simple version check (could be more sophisticated)
                if server_version != client_version:
                    logger.warning(
                        f"API version mismatch: client={client_version}, server={server_version}"
                    )
        except Exception as e:
            logger.warning(f"Failed to check API version: {str(e)}")

    def _get_auth_header(self) -> Dict[str, str]:
        """
        Get the authentication header for API requests.

        Returns:
            Dict containing the authentication header.
        """
        if self.api_key:
            return {"X-API-Key": self.api_key}
        
        # Use Basic Authentication when username/password are provided
        if self.username and self.password:
            auth_str = f"{self.username}:{self.password}"
            auth_bytes = auth_str.encode('ascii')
            base64_bytes = base64.b64encode(auth_bytes)
            base64_auth = base64_bytes.decode('ascii')
            return {"Authorization": f"Basic {base64_auth}"}
        
        # If no authentication method is available
        return {}

    def _refresh_token(self) -> None:
        """
        Refresh the authentication token.
        Note: This method is kept for backward compatibility but is not used
        as we're now using Basic Authentication.
        """
        if not self.username or not self.password:
            raise ValueError("Username and password are required for token authentication")
        
        auth_url = f"{self.base_url}/auth/token"
        response = self.session.post(
            auth_url,
            json={"username": self.username, "password": self.password},
            timeout=self.timeout,
        )
        response.raise_for_status()
        
        auth_data = response.json()
        self.token = auth_data["token"]
        # Set token expiry to 5 minutes before actual expiry
        self.token_expiry = time.time() + auth_data.get("expires_in", 3600) - 300

    @RateLimiter(DEFAULT_CONFIG["rate_limit_requests"], DEFAULT_CONFIG["rate_limit_period"])
    @CircuitBreaker(DEFAULT_CONFIG["circuit_breaker_threshold"], DEFAULT_CONFIG["circuit_breaker_timeout"])
    def _make_request(
        self, method: str, endpoint: str, cache_key: Optional[str] = None, 
        cache_ttl: Optional[int] = None, **kwargs
    ) -> requests.Response:
        """
        Make an HTTP request to the UDL API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            cache_key: Key to use for caching (if None, request is not cached)
            cache_ttl: Cache TTL in seconds (if None, default TTL is used)
            **kwargs: Additional arguments to pass to requests

        Returns:
            Response object
        """
        # Check cache for GET requests
        if method == "GET" and cache_key:
            cached_response = self.cache.get(cache_key)
            if cached_response:
                return cached_response
                
        url = f"{self.base_url}{endpoint}"
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_header())
        
        # Add request ID for tracing
        request_id = f"req-{int(time.time() * 1000)}"
        headers["X-Request-ID"] = request_id
        
        # Add API version header if available
        if "api_version" in self.config:
            headers["X-API-Version"] = self.config["api_version"]
        
        # Add default timeout if not specified
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        
        start_time = time.time()
        try:
            response = self.session.request(method, url, headers=headers, **kwargs)
            
            # Log request details
            elapsed = time.time() - start_time
            logger.debug(
                f"UDL API request: {request_id} {method} {url}, "
                f"Status: {response.status_code}, Time: {elapsed:.2f}s"
            )
            
            # Check for rate limiting headers and adjust if needed
            if "X-RateLimit-Remaining" in response.headers:
                remaining = int(response.headers["X-RateLimit-Remaining"])
                if remaining < 10:
                    logger.warning(f"Rate limit running low: {remaining} requests remaining")
            
            # Raise exception for error status codes
            response.raise_for_status()
            
            # Cache successful GET responses if cache_key is provided
            if method == "GET" and cache_key and response.ok:
                self.cache.set(cache_key, response, cache_ttl)
            
            return response
            
        except requests.exceptions.HTTPError as e:
            elapsed = time.time() - start_time
            logger.error(
                f"UDL API HTTP error: {request_id} {method} {url}, "
                f"Status: {e.response.status_code}, Time: {elapsed:.2f}s, "
                f"Error: {e.response.text[:200]}"
            )
            
            # Handle specific error codes
            if e.response.status_code == 429:
                # Extract retry-after header if available
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    wait_time = int(retry_after)
                    logger.warning(f"Rate limited, waiting for {wait_time}s before retry")
                    time.sleep(wait_time)
            
            raise
            
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            logger.error(
                f"UDL API request error: {request_id} {method} {url}, "
                f"Time: {elapsed:.2f}s, Error: {str(e)}"
            )
            raise

    def get_state_vectors(
        self, epoch: str, use_cache: bool = True, **query_params
    ) -> List[Dict[str, Any]]:
        """
        Get state vectors for a specific epoch.

        Args:
            epoch: Time of validity for state vector in ISO 8601 UTC datetime format
            use_cache: Whether to use cache for this request
            **query_params: Additional query parameters

        Returns:
            List of state vector objects
        """
        params = {"epoch": epoch, **query_params}
        
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            # Create a deterministic string from the parameters
            param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            cache_key = f"statevector_{param_str}"
        
        response = self._make_request(
            "GET", "/udl/statevector", 
            params=params, 
            cache_key=cache_key
        )
        return response.json()

    def get_conjunctions(self, use_cache: bool = True, **query_params) -> List[Dict[str, Any]]:
        """
        Get conjunction data.

        Args:
            use_cache: Whether to use cache for this request
            **query_params: Query parameters

        Returns:
            List of conjunction objects
        """
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            # Create a deterministic string from the parameters
            param_str = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))
            cache_key = f"conjunction_{param_str}"
            
        response = self._make_request(
            "GET", "/udl/conjunction", 
            params=query_params, 
            cache_key=cache_key
        )
        return response.json()

    def get_launch_events(self, msg_create_date: str, **query_params) -> List[Dict[str, Any]]:
        """
        Get launch event data.

        Args:
            msg_create_date: Timestamp of the originating message in ISO8601 UTC format
            **query_params: Additional query parameters

        Returns:
            List of launch event objects
        """
        params = {"msgCreateDate": msg_create_date, **query_params}
        response = self._make_request("GET", "/udl/launchevent", params=params)
        return response.json()

    def get_tracks(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get track data.

        Args:
            **query_params: Query parameters

        Returns:
            List of track objects
        """
        response = self._make_request("GET", "/udl/track", params=query_params)
        return response.json()

    def get_orbit_determination(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get orbit determination data.

        Args:
            **query_params: Query parameters

        Returns:
            List of orbit determination objects
        """
        response = self._make_request("GET", "/udl/orbitdetermination", params=query_params)
        return response.json()

    def get_observations(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get observation data.

        Args:
            **query_params: Query parameters

        Returns:
            List of observation objects
        """
        response = self._make_request("GET", "/udl/observation", params=query_params)
        return response.json()

    def get_onorbit_objects(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get on-orbit object data.

        Args:
            **query_params: Query parameters

        Returns:
            List of on-orbit object data
        """
        response = self._make_request("GET", "/udl/onorbit", params=query_params)
        return response.json()

    # Additional UDL API Methods

    def get_ephemeris(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get ephemeris data for predicting positions of objects.

        Args:
            **query_params: Query parameters (object_id, start_time, end_time, etc.)

        Returns:
            List of ephemeris objects
        """
        response = self._make_request("GET", "/udl/ephemeris", params=query_params)
        return response.json()

    def get_maneuvers(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get spacecraft maneuver data.

        Args:
            **query_params: Query parameters (object_id, start_time, end_time, etc.)

        Returns:
            List of maneuver objects
        """
        response = self._make_request("GET", "/udl/maneuver", params=query_params)
        return response.json()

    def get_sensor_data(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get sensor information and capabilities.

        Args:
            **query_params: Query parameters (sensor_id, type, etc.)

        Returns:
            List of sensor objects
        """
        response = self._make_request("GET", "/udl/sensor", params=query_params)
        return response.json()

    def get_sensor_tasking(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get sensor tasking requests and status.

        Args:
            **query_params: Query parameters (sensor_id, object_id, status, etc.)

        Returns:
            List of sensor tasking objects
        """
        response = self._make_request("GET", "/udl/sensor-tasking", params=query_params)
        return response.json()

    def create_sensor_tasking(self, tasking_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new sensor tasking request.

        Args:
            tasking_data: Sensor tasking request data

        Returns:
            Created sensor tasking object
        """
        response = self._make_request("POST", "/udl/sensor-tasking", json=tasking_data)
        return response.json()

    def get_site_data(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get ground station and facility information.

        Args:
            **query_params: Query parameters (site_id, country, etc.)

        Returns:
            List of site objects
        """
        response = self._make_request("GET", "/udl/site", params=query_params)
        return response.json()

    def get_weather_data(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get weather data affecting space operations.

        Args:
            **query_params: Query parameters (location, time, etc.)

        Returns:
            List of weather data objects
        """
        response = self._make_request("GET", "/udl/weather", params=query_params)
        return response.json()

    def get_elset_data(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get two-line element sets for orbital objects.

        Args:
            **query_params: Query parameters (object_id, epoch, etc.)

        Returns:
            List of ELSET objects
        """
        response = self._make_request("GET", "/udl/elset", params=query_params)
        return response.json()

    def get_rf_data(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get radio frequency data.

        Args:
            **query_params: Query parameters (frequency, object_id, etc.)

        Returns:
            List of RF data objects
        """
        response = self._make_request("GET", "/udl/rf", params=query_params)
        return response.json()

    def get_earth_orientation_parameters(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get Earth orientation parameters.

        Args:
            **query_params: Query parameters (date, etc.)

        Returns:
            List of Earth orientation parameter objects
        """
        response = self._make_request("GET", "/udl/earth-orientation-parameters", params=query_params)
        return response.json()

    def get_solar_geomagnetic_data(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get solar and geomagnetic data.

        Args:
            **query_params: Query parameters (date, type, etc.)

        Returns:
            List of solar/geomagnetic data objects
        """
        response = self._make_request("GET", "/udl/solar-geomagnetic", params=query_params)
        return response.json()

    def get_star_catalog(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get star catalog information.

        Args:
            **query_params: Query parameters (magnitude, ra, dec, etc.)

        Returns:
            List of star catalog entries
        """
        response = self._make_request("GET", "/udl/star-catalog", params=query_params)
        return response.json()

    def get_tdoa_fdoa(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get Time Difference of Arrival (TDOA) and Frequency Difference of Arrival (FDOA) data.

        Args:
            **query_params: Query parameters

        Returns:
            List of TDOA/FDOA data objects
        """
        response = self._make_request("GET", "/udl/tdoa-fdoa", params=query_params)
        return response.json()

    def get_cyber_threats(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get cyber threat notifications.

        Args:
            **query_params: Query parameters (severity, target, etc.)

        Returns:
            List of cyber threat objects
        """
        response = self._make_request("GET", "/udl/cyber-threats", params=query_params)
        return response.json()

    def get_link_status(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get link status information.

        Args:
            **query_params: Query parameters (object_id, etc.)

        Returns:
            List of link status objects
        """
        response = self._make_request("GET", "/udl/link-status", params=query_params)
        return response.json()

    def get_comm_data(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get communications data.

        Args:
            **query_params: Query parameters (object_id, frequency, etc.)

        Returns:
            List of communications data objects
        """
        response = self._make_request("GET", "/udl/comm", params=query_params)
        return response.json()

    def get_mission_ops_data(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get mission operations data.

        Args:
            **query_params: Query parameters (mission_id, etc.)

        Returns:
            List of mission operations objects
        """
        response = self._make_request("GET", "/udl/mission-ops", params=query_params)
        return response.json()

    def get_vessel_data(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get vessel tracking data.

        Args:
            **query_params: Query parameters (vessel_id, location, etc.)

        Returns:
            List of vessel objects
        """
        response = self._make_request("GET", "/udl/vessel", params=query_params)
        return response.json()

    def get_aircraft_data(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get aircraft tracking data.

        Args:
            **query_params: Query parameters (aircraft_id, location, etc.)

        Returns:
            List of aircraft objects
        """
        response = self._make_request("GET", "/udl/aircraft", params=query_params)
        return response.json()

    def get_ground_imagery(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get ground imagery data.

        Args:
            **query_params: Query parameters (location, date, etc.)

        Returns:
            List of ground imagery objects
        """
        response = self._make_request("GET", "/udl/ground-imagery", params=query_params)
        return response.json()

    def get_sky_imagery(self, **query_params) -> List[Dict[str, Any]]:
        """
        Get sky imagery data.

        Args:
            **query_params: Query parameters (location, date, etc.)

        Returns:
            List of sky imagery objects
        """
        response = self._make_request("GET", "/udl/sky-imagery", params=query_params)
        return response.json()

    def get_video_streaming(self, **query_params) -> Dict[str, Any]:
        """
        Get video streaming information.

        Args:
            **query_params: Query parameters (source_id, etc.)

        Returns:
            Video streaming information
        """
        response = self._make_request("GET", "/udl/video-streaming", params=query_params)
        return response.json()

    # Method to clear cache
    def clear_cache(self) -> None:
        """Clear the client's cache."""
        self.cache.clear()
        logger.info("Cache cleared")
        
    # Method to get API health status
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get UDL API health status.
        
        Returns:
            Dict with health status information
        """
        try:
            response = self._make_request("GET", "/udl/health", cache_ttl=60)
            return {
                "status": "ok",
                "api_available": True,
                "response_time": response.elapsed.total_seconds(),
                "details": response.json()
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "error",
                "api_available": False,
                "error": str(e)
            }

    # Method to validate schema of responses
    def validate_response(self, response_data: Any, schema_name: str) -> bool:
        """
        Validate response data against expected schema.
        
        Args:
            response_data: Response data to validate
            schema_name: Name of the schema to validate against
            
        Returns:
            True if validation passes, False otherwise
        """
        # This is a placeholder - in a real implementation you would:
        # 1. Load JSON schema definitions
        # 2. Use a library like jsonschema to validate
        logger.debug(f"Schema validation for {schema_name} not implemented")
        return True 