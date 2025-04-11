"""
UDL Secure Messaging Client

This module provides a client for interacting with the Unified Data Library (UDL) Secure Messaging API.
"""

import json
import logging
import os
import time
import datetime
import dateutil.parser
import threading
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from threading import Thread, Event
from functools import wraps
import queue

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    "sample_period": 0.34,  # Minimum time between requests (seconds)
    "circuit_breaker_threshold": 5,  # Number of failures before circuit opens
    "circuit_breaker_timeout": 60,  # Seconds before trying again after circuit opens
    "message_buffer_size": 1000,  # Maximum size of the message buffer
    "health_check_interval": 60,  # Seconds between health checks
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


class ConsumerMetrics:
    """Metrics for a UDL Secure Messaging consumer."""
    
    def __init__(self):
        """Initialize metrics."""
        self.messages_received = 0
        self.messages_processed = 0
        self.errors = 0
        self.last_offset = -1
        self.last_message_time = None
        self.started_at = time.time()
        self.latency_sum = 0
        self.latency_count = 0
        self.consumer_lag = 0
        self.topic_partition = ""
        self.status = "initialized"
        self.last_error = None
        self.last_error_time = None
        
    def record_message_received(self, count: int = 1) -> None:
        """Record message(s) received."""
        self.messages_received += count
        self.last_message_time = time.time()
        
    def record_message_processed(self, latency: Optional[float] = None) -> None:
        """Record message processed with optional latency."""
        self.messages_processed += 1
        if latency is not None:
            self.latency_sum += latency
            self.latency_count += 1
            
    def record_error(self, error: str) -> None:
        """Record an error."""
        self.errors += 1
        self.last_error = error
        self.last_error_time = time.time()
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        now = time.time()
        metrics = {
            "messages_received": self.messages_received,
            "messages_processed": self.messages_processed,
            "errors": self.errors,
            "last_offset": self.last_offset,
            "uptime_seconds": now - self.started_at,
            "consumer_lag": self.consumer_lag,
            "status": self.status,
        }
        
        if self.last_message_time:
            metrics["seconds_since_last_message"] = now - self.last_message_time
            
        if self.latency_count > 0:
            metrics["average_latency"] = self.latency_sum / self.latency_count
            
        if self.last_error:
            metrics["last_error"] = self.last_error
            if self.last_error_time:
                metrics["seconds_since_last_error"] = now - self.last_error_time
                
        return metrics


class UDLMessagingClient:
    """Client for interacting with the Unified Data Library (UDL) Secure Messaging API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        sample_period: Optional[float] = None,
        config_file: Optional[str] = None,
    ):
        """
        Initialize the UDL Secure Messaging client.

        Args:
            base_url: The base URL for the UDL API
            username: Username for authentication
            password: Password for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            sample_period: Minimum time between requests (seconds) to respect rate limits
            config_file: Path to configuration file (JSON)
        """
        # Load configuration
        self.config = self._load_configuration(config_file)
        
        # Override with constructor parameters if provided
        self.base_url = base_url or self.config["base_url"]
        self.timeout = timeout or self.config["timeout"]
        self.username = username or os.environ.get("UDL_USERNAME")
        self.password = password or os.environ.get("UDL_PASSWORD")
        self.sample_period = sample_period or self.config["sample_period"]
        self.sample_period = max(0.34, self.sample_period)  # Ensure not below minimum allowed
        
        # Validate credentials
        if not self.username or not self.password:
            raise ValueError("Username and password are required for Secure Messaging API")
            
        # Internal state
        self.topic_offsets = {}  # Track latest offset for each topic
        self._stop_event = Event()
        self._consumers = {}  # Track active consumer threads
        self._metrics = {}  # Metrics for each consumer
        self._overall_metrics = {
            "api_calls": 0,
            "api_errors": 0,
            "total_messages_processed": 0,
            "start_time": time.time(),
            "instance_id": str(uuid.uuid4()),
        }
        
        # Health check data
        self._health_check_thread = None
        self._health_check_stop_event = Event()
        self._last_health_check = None
        self._health_status = {"status": "unknown"}
        
        # Set up rate limiter and circuit breaker
        self.rate_limiter = RateLimiter(
            self.config["rate_limit_requests"],
            self.config["rate_limit_period"]
        )
        self.circuit_breaker = CircuitBreaker(
            self.config["circuit_breaker_threshold"],
            self.config["circuit_breaker_timeout"]
        )

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
        
        # Set authentication
        self.session.auth = (self.username, self.password)
        
        # Start health check thread
        self._start_health_check_thread()
    
    def _load_configuration(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file and environment.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        # Start with default configuration
        config = DEFAULT_CONFIG.copy()
        
        # Override with config file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                config.update(file_config)
                logger.info(f"Loaded UDL Messaging configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_file}: {str(e)}")
        
        # Override with environment variables
        env_prefix = "UDL_MESSAGING_"
        for key in config:
            env_key = f"{env_prefix}{key.upper()}"
            if env_key in os.environ:
                # Convert environment variable to appropriate type
                env_value = os.environ[env_key]
                if isinstance(config[key], bool):
                    config[key] = env_value.lower() in ('true', 'yes', '1')
                elif isinstance(config[key], int):
                    config[key] = int(env_value)
                elif isinstance(config[key], float):
                    config[key] = float(env_value)
                elif isinstance(config[key], list):
                    config[key] = json.loads(env_value)
                else:
                    config[key] = env_value
        
        return config
    
    def _start_health_check_thread(self) -> None:
        """Start background thread for periodic health checks."""
        if not self._health_check_thread:
            self._health_check_thread = Thread(
                target=self._health_check_loop,
                daemon=True,
                name="udl-messaging-health"
            )
            self._health_check_thread.start()
            logger.debug("Started UDL Messaging health check thread")
    
    def _health_check_loop(self) -> None:
        """Background thread for periodic health checks."""
        interval = self.config["health_check_interval"]
        
        while not self._health_check_stop_event.is_set():
            try:
                # Perform health check
                health = self.check_health()
                self._last_health_check = time.time()
                self._health_status = health
                
                # Log only if status changed or it's in error state
                if health["status"] != "ok":
                    logger.warning(f"UDL Messaging API health check: {health['status']}")
            except Exception as e:
                logger.error(f"Error in health check: {str(e)}")
                self._health_status = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
            
            # Wait for next interval or until stop is requested
            self._health_check_stop_event.wait(interval)
    
    def stop_health_check(self) -> None:
        """Stop the health check thread."""
        if self._health_check_thread:
            self._health_check_stop_event.set()
            self._health_check_thread.join(timeout=1.0)
            self._health_check_thread = None
            logger.debug("Stopped UDL Messaging health check thread")
    
    @RateLimiter(DEFAULT_CONFIG["rate_limit_requests"], DEFAULT_CONFIG["rate_limit_period"])
    @CircuitBreaker(DEFAULT_CONFIG["circuit_breaker_threshold"], DEFAULT_CONFIG["circuit_breaker_timeout"])
    def _make_request(
        self, method: str, endpoint: str, **kwargs
    ) -> requests.Response:
        """
        Make an HTTP request to the UDL Secure Messaging API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments to pass to requests

        Returns:
            Response object
        """
        url = f"{self.base_url}{endpoint}"
        
        # Add request ID for tracing
        request_id = f"sm-{int(time.time() * 1000)}"
        headers = kwargs.pop("headers", {})
        headers["X-Request-ID"] = request_id
        
        # Add default timeout if not specified
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
            
        # Update metrics
        self._overall_metrics["api_calls"] += 1
        
        # Make the request with timing
        start_time = time.time()
        try:
            response = self.session.request(method, url, headers=headers, **kwargs)
            
            # Log request details
            elapsed = time.time() - start_time
            logger.debug(
                f"UDL Messaging API request: {request_id} {method} {url}, "
                f"Status: {response.status_code}, Time: {elapsed:.2f}s"
            )
            
            # Raise exception for error status codes
            response.raise_for_status()
            
            return response
            
        except requests.exceptions.HTTPError as e:
            # Update error metrics
            self._overall_metrics["api_errors"] += 1
            
            # Log detailed error information
            elapsed = time.time() - start_time
            error_detail = f"Status: {e.response.status_code}" if hasattr(e, 'response') else "No response"
            logger.error(
                f"UDL Messaging API HTTP error: {request_id} {method} {url}, "
                f"{error_detail}, Time: {elapsed:.2f}s"
            )
            
            # Special handling for common errors
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 401:
                    logger.error("Authentication failed for UDL Messaging API. Check credentials.")
                elif e.response.status_code == 403:
                    logger.error("Permission denied. Your account may not have access to Secure Messaging.")
                elif e.response.status_code == 429:
                    # Extract retry-after header if available
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = int(retry_after)
                        logger.warning(f"Rate limited, waiting for {wait_time}s before retry")
                        time.sleep(wait_time)
                        
            raise
            
        except requests.exceptions.RequestException as e:
            # Update error metrics
            self._overall_metrics["api_errors"] += 1
            
            # Log error details
            elapsed = time.time() - start_time
            logger.error(
                f"UDL Messaging API request error: {request_id} {method} {url}, "
                f"Time: {elapsed:.2f}s, Error: {str(e)}"
            )
            raise

    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the UDL Secure Messaging API.
        
        Returns:
            Health status information
        """
        try:
            # Try to list topics as a health check
            start_time = time.time()
            topics = self.list_topics()
            
            return {
                "status": "ok",
                "topics_available": len(topics),
                "response_time": time.time() - start_time,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.datetime.utcnow().isoformat()
            }

    def list_topics(self) -> List[Dict[str, Any]]:
        """
        List available UDL Secure Messaging topics.
        
        Returns:
            List of topic information dictionaries.
        """
        response = self._make_request("GET", "/sm/listTopics")
        return response.json()

    def describe_topic(self, topic: str, partition: int = 0) -> Dict[str, Any]:
        """
        Get detailed information about a specific topic.
        
        Args:
            topic: Name of the topic to describe.
            partition: Topic partition (default: 0).
            
        Returns:
            Dictionary with topic details.
        """
        response = self._make_request("GET", f"/sm/describeTopic/{topic}/{partition}")
        return response.json()

    def get_latest_offset(self, topic: str, partition: int = 0) -> int:
        """
        Get the latest offset for a topic.
        
        Args:
            topic: Name of the topic.
            partition: Topic partition (default: 0).
            
        Returns:
            Latest offset as an integer.
        """
        response = self._make_request("GET", f"/sm/getLatestOffset/{topic}/{partition}")
        return int(response.text)

    def get_earliest_offset(self, topic: str, partition: int = 0) -> int:
        """
        Get the earliest available offset for a topic.
        
        Args:
            topic: Name of the topic.
            partition: Topic partition (default: 0).
            
        Returns:
            Earliest offset as an integer.
        """
        topic_info = self.describe_topic(topic, partition)
        return int(topic_info.get("minPos", 0))

    def get_messages(
        self, 
        topic: str, 
        offset: int = -1, 
        partition: int = 0,
        query_params: Optional[Dict[str, str]] = None,
        max_retries: int = 3
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get messages from a topic starting at a specific offset.
        
        Args:
            topic: Name of the topic.
            offset: Starting offset (-1 for latest).
            partition: Topic partition (default: 0).
            query_params: Optional query parameters to filter results.
            max_retries: Maximum number of retries for this specific request.
            
        Returns:
            Tuple of (list of messages, next offset)
        """
        params = query_params.copy() if query_params else {}
        
        # Try up to max_retries times
        retries = 0
        while retries <= max_retries:
            try:
                response = self._make_request(
                    "GET", 
                    f"/sm/getMessages/{topic}/{partition}/{offset}",
                    params=params
                )
                
                # Get the next offset from response headers
                next_offset = response.headers.get("KAFKA_NEXT_OFFSET", offset)
                if next_offset is not None:
                    next_offset = int(next_offset)
                
                # Parse response body as JSON
                messages = response.json()
                
                # Update consumer metrics if this is part of a consumer
                key = f"{topic}:{partition}"
                if key in self._metrics:
                    metrics = self._metrics[key]
                    metrics.record_message_received(len(messages))
                    metrics.last_offset = next_offset
                    
                    # Calculate consumer lag if possible
                    try:
                        latest_offset = self.get_latest_offset(topic, partition)
                        metrics.consumer_lag = latest_offset - next_offset
                    except Exception:
                        pass  # Ignore errors in lag calculation
                
                return messages, next_offset
                
            except Exception as e:
                retries += 1
                if retries <= max_retries:
                    sleep_time = self.config["backoff_factor"] * (2 ** (retries - 1))
                    logger.warning(
                        f"Error getting messages from {topic}:{partition}, "
                        f"retrying in {sleep_time:.1f}s ({retries}/{max_retries}): {str(e)}"
                    )
                    time.sleep(sleep_time)
                else:
                    # Update consumer metrics if this is part of a consumer
                    key = f"{topic}:{partition}"
                    if key in self._metrics:
                        self._metrics[key].record_error(str(e))
                    raise

    def start_consumer(
        self, 
        topic: str, 
        callback_fn: Callable[[Dict[str, Any]], None], 
        partition: int = 0, 
        start_from_latest: bool = True,
        query_params: Optional[Dict[str, str]] = None,
        process_historical: bool = False,
        buffer_size: Optional[int] = None,
        error_handler: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """
        Start a continuous consumer for a topic.
        
        Args:
            topic: Name of the topic to consume.
            callback_fn: Function to call with received messages.
            partition: Topic partition (default: 0).
            start_from_latest: Whether to start from latest offset or beginning.
            query_params: Optional query parameters to filter results.
            process_historical: If True, process all historical data before streaming.
            buffer_size: Size of the message buffer (default from config).
            error_handler: Optional function to call on consumer errors.
        """
        key = f"{topic}:{partition}"
        if key in self._consumers and self._consumers[key].is_alive():
            logger.warning(f"Consumer for {key} already running")
            return
        
        # Initialize metrics for this consumer
        self._metrics[key] = ConsumerMetrics()
        self._metrics[key].topic_partition = key
        self._metrics[key].status = "starting"
        
        # Create message buffer for the consumer
        message_buffer = queue.Queue(maxsize=buffer_size or self.config["message_buffer_size"])
        
        # Create and start consumer thread
        consumer_thread = Thread(
            target=self._consumer_thread,
            args=(
                topic, 
                message_buffer, 
                partition, 
                start_from_latest, 
                query_params, 
                process_historical,
                error_handler
            ),
            daemon=True,
            name=f"udl-consumer-{topic}-{partition}"
        )
        self._consumers[key] = consumer_thread
        consumer_thread.start()
        
        # Create and start worker thread to process messages
        worker_thread = Thread(
            target=self._worker_thread,
            args=(key, message_buffer, callback_fn, error_handler),
            daemon=True,
            name=f"udl-worker-{topic}-{partition}"
        )
        worker_thread.start()
        
        logger.info(f"Started consumer and worker for {key}")

    def stop_consumer(self, topic: str, partition: int = 0, wait: bool = False) -> None:
        """
        Stop a specific consumer.
        
        Args:
            topic: Name of the topic.
            partition: Topic partition.
            wait: If True, wait for the consumer to terminate.
        """
        key = f"{topic}:{partition}"
        if key in self._consumers:
            if self._metrics[key].status != "stopping":
                self._metrics[key].status = "stopping"
                logger.info(f"Stopping consumer for {key}")
            
            # Consumer will check _stop_event and terminate
            if wait and self._consumers[key].is_alive():
                self._consumers[key].join(timeout=5.0)
                if self._consumers[key].is_alive():
                    logger.warning(f"Consumer for {key} did not terminate in time")
            
            # Remove from active consumers if not alive
            if not self._consumers[key].is_alive():
                del self._consumers[key]
                logger.info(f"Consumer for {key} stopped")
        else:
            logger.warning(f"No active consumer found for {key}")

    def stop_all_consumers(self, wait: bool = True) -> None:
        """
        Stop all active consumer threads.
        
        Args:
            wait: If True, wait for consumers to terminate.
        """
        self._stop_event.set()
        
        if wait:
            # Wait for all consumers to terminate
            for key, thread in list(self._consumers.items()):
                if thread.is_alive():
                    logger.info(f"Waiting for consumer {key} to terminate...")
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        logger.warning(f"Consumer {key} did not terminate in time")
        
        # Clear consumer registry
        self._consumers.clear()
        
        # Reset stop event for future use
        self._stop_event.clear()
        logger.info("All consumers stopped")
    
    def get_consumer_metrics(self, topic: Optional[str] = None, partition: int = 0) -> Dict[str, Any]:
        """
        Get metrics for a specific consumer or all consumers.
        
        Args:
            topic: Name of the topic (or None for all).
            partition: Topic partition.
            
        Returns:
            Dictionary with consumer metrics.
        """
        if topic:
            key = f"{topic}:{partition}"
            if key in self._metrics:
                return self._metrics[key].get_metrics()
            else:
                return {"error": "Consumer not found"}
        else:
            # Return metrics for all consumers
            result = {}
            for key, metrics in self._metrics.items():
                result[key] = metrics.get_metrics()
            return result
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """
        Get overall metrics for the messaging client.
        
        Returns:
            Dictionary with overall metrics.
        """
        # Calculate derived metrics
        uptime = time.time() - self._overall_metrics["start_time"]
        active_consumers = len([t for t in self._consumers.values() if t.is_alive()])
        
        # Combine with base metrics
        metrics = {
            **self._overall_metrics,
            "uptime_seconds": uptime,
            "active_consumers": active_consumers,
        }
        
        # Add health status
        if self._last_health_check:
            metrics["last_health_check"] = self._last_health_check
            metrics["health_status"] = self._health_status["status"]
            if "error" in self._health_status:
                metrics["health_error"] = self._health_status["error"]
        
        return metrics
    
    def _consumer_thread(
        self, 
        topic: str, 
        message_buffer: queue.Queue,
        partition: int,
        start_from_latest: bool,
        query_params: Optional[Dict[str, str]],
        process_historical: bool,
        error_handler: Optional[Callable[[Exception], None]] = None
    ) -> None:
        """
        Background thread for continuous topic consumption.
        
        Args:
            topic: Name of the topic to consume.
            message_buffer: Queue to put messages into.
            partition: Topic partition.
            start_from_latest: Whether to start from latest offset.
            query_params: Optional query parameters.
            process_historical: If True, process all historical data.
            error_handler: Optional function to call on errors.
        """
        key = f"{topic}:{partition}"
        metrics = self._metrics[key]
        metrics.status = "running"
        
        try:
            # Determine starting offset
            if start_from_latest and not process_historical:
                # Start from latest offset
                offset = -1
            else:
                # Get topic info to find earliest offset
                topic_info = self.describe_topic(topic, partition)
                offset = topic_info.get("minPos", 0) if process_historical else topic_info.get("maxPos", -1)
                
            logger.info(f"Starting consumer for {key} from offset {offset}")
            
            # Track the current offset
            current_offset = offset
            
            # Main consumer loop
            while not self._stop_event.is_set():
                try:
                    # Get messages from the topic at the current offset
                    start_time = time.time()
                    messages, next_offset = self.get_messages(topic, current_offset, partition, query_params)
                    
                    # Process messages if any were received
                    if messages:
                        receive_time_utc = datetime.datetime.utcnow()
                        
                        # Enrich messages with metadata
                        for msg in messages:
                            # Add message metadata
                            msg["_receive_time_utc"] = receive_time_utc.isoformat()
                            msg["_topic"] = topic
                            msg["_partition"] = partition
                            msg["_offset"] = current_offset
                            
                            # Add latency information if available
                            latency_info = self.calculate_latency(msg)
                            if latency_info:
                                msg["_latency"] = latency_info
                            
                            # Put message in queue for worker thread
                            try:
                                message_buffer.put(msg, block=True, timeout=1.0)
                            except queue.Full:
                                logger.warning(f"Message buffer full for {key}, dropping message")
                                metrics.record_error("Message buffer full, messages dropped")
                    
                    # Update offset for next iteration
                    current_offset = next_offset
                    
                    # Check if we've reached the current end of the topic
                    if str(next_offset) == str(current_offset):
                        # We've caught up, sleep for a bit before checking again
                        time.sleep(self.sample_period)
                    else:
                        # Calculate time elapsed and sleep if needed to respect rate limits
                        elapsed = time.time() - start_time
                        if elapsed < self.sample_period:
                            time.sleep(self.sample_period - elapsed)
                            
                except Exception as e:
                    error_msg = f"Error in consumer thread for {key}: {str(e)}"
                    logger.error(error_msg)
                    metrics.record_error(error_msg)
                    
                    # Call error handler if provided
                    if error_handler:
                        try:
                            error_handler(e)
                        except Exception as handler_error:
                            logger.error(f"Error in error handler for {key}: {str(handler_error)}")
                    
                    # Back off on errors to avoid hammering the server
                    time.sleep(min(30, self.sample_period * 5))
        
        except Exception as e:
            error_msg = f"Fatal error in consumer thread for {key}: {str(e)}"
            logger.error(error_msg)
            metrics.record_error(error_msg)
            metrics.status = "error"
            
            # Call error handler if provided
            if error_handler:
                try:
                    error_handler(e)
                except Exception as handler_error:
                    logger.error(f"Error in error handler for {key}: {str(handler_error)}")
        
        # Mark consumer as stopped
        metrics.status = "stopped"
        logger.info(f"Consumer thread for {key} terminated")
    
    def _worker_thread(
        self, 
        key: str, 
        message_buffer: queue.Queue, 
        callback_fn: Callable[[Dict[str, Any]], None],
        error_handler: Optional[Callable[[Exception], None]] = None
    ) -> None:
        """
        Background thread for processing messages from the buffer.
        
        Args:
            key: Topic:partition key.
            message_buffer: Queue with messages to process.
            callback_fn: Function to call with each message.
            error_handler: Optional function to call on errors.
        """
        metrics = self._metrics[key]
        
        while not self._stop_event.is_set():
            try:
                # Get message from buffer with timeout
                try:
                    message = message_buffer.get(block=True, timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the message
                try:
                    start_time = time.time()
                    callback_fn(message)
                    processing_time = time.time() - start_time
                    
                    # Update metrics
                    metrics.record_message_processed(processing_time)
                    self._overall_metrics["total_messages_processed"] += 1
                    
                    # Mark as done in the queue
                    message_buffer.task_done()
                    
                except Exception as e:
                    error_msg = f"Error in callback function for {key}: {str(e)}"
                    logger.error(error_msg)
                    metrics.record_error(error_msg)
                    
                    # Call error handler if provided
                    if error_handler:
                        try:
                            error_handler(e)
                        except Exception as handler_error:
                            logger.error(f"Error in error handler for {key}: {str(handler_error)}")
                    
                    # Mark as done in the queue
                    message_buffer.task_done()
            
            except Exception as e:
                error_msg = f"Error in worker thread for {key}: {str(e)}"
                logger.error(error_msg)
                metrics.record_error(error_msg)
                
                # Call error handler if provided
                if error_handler:
                    try:
                        error_handler(e)
                    except Exception as handler_error:
                        logger.error(f"Error in error handler for {key}: {str(handler_error)}")
        
        logger.info(f"Worker thread for {key} terminated")

    def calculate_latency(self, message: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate latency metrics for a message if it has timestamp fields.
        
        Args:
            message: Message from UDL Secure Messaging.
            
        Returns:
            Dictionary with latency metrics (or empty if not calculable).
        """
        latency = {}
        receive_time_utc = datetime.datetime.utcnow()
        
        if "epoch" in message and "createdAt" in message:
            try:
                v_time_collect = dateutil.parser.parse(message["epoch"]).replace(tzinfo=None)
                v_time_create = dateutil.parser.parse(message["createdAt"]).replace(tzinfo=None)
                
                # Time between collection and creation in UDL
                latency["time_to_push"] = (v_time_create - v_time_collect).total_seconds()
                
                # Time from UDL creation to client receipt
                latency["time_to_pull"] = (receive_time_utc - v_time_create).total_seconds() - self.sample_period
                
                # Total latency
                latency["total_latency"] = (receive_time_utc - v_time_collect).total_seconds()
            except (ValueError, TypeError) as e:
                logger.debug(f"Couldn't calculate latency for message: {e}")
                
        return latency
    
    def __del__(self):
        """Clean up when object is garbage collected."""
        try:
            self.stop_all_consumers(wait=False)
            self.stop_health_check()
        except:
            pass  # Ignore errors during cleanup 