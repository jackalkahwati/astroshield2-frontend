import logging
import time
import contextlib
from typing import Dict, Any, Optional, List
import uuid

# Optional OpenTelemetry imports - gracefully handle if not installed
try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, SpanKind
    from opentelemetry.trace.status import Status, StatusCode
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    
logger = logging.getLogger(__name__)

class Span:
    """Simple span implementation for tracing and monitoring."""
    
    def __init__(self, name: str, parent_span=None):
        self.name = name
        self.parent_span = parent_span
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.span_id = str(uuid.uuid4())[:8]
        logger.debug(f"Span started: {name}")
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a key-value attribute on the span."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def end(self) -> None:
        """End the span, recording its duration."""
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000
        logger.debug(f"Span ended: {self.name} (duration: {duration_ms:.2f}ms)")
    
    def __str__(self) -> str:
        return f"Span({self.name}, attributes={self.attributes})"

@contextlib.contextmanager
def create_span(name: str, parent_span: Optional[Span] = None):
    """Context manager for creating and managing spans."""
    span = Span(name, parent_span)
    try:
        yield span
    except Exception as e:
        span.add_event("exception", {"exception.type": type(e).__name__, "exception.message": str(e)})
        raise
    finally:
        span.end()

def set_global_attribute(key: str, value: Any) -> None:
    """Set a global attribute that will be added to all spans."""
    # In a real implementation, this would store the attribute in a global context
    # For now, we'll just log it
    logger.debug(f"Global attribute set: {key}={value}")

def create_counter(name: str, description: str = None, unit: str = None) -> 'Counter':
    """Create a counter metric."""
    return Counter(name, description, unit)

def create_gauge(name: str, description: str = None, unit: str = None) -> 'Gauge':
    """Create a gauge metric."""
    return Gauge(name, description, unit)

def create_histogram(name: str, description: str = None, unit: str = None) -> 'Histogram':
    """Create a histogram metric."""
    return Histogram(name, description, unit)

class Counter:
    """Simple counter metric."""
    
    def __init__(self, name: str, description: str = None, unit: str = None):
        self.name = name
        self.description = description
        self.unit = unit
        self.value = 0
    
    def add(self, value: int = 1, attributes: Dict[str, Any] = None) -> None:
        """Increment the counter by the given value."""
        self.value += value
        logger.debug(f"Counter {self.name} incremented by {value} to {self.value}")

class Gauge:
    """Simple gauge metric."""
    
    def __init__(self, name: str, description: str = None, unit: str = None):
        self.name = name
        self.description = description
        self.unit = unit
        self.value = 0
    
    def set(self, value: float, attributes: Dict[str, Any] = None) -> None:
        """Set the gauge to the given value."""
        self.value = value
        logger.debug(f"Gauge {self.name} set to {value}")

class Histogram:
    """Simple histogram metric."""
    
    def __init__(self, name: str, description: str = None, unit: str = None):
        self.name = name
        self.description = description
        self.unit = unit
        self.values: List[float] = []
    
    def record(self, value: float, attributes: Dict[str, Any] = None) -> None:
        """Record a value in the histogram."""
        self.values.append(value)
        logger.debug(f"Histogram {self.name} recorded value {value}")

class MonitoringService:
    """
    Service for application monitoring, tracing, and metrics.
    Provides a unified interface whether OpenTelemetry is available or not.
    """
    
    def __init__(self):
        """Initialize the monitoring service"""
        self.has_opentelemetry = HAS_OPENTELEMETRY
        
        if self.has_opentelemetry:
            logger.info("Initializing monitoring with OpenTelemetry")
            self.tracer = trace.get_tracer("astroshield.monitoring")
        else:
            logger.info("OpenTelemetry not available, using minimal monitoring")
            self.tracer = None
    
    @contextlib.contextmanager
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Create a span for an operation.
        
        Args:
            name: The name of the span
            attributes: Optional attributes to set on the span
            
        Returns:
            Context manager with the span
        """
        if self.has_opentelemetry:
            with self.tracer.start_as_current_span(
                name, 
                kind=SpanKind.INTERNAL
            ) as span:
                # Set attributes if provided
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                        
                try:
                    yield span
                except Exception as e:
                    # Record error details on the span
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                    raise
        else:
            # Create a basic span implementation
            span = Span(name)
            
            # Set attributes if provided
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            try:
                yield span
            except Exception as e:
                # Record error details on the span
                span.set_status("ERROR", str(e))
                logger.error(f"Error in span {name}: {str(e)}")
                raise
            finally:
                span.end()
    
    def record_metric(self, name: str, value: float, attributes: Optional[Dict[str, Any]] = None):
        """
        Record a metric value.
        
        Args:
            name: The metric name
            value: The metric value
            attributes: Optional attributes/tags
        """
        # For minimal implementation, just log metrics
        if attributes:
            attr_str = ", ".join(f"{k}={v}" for k, v in attributes.items())
            logger.info(f"METRIC: {name}={value} ({attr_str})")
        else:
            logger.info(f"METRIC: {name}={value}") 