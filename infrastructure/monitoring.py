from prometheus_client import Counter, Gauge, Histogram, start_http_server
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MonitoringService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MonitoringService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            # Prometheus metrics
            self.request_counter = Counter(
                'astroshield_requests_total',
                'Total number of requests',
                ['method', 'endpoint', 'status']
            )
            self.active_requests = Gauge(
                'astroshield_active_requests',
                'Number of active requests',
                ['endpoint']
            )
            self.request_duration = Histogram(
                'astroshield_request_duration_seconds',
                'Request duration in seconds',
                ['endpoint'],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
            )
            
            # OpenTelemetry setup
            trace.set_tracer_provider(TracerProvider())
            metrics.set_meter_provider(MeterProvider())
            
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
            
            # Metrics for spacecraft operations
            self.spacecraft_count = self.meter.create_counter(
                "spacecraft_count",
                description="Number of spacecraft being tracked"
            )
            self.collision_risk = self.meter.create_gauge(
                "collision_risk",
                description="Current collision risk level"
            )
            
            self.initialized = True

    def start_prometheus_server(self, port: int = 8000):
        """Start Prometheus metrics server"""
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {str(e)}")
            raise

    def configure_telemetry(
        self,
        otlp_endpoint: str = "localhost:4317",
        service_name: str = "astroshield"
    ):
        """Configure OpenTelemetry exporters"""
        try:
            # Configure trace exporter
            otlp_trace_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            span_processor = BatchSpanProcessor(otlp_trace_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Configure metrics exporter
            otlp_metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
            metric_reader = PeriodicExportingMetricReader(otlp_metric_exporter)
            metrics.get_meter_provider().add_metric_reader(metric_reader)
            
            logger.info("Telemetry configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure telemetry: {str(e)}")
            raise

    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Record API request metrics"""
        try:
            self.request_counter.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
            self.request_duration.labels(endpoint=endpoint).observe(duration)
        except Exception as e:
            logger.error(f"Failed to record request metrics: {str(e)}")

    def track_spacecraft(self, count: int):
        """Update spacecraft tracking count"""
        try:
            self.spacecraft_count.add(count)
        except Exception as e:
            logger.error(f"Failed to update spacecraft count: {str(e)}")

    def update_collision_risk(self, risk_level: float):
        """Update collision risk level"""
        try:
            self.collision_risk.set(risk_level)
        except Exception as e:
            logger.error(f"Failed to update collision risk: {str(e)}")

    def create_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Create a new trace span"""
        try:
            return self.tracer.start_span(
                name,
                attributes=attributes or {}
            )
        except Exception as e:
            logger.error(f"Failed to create span: {str(e)}")
            return None

# Example usage:
# monitoring = MonitoringService()
# monitoring.start_prometheus_server()
# monitoring.configure_telemetry()
#
# # Record API request
# monitoring.record_request("GET", "/api/spacecraft", 200, 0.5)
#
# # Update spacecraft metrics
# monitoring.track_spacecraft(10)
# monitoring.update_collision_risk(0.25)
#
# # Create trace span
# with monitoring.create_span("process_telemetry", {"spacecraft_id": "123"}) as span:
#     # Process telemetry data
#     span.set_attribute("data_points", 100)
