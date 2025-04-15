"""
Metrics collection for AstroShield components.
"""
import time
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger("astroshield.monitoring")

# Define metrics
MESSAGE_COUNTER = Counter(
    'astroshield_messages_processed_total', 
    'Total number of Kafka messages processed',
    ['message_type', 'status']
)

PROCESSING_TIME = Histogram(
    'astroshield_message_processing_seconds', 
    'Time spent processing messages',
    ['message_type']
)

CONSUMER_LAG = Gauge(
    'astroshield_consumer_lag', 
    'Kafka consumer lag in messages',
    ['topic', 'partition']
)

def initialize_metrics(port=8000):
    """
    Start metrics HTTP server
    
    Args:
        port: Port to listen on
    """
    try:
        start_http_server(port)
        logger.info(f"Started metrics server on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {str(e)}")
    
class MetricsMiddleware:
    """Middleware for collecting metrics on message processing"""
    
    def __init__(self, message_type):
        """
        Initialize the middleware
        
        Args:
            message_type: Type of message being processed
        """
        self.message_type = message_type
        
    async def __call__(self, message, next_handler):
        """
        Process a message and collect metrics
        
        Args:
            message: Message to process
            next_handler: Next handler in the chain
            
        Returns:
            Result from the next handler
        """
        start_time = time.time()
        try:
            result = await next_handler(message)
            MESSAGE_COUNTER.labels(self.message_type, 'success').inc()
            return result
        except Exception as e:
            MESSAGE_COUNTER.labels(self.message_type, 'error').inc()
            raise
        finally:
            PROCESSING_TIME.labels(self.message_type).observe(time.time() - start_time)
            
def record_consumer_lag(topic, partition, lag):
    """
    Record consumer lag
    
    Args:
        topic: Kafka topic
        partition: Kafka partition
        lag: Lag in messages
    """
    CONSUMER_LAG.labels(topic=topic, partition=partition).set(lag)
    
def track_message_rate(message_type, status='success'):
    """
    Increment message counter
    
    Args:
        message_type: Type of message
        status: Processing status
    """
    MESSAGE_COUNTER.labels(message_type=message_type, status=status).inc() 