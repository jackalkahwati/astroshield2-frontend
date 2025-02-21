"""System test script for ISS resupply mission trajectory prediction."""

import os
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
from typing import Dict, Any
from dotenv import load_dotenv

from kafka import KafkaProducer, KafkaConsumer
from models.monitoring.prediction_metrics import PerformanceTracker
from models.monitoring.real_time_monitor import RealTimeMonitor

# Load test environment variables
test_env_path = Path(__file__).parent / '.env.test'
load_dotenv(test_env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('system_test')

class SystemTest:
    """Runs complete system test for trajectory prediction."""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker(log_dir="logs/predictions")
        self.real_time_monitor = RealTimeMonitor(
            metrics_interval=float(os.getenv('MESSAGE_INTERVAL_SECONDS', '1.0'))
        )
        
        # Test configuration
        self.test_duration = timedelta(
            minutes=int(os.getenv('TEST_DURATION_MINUTES', '5'))
        )
        self.message_interval = float(
            os.getenv('MESSAGE_INTERVAL_SECONDS', '1.0')
        )
        self.max_retries = int(os.getenv('MAX_RETRY_ATTEMPTS', '3'))
        
        # Kafka configuration
        self.producer = self._setup_kafka_producer()
        self.consumer = self._setup_kafka_consumer()
        
        logger.info(f"Test duration: {self.test_duration}")
        logger.info(f"Message interval: {self.message_interval}s")
        logger.info(f"Max retries: {self.max_retries}")
    
    def _setup_kafka_producer(self) -> KafkaProducer:
        """Setup Kafka producer with proper configuration."""
        try:
            return KafkaProducer(
                bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS').split(','),
                security_protocol=os.getenv('KAFKA_SECURITY_PROTOCOL'),
                sasl_mechanism=os.getenv('KAFKA_SASL_MECHANISM'),
                sasl_plain_username=os.getenv('KAFKA_PRODUCER_USERNAME'),
                sasl_plain_password=os.getenv('KAFKA_PRODUCER_PASSWORD'),
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                retries=self.max_retries
            )
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            raise
    
    def _setup_kafka_consumer(self) -> KafkaConsumer:
        """Setup Kafka consumer with proper configuration."""
        try:
            return KafkaConsumer(
                'trajectory.predictions',
                bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS').split(','),
                security_protocol=os.getenv('KAFKA_SECURITY_PROTOCOL'),
                sasl_mechanism=os.getenv('KAFKA_SASL_MECHANISM'),
                sasl_plain_username=os.getenv('KAFKA_CONSUMER_USERNAME'),
                sasl_plain_password=os.getenv('KAFKA_CONSUMER_PASSWORD'),
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id='test_consumer_group'
            )
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}")
            raise
    
    def generate_test_message(self, sequence: int) -> Dict[str, Any]:
        """Generate a test message with varying parameters."""
        base_altitude = 120000.0  # Starting altitude in meters
        descent_rate = 1000.0  # Meters per message
        
        return {
            "messageId": f"test_msg_{sequence}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "triggerType": "REENTRY_DETECTION",
            "objectId": "ISS_RESUPPLY_27",
            "measurements": {
                "altitude": base_altitude - (sequence * descent_rate),
                "velocity": 7800.0 - (sequence * 100.0),
                "dynamic_pressure": 45000.0 + (sequence * 1000.0)
            },
            "confidence": 0.95,
            "sensorId": "TEST_SENSOR_001"
        }
    
    def run_test(self):
        """Run the complete system test."""
        logger.info("Starting system test")
        
        try:
            # Start monitoring
            self.real_time_monitor.start()
            
            start_time = datetime.now()
            sequence = 0
            failed_messages = 0
            
            while datetime.now() - start_time < self.test_duration:
                # Generate and send test message
                message = self.generate_test_message(sequence)
                
                send_start = time.time()
                try:
                    future = self.producer.send('trajectory.triggers', value=message)
                    record_metadata = future.get(timeout=10)
                    send_time = time.time() - send_start
                    
                    logger.info(
                        f"Message {sequence} sent successfully: "
                        f"topic={record_metadata.topic}, "
                        f"partition={record_metadata.partition}, "
                        f"offset={record_metadata.offset}, "
                        f"time={send_time:.3f}s"
                    )
                    
                    # Record performance metrics
                    self.real_time_monitor.record_prediction_time(send_time)
                    
                except Exception as e:
                    logger.error(f"Error sending message {sequence}: {e}")
                    failed_messages += 1
                    if failed_messages >= self.max_retries:
                        raise Exception(f"Too many failed messages: {failed_messages}")
                
                # Check for responses with timeout
                try:
                    for msg in self.consumer:
                        prediction = msg.value
                        logger.info(f"Received prediction: {prediction['messageId']}")
                        
                        # Validate prediction
                        if 'predictedImpact' in prediction:
                            self.performance_tracker.record_prediction(
                                prediction_id=prediction['messageId'],
                                predicted_impact=prediction['predictedImpact'],
                                initial_state=prediction.get('initialState', {}),
                                confidence=prediction['predictedImpact']['confidence'],
                                environmental_conditions=prediction.get('environmentalConditions', {}),
                                computation_time=prediction.get('computationMetrics', {}).get('processingTime', 0.0)
                            )
                        break  # Process one message at a time
                except Exception as e:
                    logger.warning(f"Error processing response: {e}")
                
                sequence += 1
                time.sleep(self.message_interval)
            
            # Get final performance summary
            summary = self.real_time_monitor.get_performance_summary()
            logger.info("Performance Summary:")
            logger.info(json.dumps(summary, indent=2))
            
            # Get validation results
            validation = self.performance_tracker.validate_against_known_cases()
            logger.info("Validation Results:")
            logger.info(json.dumps(validation, indent=2))
            
            # Log test statistics
            logger.info(f"Total messages sent: {sequence}")
            logger.info(f"Failed messages: {failed_messages}")
            
        except Exception as e:
            logger.error(f"System test failed: {e}")
            raise
        
        finally:
            self.real_time_monitor.stop()
            self.producer.close()
            self.consumer.close()
            logger.info("System test completed")

def main():
    """Run the system test."""
    # Create log directories
    Path('logs').mkdir(exist_ok=True)
    Path('logs/predictions').mkdir(exist_ok=True)
    
    # Run test
    test = SystemTest()
    test.run_test()

if __name__ == "__main__":
    main() 