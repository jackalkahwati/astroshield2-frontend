"""Mock system test script for ISS resupply mission trajectory prediction."""

import os
import json
import time
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
from typing import Dict, Any, List
from queue import Queue, Empty
from threading import Thread, Event
from unittest.mock import MagicMock
import gc

from models.monitoring.prediction_metrics import PerformanceTracker
from models.monitoring.real_time_monitor import RealTimeMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mock_system_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mock_system_test')

class MockKafka:
    """Mock Kafka implementation for testing."""
    
    def __init__(self):
        self.messages = Queue()
        self.responses = Queue()
        self.stop_event = Event()
        self.processing_thread = None
        self.processed_count = 0
        
    def start(self):
        """Start the mock Kafka processing."""
        self.processing_thread = Thread(target=self._process_messages)
        self.processing_thread.start()
    
    def stop(self):
        """Stop the mock Kafka processing."""
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join()
    
    def _process_messages(self):
        """Process messages in the queue."""
        while not self.stop_event.is_set():
            try:
                message = self.messages.get(timeout=1.0)
                # Simulate processing delay
                time.sleep(0.1)
                # Generate mock response
                response = self._generate_response(message)
                self.responses.put(response)
                self.processed_count += 1
                # Clear message from memory
                del message
                gc.collect()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _generate_response(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a mock prediction response."""
        return {
            "messageId": f"response_{message['messageId']}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predictionType": "REENTRY",
            "objectId": message['objectId'],
            "initialState": {
                "position": {"x": 6771000.0, "y": 0.0, "z": 0.0},
                "velocity": {"vx": -7.8, "vy": 0.0, "vz": 0.0},
                "epoch": datetime.now(timezone.utc).isoformat()
            },
            "predictedImpact": {
                "latitude": 28.5,
                "longitude": -80.5,
                "time": datetime.now(timezone.utc).isoformat(),
                "confidence": 0.95
            },
            "environmentalConditions": {
                "atmosphericDensity": 1.225,
                "temperature": 288.15,
                "windSpeed": 10.0,
                "windDirection": 270.0
            },
            "computationMetrics": {
                "processingTime": 0.1,
                "iterationCount": 1000,
                "convergenceStatus": "CONVERGED"
            }
        }

class MockSystemTest:
    """Runs mock system test for trajectory prediction."""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker(log_dir="logs/predictions")
        self.real_time_monitor = RealTimeMonitor(metrics_interval=1.0)
        
        # Test configuration
        self.test_duration = timedelta(minutes=1)  # Shorter duration for mock test
        self.message_interval = 0.5  # seconds
        self.max_retries = 3
        
        # Mock Kafka setup
        self.mock_kafka = MockKafka()
        self.producer = MagicMock()
        self.consumer = self.mock_kafka.responses  # Use responses queue instead of messages
    
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
    
    def _generate_prediction_filename(self, prediction_id: str) -> str:
        """Generate a safe filename for prediction data."""
        # Hash the prediction ID to ensure safe filename
        hash_obj = hashlib.md5(prediction_id.encode())
        return f"prediction_{hash_obj.hexdigest()[:8]}.json"
    
    def run_test(self):
        """Run the complete mock system test."""
        logger.info("Starting mock system test")
        
        try:
            # Start monitoring and mock Kafka
            self.real_time_monitor.start()
            self.mock_kafka.start()
            
            start_time = datetime.now()
            sequence = 0
            failed_messages = 0
            predictions: List[Dict[str, Any]] = []
            
            while datetime.now() - start_time < self.test_duration:
                # Generate and send test message
                message = self.generate_test_message(sequence)
                
                send_start = time.time()
                try:
                    # Send to mock Kafka
                    self.mock_kafka.messages.put(message)
                    send_time = time.time() - send_start
                    
                    logger.info(
                        f"Message {sequence} sent successfully: "
                        f"time={send_time:.3f}s"
                    )
                    
                    # Record performance metrics
                    self.real_time_monitor.record_prediction_time(send_time)
                    
                except Exception as e:
                    logger.error(f"Error sending message {sequence}: {e}")
                    failed_messages += 1
                    if failed_messages >= self.max_retries:
                        raise Exception(f"Too many failed messages: {failed_messages}")
                
                # Check for responses
                try:
                    # Try to get response with timeout
                    response = self.consumer.get(timeout=0.1)  # Reduced timeout
                    if isinstance(response, dict) and 'predictedImpact' in response:
                        logger.info(f"Received prediction: {response['messageId']}")
                        predictions.append(response)
                        
                        # Generate safe filename for prediction
                        filename = self._generate_prediction_filename(response['messageId'])
                        
                        # Record prediction metrics
                        self.performance_tracker.record_prediction(
                            prediction_id=response['messageId'],
                            predicted_impact=response['predictedImpact'],
                            initial_state=response['initialState'],
                            confidence=response['predictedImpact']['confidence'],
                            environmental_conditions=response['environmentalConditions'],
                            computation_time=response['computationMetrics']['processingTime']
                        )
                        
                        # Clear response from memory
                        del response
                        gc.collect()
                except Empty:
                    pass  # Expected when no responses are available
                except Exception as e:
                    logger.warning(f"Error processing response: {e}")
                
                sequence += 1
                time.sleep(self.message_interval)
                
                # Periodically clean up memory
                if sequence % 10 == 0:
                    gc.collect()
            
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
            logger.info(f"Predictions received: {len(predictions)}")
            logger.info(f"Messages processed by Kafka: {self.mock_kafka.processed_count}")
            
        except Exception as e:
            logger.error(f"Mock system test failed: {e}")
            raise
        
        finally:
            self.real_time_monitor.stop()
            self.mock_kafka.stop()
            logger.info("Mock system test completed")

def main():
    """Run the mock system test."""
    # Create log directories
    Path('logs').mkdir(exist_ok=True)
    Path('logs/predictions').mkdir(exist_ok=True)
    
    # Run test
    test = MockSystemTest()
    test.run_test()

if __name__ == "__main__":
    main() 