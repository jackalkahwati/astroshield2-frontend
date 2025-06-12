#!/usr/bin/env python3
"""
UDL (Unified Data Library) Connectivity Test
Tests authentication, data retrieval, and all major UDL endpoints
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from asttroshield.api_client.udl_client import UDLClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UDLConnectivityTest:
    """Test UDL connectivity and data retrieval capabilities."""
    
    def __init__(self):
        """Initialize the UDL test client."""
        # Get UDL configuration from environment variables
        self.base_url = os.environ.get("UDL_BASE_URL", "https://unifieddatalibrary.com")
        self.api_key = os.environ.get("UDL_API_KEY")
        self.username = os.environ.get("UDL_USERNAME")
        self.password = os.environ.get("UDL_PASSWORD")
        
        # Initialize client if we have credentials
        if self.api_key:
            self.client = UDLClient(self.base_url, self.api_key)
            self.auth_method = "API Key"
        elif self.username and self.password:
            # For username/password auth, we'll need to modify the client
            self.client = UDLClient(self.base_url, "dummy_key")  # Will be replaced with token
            self.auth_method = "Username/Password"
        else:
            self.client = None
            self.auth_method = "None"
            
        self.test_results = {}
        
    def check_credentials(self) -> bool:
        """Check if UDL credentials are configured."""
        logger.info("=== UDL Credential Check ===")
        
        has_api_key = bool(self.api_key)
        has_username_password = bool(self.username and self.password)
        
        logger.info(f"UDL Base URL: {self.base_url}")
        logger.info(f"API Key: {'✓ Configured' if has_api_key else '✗ Not configured'}")
        logger.info(f"Username: {'✓ Configured' if self.username else '✗ Not configured'}")
        logger.info(f"Password: {'✓ Configured' if self.password else '✗ Not configured'}")
        logger.info(f"Authentication Method: {self.auth_method}")
        
        if not (has_api_key or has_username_password):
            logger.error("No UDL credentials configured!")
            logger.error("Please set either:")
            logger.error("  - UDL_API_KEY environment variable")
            logger.error("  - UDL_USERNAME and UDL_PASSWORD environment variables")
            return False
            
        return True
        
    def test_basic_connectivity(self) -> bool:
        """Test basic connectivity to UDL."""
        logger.info("\n=== Basic Connectivity Test ===")
        
        if not self.client:
            logger.error("No UDL client available")
            return False
            
        try:
            # Test a simple endpoint that should always be available
            result = self.client.get_system_health_summary()
            logger.info("✓ Successfully connected to UDL")
            logger.info(f"Response keys: {list(result.keys())}")
            self.test_results['basic_connectivity'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to connect to UDL: {e}")
            self.test_results['basic_connectivity'] = False
            return False
            
    def test_space_weather_data(self) -> bool:
        """Test space weather data retrieval."""
        logger.info("\n=== Space Weather Data Test ===")
        
        try:
            # Get current space weather conditions
            space_weather = self.client.get_space_weather_data()
            logger.info("✓ Successfully retrieved space weather data")
            logger.info(f"Space weather keys: {list(space_weather.keys()) if space_weather else 'No data'}")
            
            # Get radiation belt data
            radiation_data = self.client.get_radiation_belt_data()
            logger.info("✓ Successfully retrieved radiation belt data")
            logger.info(f"Radiation data keys: {list(radiation_data.keys()) if radiation_data else 'No data'}")
            
            self.test_results['space_weather'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to retrieve space weather data: {e}")
            self.test_results['space_weather'] = False
            return False
            
    def test_object_tracking_data(self) -> bool:
        """Test object tracking and orbital data retrieval."""
        logger.info("\n=== Object Tracking Data Test ===")
        
        # Test with a common satellite ID (ISS)
        test_object_id = "25544"  # International Space Station
        
        try:
            # Get current state vector
            state_vector = self.client.get_state_vector(test_object_id)
            logger.info(f"✓ Successfully retrieved state vector for object {test_object_id}")
            logger.info(f"State vector keys: {list(state_vector.keys()) if state_vector else 'No data'}")
            
            # Get ELSET data
            elset_data = self.client.get_elset_data(test_object_id)
            logger.info(f"✓ Successfully retrieved ELSET data for object {test_object_id}")
            logger.info(f"ELSET keys: {list(elset_data.keys()) if elset_data else 'No data'}")
            
            # Get object health status
            health_status = self.client.get_object_health(test_object_id)
            logger.info(f"✓ Successfully retrieved health status for object {test_object_id}")
            logger.info(f"Health status keys: {list(health_status.keys()) if health_status else 'No data'}")
            
            self.test_results['object_tracking'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to retrieve object tracking data: {e}")
            self.test_results['object_tracking'] = False
            return False
            
    def test_conjunction_data(self) -> bool:
        """Test conjunction analysis data retrieval."""
        logger.info("\n=== Conjunction Data Test ===")
        
        test_object_id = "25544"  # ISS
        
        try:
            # Get conjunction data
            conjunction_data = self.client.get_conjunction_data(test_object_id)
            logger.info(f"✓ Successfully retrieved conjunction data for object {test_object_id}")
            logger.info(f"Conjunction keys: {list(conjunction_data.keys()) if conjunction_data else 'No data'}")
            
            # Get historical conjunction data (last 24 hours)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            conjunction_history = self.client.get_conjunction_history(
                test_object_id,
                start_time.isoformat(),
                end_time.isoformat()
            )
            logger.info(f"✓ Successfully retrieved conjunction history for object {test_object_id}")
            logger.info(f"History entries: {len(conjunction_history) if isinstance(conjunction_history, list) else 'Not a list'}")
            
            self.test_results['conjunction_data'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to retrieve conjunction data: {e}")
            self.test_results['conjunction_data'] = False
            return False
            
    def test_rf_interference_data(self) -> bool:
        """Test RF interference data retrieval."""
        logger.info("\n=== RF Interference Data Test ===")
        
        try:
            # Test RF interference data with a common frequency range
            frequency_range = {
                'min': 2400.0,  # 2.4 GHz
                'max': 2500.0   # 2.5 GHz
            }
            
            rf_data = self.client.get_rf_interference(frequency_range)
            logger.info("✓ Successfully retrieved RF interference data")
            logger.info(f"RF data keys: {list(rf_data.keys()) if rf_data else 'No data'}")
            
            # Check if we have RF emitters in the response
            if 'rfEmitters' in rf_data:
                emitter_count = len(rf_data['rfEmitters'])
                logger.info(f"Found {emitter_count} RF emitters in frequency range")
            
            self.test_results['rf_interference'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to retrieve RF interference data: {e}")
            self.test_results['rf_interference'] = False
            return False
            
    def test_new_udl_133_services(self) -> bool:
        """Test new UDL 1.33.0 services."""
        logger.info("\n=== UDL 1.33.0 New Services Test ===")
        
        try:
            # Test EMI Reports
            emi_reports = self.client.get_emi_reports()
            logger.info("✓ Successfully retrieved EMI reports")
            logger.info(f"EMI reports keys: {list(emi_reports.keys()) if emi_reports else 'No data'}")
            
            # Test Laser Deconflict Requests
            laser_requests = self.client.get_laser_deconflict_requests()
            logger.info("✓ Successfully retrieved laser deconflict requests")
            logger.info(f"Laser requests keys: {list(laser_requests.keys()) if laser_requests else 'No data'}")
            
            # Test Deconflict Sets
            deconflict_sets = self.client.get_deconflict_sets()
            logger.info("✓ Successfully retrieved deconflict sets")
            logger.info(f"Deconflict sets keys: {list(deconflict_sets.keys()) if deconflict_sets else 'No data'}")
            
            # Test ECPEDR Data
            ecpedr_data = self.client.get_ecpedr_data()
            logger.info("✓ Successfully retrieved ECPEDR data")
            logger.info(f"ECPEDR keys: {list(ecpedr_data.keys()) if ecpedr_data else 'No data'}")
            
            self.test_results['udl_133_services'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to retrieve UDL 1.33.0 services: {e}")
            self.test_results['udl_133_services'] = False
            return False
            
    def test_sensor_operations(self) -> bool:
        """Test sensor-related operations."""
        logger.info("\n=== Sensor Operations Test ===")
        
        test_sensor_id = "SENSOR_001"
        
        try:
            # Test sensor heartbeat
            heartbeat_result = self.client.send_sensor_heartbeat(
                test_sensor_id,
                "OPERATIONAL",
                {"test": True, "timestamp": datetime.utcnow().isoformat()}
            )
            logger.info(f"✓ Successfully sent sensor heartbeat for {test_sensor_id}")
            logger.info(f"Heartbeat result: {heartbeat_result}")
            
            # Test notification creation
            notification_result = self.client.create_notification(
                "TEST",
                "UDL connectivity test notification",
                "INFO"
            )
            logger.info("✓ Successfully created test notification")
            logger.info(f"Notification result: {notification_result}")
            
            self.test_results['sensor_operations'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed sensor operations test: {e}")
            self.test_results['sensor_operations'] = False
            return False
            
    def test_batch_operations(self) -> bool:
        """Test batch data retrieval operations."""
        logger.info("\n=== Batch Operations Test ===")
        
        try:
            # Test multiple object status retrieval
            test_object_ids = ["25544", "20580", "27424"]  # ISS, NOAA-18, COSMOS 2251 DEB
            
            batch_status = self.client.get_multiple_object_status(test_object_ids)
            logger.info(f"✓ Successfully retrieved batch status for {len(test_object_ids)} objects")
            logger.info(f"Batch status keys: {list(batch_status.keys()) if batch_status else 'No data'}")
            
            # Test system health summary
            health_summary = self.client.get_system_health_summary()
            logger.info("✓ Successfully retrieved system health summary")
            logger.info(f"Health summary endpoints: {list(health_summary.keys()) if health_summary else 'No data'}")
            
            self.test_results['batch_operations'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed batch operations test: {e}")
            self.test_results['batch_operations'] = False
            return False
            
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all UDL connectivity tests."""
        logger.info("🚀 Starting UDL Connectivity Tests")
        logger.info("=" * 50)
        
        # Check credentials first
        if not self.check_credentials():
            return {"credentials": False}
            
        # Run all tests
        tests = [
            ("Basic Connectivity", self.test_basic_connectivity),
            ("Space Weather Data", self.test_space_weather_data),
            ("Object Tracking Data", self.test_object_tracking_data),
            ("Conjunction Data", self.test_conjunction_data),
            ("RF Interference Data", self.test_rf_interference_data),
            ("UDL 1.33.0 Services", self.test_new_udl_133_services),
            ("Sensor Operations", self.test_sensor_operations),
            ("Batch Operations", self.test_batch_operations),
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                logger.error(f"✗ {test_name} failed with exception: {e}")
                self.test_results[test_name.lower().replace(" ", "_")] = False
                
        return self.test_results
        
    def print_summary(self):
        """Print test results summary."""
        logger.info("\n" + "=" * 50)
        logger.info("🔍 UDL Connectivity Test Summary")
        logger.info("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        if total_tests > 0:
            logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        else:
            logger.info("Success Rate: N/A (no tests run)")
        
        logger.info("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"  {test_name.replace('_', ' ').title()}: {status}")
            
        if passed_tests == total_tests:
            logger.info("\n🎉 All UDL connectivity tests passed!")
            logger.info("AstroShield is fully connected to the UDL ecosystem.")
        else:
            logger.info(f"\n⚠️  {total_tests - passed_tests} test(s) failed.")
            logger.info("Check the logs above for specific error details.")
            
        return passed_tests == total_tests


def main():
    """Main function to run UDL connectivity tests."""
    # Check if we're in mock mode
    mock_mode = os.environ.get("UDL_MOCK_MODE", "false").lower() == "true"
    
    if mock_mode:
        logger.info("🧪 Running in MOCK MODE")
        logger.info("Set UDL_MOCK_MODE=false to test real UDL connection")
        
        # Set mock credentials for testing
        os.environ["UDL_API_KEY"] = "mock_api_key_for_testing"
        os.environ["UDL_BASE_URL"] = "https://mock-udl.example.com"
    
    # Create and run tests
    test_runner = UDLConnectivityTest()
    results = test_runner.run_all_tests()
    success = test_runner.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 