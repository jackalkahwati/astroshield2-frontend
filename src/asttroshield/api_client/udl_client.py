"""UDL Client for interacting with the Unified Data Library."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import requests

class UDLClient:
    """Client for interacting with the UDL API."""
    
    def __init__(self, base_url: str, api_key: str):
        """Initialize the UDL client."""
        self.base_url = base_url
        self._api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
    @property
    def api_key(self) -> Optional[str]:
        """Get the API key."""
        return self._api_key
        
    def get_sensor_status(self, sensor_id: str) -> Dict[str, Any]:
        """Get the current status of a sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            
        Returns:
            Dict containing sensor status information
        """
        endpoint = f'/udl/sensor/{sensor_id}'
        response = self.session.get(f'{self.base_url}{endpoint}')
        response.raise_for_status()
        return response.json()

    def send_sensor_heartbeat(self, sensor_id: str, status: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Send a heartbeat for a sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            status: Current status of the sensor (e.g., "OPERATIONAL")
            metadata: Additional metadata about the sensor's state
            
        Returns:
            Dict containing the response from the API
        """
        endpoint = '/udl/linkstatus'
        data = {
            'timeHeartbeat': datetime.utcnow().isoformat(),
            'idSensor': sensor_id,
            'status': status,
            'metadata': metadata
        }
        response = self.session.post(f'{self.base_url}{endpoint}', json=data)
        response.raise_for_status()
        return response.json()

    def get_link_status_history(self, sensor_id: str, start_time: str, end_time: str) -> Dict[str, Any]:
        """Get historical link status data for a sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            start_time: ISO format start time
            end_time: ISO format end time
            
        Returns:
            Dict containing historical link status data
        """
        endpoint = '/udl/linkstatus/history'
        params = {
            'idSensor': sensor_id,
            'startTime': start_time,
            'endTime': end_time
        }
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    def create_notification(self, msg_type: str, message: str, severity: str = 'INFO') -> Dict[str, Any]:
        """Create a new notification in the system.
        
        Args:
            msg_type: Type of message
            message: Notification message content
            severity: Severity level of the notification
            
        Returns:
            Dict containing the created notification
        """
        endpoint = '/udl/notification'
        data = {
            'msgType': msg_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat()
        }
        response = self.session.post(f'{self.base_url}{endpoint}', json=data)
        response.raise_for_status()
        return response.json()

    def get_sensor_maintenance(self, sensor_id: str) -> Dict[str, Any]:
        """Get maintenance information for a sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            
        Returns:
            Dict containing sensor maintenance information
        """
        endpoint = '/udl/sensormaintenance'
        params = {'idSensor': sensor_id}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    def get_sensor_calibration(self, sensor_id: str) -> Dict[str, Any]:
        """Get calibration information for a sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            
        Returns:
            Dict containing sensor calibration information
        """
        endpoint = '/udl/sensorcalibration'
        params = {'idSensor': sensor_id}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    # Space Weather Indicators
    def get_space_weather_data(self) -> Dict[str, Any]:
        """Get current space weather conditions."""
        endpoint = '/udl/sgi'  # Solar/Geomagnetic Index
        response = self.session.get(f'{self.base_url}{endpoint}')
        response.raise_for_status()
        return response.json()

    def get_radiation_belt_data(self) -> Dict[str, Any]:
        """Get radiation belt data."""
        endpoint = '/udl/sgi'
        params = {'dataType': 'RADIATION_BELT'}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    # Conjunction Data
    def get_conjunction_data(self, object_id: str) -> Dict[str, Any]:
        """Get conjunction data for a specific object."""
        endpoint = '/udl/conjunction'
        params = {'objectId': object_id}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    def get_conjunction_history(self, object_id: str, start_time: str, end_time: str) -> Dict[str, Any]:
        """Get historical conjunction data."""
        endpoint = '/udl/conjunction/history'
        params = {
            'objectId': object_id,
            'startTime': start_time,
            'endTime': end_time
        }
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    # RF Interference
    def get_rf_interference(self, frequency_range: Dict[str, float]) -> Dict[str, Any]:
        """Get RF interference data."""
        endpoint = '/udl/rfemitter'
        params = {
            'minFreq': frequency_range.get('min'),
            'maxFreq': frequency_range.get('max')
        }
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    # Orbital Data
    def get_state_vector(self, object_id: str) -> Dict[str, Any]:
        """Get current state vector for an object."""
        endpoint = '/udl/statevector/current'
        params = {'objectId': object_id}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    def get_state_vector_history(self, object_id: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """Get historical state vectors for an object.
        
        Args:
            object_id: Unique identifier for the object
            start_time: ISO format start time
            end_time: ISO format end time
            
        Returns:
            List of state vectors over the specified time period
        """
        endpoint = '/udl/statevector/history'
        params = {
            'objectId': object_id,
            'startTime': start_time,
            'endTime': end_time
        }
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    def get_elset_data(self, object_id: str) -> Dict[str, Any]:
        """Get current ELSET data for an object."""
        endpoint = '/udl/elset/current'
        params = {'objectId': object_id}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    # Maneuver Detection
    def get_maneuver_data(self, object_id: str) -> Dict[str, Any]:
        """Get maneuver data for an object."""
        endpoint = '/udl/maneuver'
        params = {'objectId': object_id}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    # Link Status and Communications
    def get_link_status(self, object_id: str) -> Dict[str, Any]:
        """Get current link status for an object."""
        endpoint = '/udl/linkstatus'
        params = {'objectId': object_id}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    def get_comm_status(self, object_id: str) -> Dict[str, Any]:
        """Get communications status for an object."""
        endpoint = '/udl/comm'
        params = {'objectId': object_id}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    # Object Status and Health
    def get_object_health(self, object_id: str) -> Dict[str, Any]:
        """Get health status of an object."""
        endpoint = '/udl/onorbit'
        params = {'objectId': object_id}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    def get_object_events(self, object_id: str) -> Dict[str, Any]:
        """Get events related to an object."""
        endpoint = '/udl/onorbitevent'
        params = {'objectId': object_id}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    # Batch Operations
    def get_multiple_object_status(self, object_ids: List[str]) -> Dict[str, Any]:
        """Get status for multiple objects at once."""
        endpoint = '/udl/onorbit'
        params = {'objectIds': ','.join(object_ids)}
        response = self.session.get(f'{self.base_url}{endpoint}', params=params)
        response.raise_for_status()
        return response.json()

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        endpoints = [
            '/udl/sgi',  # Space weather
            '/udl/conjunction',  # Active conjunctions
            '/udl/linkstatus',  # Communication status
            '/udl/rfemitter'  # RF interference
        ]
        
        summary = {}
        for endpoint in endpoints:
            try:
                response = self.session.get(f'{self.base_url}{endpoint}')
                response.raise_for_status()
                summary[endpoint] = response.json()
            except Exception as e:
                summary[endpoint] = {'error': str(e)}
        
        return summary 