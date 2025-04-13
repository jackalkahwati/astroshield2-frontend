"""Atmospheric Transit Detection and Tracking Module."""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .trajectory_predictor import TrajectoryPredictor

class TransitType(Enum):
    """Types of atmospheric transit objects."""
    SPACE_LAUNCH = "space_launch"
    REENTRY = "reentry"
    UNKNOWN = "unknown"

@dataclass
class GeophysicalData:
    """Container for geophysical measurements."""
    time: datetime
    ionospheric_tec: float  # Total Electron Content
    magnetic_field: Dict[str, float]  # B-field components
    location: Dict[str, float]  # lat, lon, alt
    confidence: float

@dataclass
class SDRMeasurement:
    """Container for SDR measurements."""
    time: datetime
    frequency: float
    power: float
    doppler_shift: float
    location: Dict[str, float]  # lat, lon, alt
    confidence: float

@dataclass
class TransitObject:
    """Detected transit object characteristics."""
    time_first_detection: datetime
    location: Dict[str, float]  # lat, lon, alt
    velocity: Dict[str, float]  # vx, vy, vz
    transit_type: TransitType
    confidence: float
    predicted_impact: Optional[Dict[str, Any]] = None

class UDLDataIntegrator:
    """Integrates UDL data sources with atmospheric transit detection."""
    
    def __init__(self, udl_client: 'UDLClient', config: Dict[str, Any]):
        """Initialize the UDL data integrator.
        
        Args:
            udl_client: UDL API client instance
            config: Configuration parameters
        """
        self.udl_client = udl_client
        self.config = config
        
    def get_ionospheric_data(self) -> List[GeophysicalData]:
        """Get ionospheric data from UDL space weather endpoint."""
        space_weather = self.udl_client.get_space_weather_data()
        
        # Convert UDL data to GeophysicalData format
        data = []
        if 'tec_data' in space_weather:
            for measurement in space_weather['tec_data']:
                data.append(GeophysicalData(
                    time=datetime.fromisoformat(measurement['time']),
                    ionospheric_tec=measurement['tec_value'],
                    magnetic_field={
                        'x': measurement.get('b_field_x', 0.0),
                        'y': measurement.get('b_field_y', 0.0),
                        'z': measurement.get('b_field_z', 0.0)
                    },
                    location={
                        'lat': measurement['latitude'],
                        'lon': measurement['longitude'],
                        'alt': measurement['altitude']
                    },
                    confidence=measurement.get('confidence', 0.8)
                ))
        
        return data
    
    def get_sdr_data(self, frequency_range: Dict[str, float]) -> List[SDRMeasurement]:
        """Get SDR data from UDL RF interference endpoint."""
        rf_data = self.udl_client.get_rf_interference(frequency_range)
        
        # Convert UDL data to SDRMeasurement format
        data = []
        if 'measurements' in rf_data:
            for measurement in rf_data['measurements']:
                data.append(SDRMeasurement(
                    time=datetime.fromisoformat(measurement['time']),
                    frequency=measurement['frequency'],
                    power=measurement['power_level'],
                    doppler_shift=measurement.get('doppler_shift', 0.0),
                    location={
                        'lat': measurement['latitude'],
                        'lon': measurement['longitude'],
                        'alt': measurement['altitude']
                    },
                    confidence=measurement.get('confidence', 0.8)
                ))
        
        return data
    
    def get_state_vector_data(self, object_id: str) -> Dict[str, Any]:
        """Get state vector data for trajectory analysis."""
        current_state = self.udl_client.get_state_vector(object_id)
        
        # Get recent history for velocity analysis
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        history = self.udl_client.get_state_vector_history(
            object_id,
            start_time.isoformat(),
            end_time.isoformat()
        )
        
        return {
            'current': current_state,
            'history': history
        }
    
    def analyze_trajectory(self, state_data: Dict[str, Any]) -> Optional[TransitObject]:
        """Analyze state vector data for transit detection."""
        if not state_data.get('current') or not state_data.get('history'):
            return None
            
        current = state_data['current']
        history = state_data['history']
        
        # Calculate velocity components
        if len(history) >= 2:
            dt = (datetime.fromisoformat(history[-1]['epoch']) - 
                 datetime.fromisoformat(history[0]['epoch'])).total_seconds()
            
            if dt > 0:
                dx = history[-1]['xpos'] - history[0]['xpos']
                dy = history[-1]['ypos'] - history[0]['ypos']
                dz = history[-1]['zpos'] - history[0]['zpos']
                
                vx = dx / dt
                vy = dy / dt
                vz = dz / dt
                
                # Convert to km/s
                velocity = {
                    'vx': vx / 1000,
                    'vy': vy / 1000,
                    'vz': vz / 1000
                }
                
                # Calculate altitude
                x = current['xpos']
                y = current['ypos']
                z = current['zpos']
                r = np.sqrt(x**2 + y**2 + z**2)
                alt = (r - 6371000) / 1000  # Convert to km
                
                if self.config['altitude_range']['min'] <= alt <= self.config['altitude_range']['max']:
                    return TransitObject(
                        time_first_detection=datetime.fromisoformat(history[0]['epoch']),
                        location={
                            'lat': np.degrees(np.arcsin(z/r)),
                            'lon': np.degrees(np.arctan2(y, x)),
                            'alt': alt
                        },
                        velocity=velocity,
                        transit_type=TransitType.UNKNOWN,
                        confidence=0.9 if current.get('uct', False) else 0.7
                    )
        
        return None

class AtmosphericTransitDetector:
    """Detector for objects transiting through the upper atmosphere."""
    
    def __init__(self, config: Dict[str, Any], udl_client: Optional['UDLClient'] = None):
        """Initialize the detector.
        
        Args:
            config: Configuration parameters
            udl_client: Optional UDL client for data integration
        """
        self.config = config
        self.udl_integrator = UDLDataIntegrator(udl_client, config) if udl_client else None
        self.MIN_ALT = config['altitude_range']['min']  # 30 km
        self.MAX_ALT = config['altitude_range']['max']  # 300 km
        self.VELOCITY_THRESHOLD = config['velocity_threshold']  # km/s
        self.CONFIDENCE_THRESHOLD = config['confidence_threshold']
        self.trajectory_predictor = TrajectoryPredictor(config)
        
    def process_geophysical_data(self, data: List[GeophysicalData]) -> List[Dict[str, Any]]:
        """Process geophysical measurements for transit detection.
        
        Args:
            data: List of geophysical measurements
            
        Returns:
            List of potential transit detections
        """
        detections = []
        
        # Group measurements by time windows
        time_windows = self._group_by_time_window(data)
        
        for window in time_windows:
            # Analyze TEC perturbations
            tec_anomalies = self._detect_tec_anomalies(window)
            
            # Analyze magnetic field disturbances
            magnetic_anomalies = self._detect_magnetic_anomalies(window)
            
            # Correlate anomalies
            if tec_anomalies and magnetic_anomalies:
                detection = self._correlate_geophysical_anomalies(
                    tec_anomalies,
                    magnetic_anomalies
                )
                if detection:
                    detections.append(detection)
        
        return detections
    
    def process_sdr_data(self, data: List[SDRMeasurement]) -> List[Dict[str, Any]]:
        """Process SDR measurements for transit detection.
        
        Args:
            data: List of SDR measurements
            
        Returns:
            List of potential transit detections
        """
        detections = []
        
        # Group measurements by frequency bands
        frequency_groups = self._group_by_frequency(data)
        
        for group in frequency_groups:
            # Analyze Doppler shifts
            doppler_tracks = self._analyze_doppler_tracks(group)
            
            # Analyze signal power variations
            power_tracks = self._analyze_power_variations(group)
            
            # Correlate tracks
            if doppler_tracks and power_tracks:
                detection = self._correlate_sdr_tracks(
                    doppler_tracks,
                    power_tracks
                )
                if detection:
                    detections.append(detection)
        
        return detections
    
    def detect_transits(self, 
                       geophysical_data: List[GeophysicalData],
                       sdr_data: List[SDRMeasurement]) -> List[TransitObject]:
        """Detect atmospheric transits using combined data sources.
        
        Args:
            geophysical_data: List of geophysical measurements
            sdr_data: List of SDR measurements
            
        Returns:
            List of detected transit objects
        """
        # Process each data source
        geo_detections = self.process_geophysical_data(geophysical_data)
        sdr_detections = self.process_sdr_data(sdr_data)
        
        # Correlate detections across sources
        transit_objects = self._correlate_detections(
            geo_detections,
            sdr_detections
        )
        
        # Filter by confidence
        transit_objects = [
            obj for obj in transit_objects
            if obj.confidence >= self.CONFIDENCE_THRESHOLD
        ]
        
        # Classify transit types
        for obj in transit_objects:
            obj.transit_type = self._classify_transit_type(obj)
            
            # For Earth-bound objects, predict impact
            if obj.transit_type == TransitType.REENTRY:
                obj.predicted_impact = self._predict_impact(obj)
        
        return transit_objects
    
    def detect_transits_with_udl(self, object_id: Optional[str] = None) -> List[TransitObject]:
        """Detect atmospheric transits using UDL data.
        
        Args:
            object_id: Optional object ID for state vector analysis
            
        Returns:
            List of detected transit objects
        """
        if not self.udl_integrator:
            return []
            
        # Get data from UDL
        geo_data = self.udl_integrator.get_ionospheric_data()
        
        sdr_data = self.udl_integrator.get_sdr_data({
            'min': self.config['sdr']['frequency_bands'][0]['min_freq'],
            'max': self.config['sdr']['frequency_bands'][-1]['max_freq']
        })
        
        # Process geophysical and SDR data
        geo_detections = self.process_geophysical_data(geo_data)
        sdr_detections = self.process_sdr_data(sdr_data)
        
        # Correlate detections
        transit_objects = self._correlate_detections(
            geo_detections,
            sdr_detections
        )
        
        # If object_id provided, analyze state vectors
        if object_id:
            state_data = self.udl_integrator.get_state_vector_data(object_id)
            state_object = self.udl_integrator.analyze_trajectory(state_data)
            
            if state_object:
                transit_objects.append(state_object)
        
        # Classify transit types
        for obj in transit_objects:
            obj.transit_type = self._classify_transit_type(obj)
            
            # For Earth-bound objects, predict impact
            if obj.transit_type == TransitType.REENTRY:
                obj.predicted_impact = self._predict_impact(obj)
        
        return transit_objects
    
    def _group_by_time_window(self, data: List[GeophysicalData]) -> List[List[GeophysicalData]]:
        """Group measurements into time windows for analysis."""
        if not data:
            return []
            
        windows = []
        current_window = []
        window_start = data[0].time
        
        for measurement in data:
            time_diff = (measurement.time - window_start).total_seconds()
            
            if time_diff <= self.config['time']['window_size']:
                current_window.append(measurement)
            else:
                if current_window:
                    windows.append(current_window)
                current_window = [measurement]
                window_start = measurement.time
        
        if current_window:
            windows.append(current_window)
            
        return windows
    
    def _group_by_frequency(self, data: List[SDRMeasurement]) -> List[List[SDRMeasurement]]:
        """Group SDR measurements by frequency bands."""
        if not data:
            return []
            
        frequency_bands = self.config['sdr']['frequency_bands']
        grouped_data = {band['name']: [] for band in frequency_bands}
        
        for measurement in data:
            for band in frequency_bands:
                if band['min_freq'] <= measurement.frequency <= band['max_freq']:
                    grouped_data[band['name']].append(measurement)
                    break
        
        return [group for group in grouped_data.values() if group]
    
    def _detect_tec_anomalies(self, data: List[GeophysicalData]) -> List[Dict[str, Any]]:
        """Detect anomalies in Total Electron Content."""
        if not data:
            return []
            
        anomalies = []
        tec_values = [d.ionospheric_tec for d in data]
        tec_mean = np.mean(tec_values)
        tec_std = np.std(tec_values)
        
        # Detect significant deviations (>3 sigma)
        for i, measurement in enumerate(data):
            if abs(measurement.ionospheric_tec - tec_mean) > 3 * tec_std:
                # Check if anomaly duration meets minimum threshold
                duration = 1
                for j in range(i + 1, len(data)):
                    if abs(data[j].ionospheric_tec - tec_mean) > 3 * tec_std:
                        duration += 1
                    else:
                        break
                
                if duration >= self.config['ionospheric']['min_perturbation_duration']:
                    anomalies.append({
                        'time': measurement.time,
                        'magnitude': abs(measurement.ionospheric_tec - tec_mean),
                        'duration': duration,
                        'location': measurement.location,
                        'confidence': measurement.confidence
                    })
        
        return anomalies
    
    def _detect_magnetic_anomalies(self, data: List[GeophysicalData]) -> List[Dict[str, Any]]:
        """Detect anomalies in magnetic field measurements."""
        if not data:
            return []
            
        anomalies = []
        field_threshold = self.config['magnetic']['field_threshold']
        
        for i, measurement in enumerate(data):
            # Calculate field magnitude
            field = measurement.magnetic_field
            magnitude = np.sqrt(field['x']**2 + field['y']**2 + field['z']**2)
            
            if magnitude > field_threshold:
                # Check anomaly duration
                duration = 1
                for j in range(i + 1, len(data)):
                    next_field = data[j].magnetic_field
                    next_mag = np.sqrt(next_field['x']**2 + next_field['y']**2 + next_field['z']**2)
                    if next_mag > field_threshold:
                        duration += 1
                    else:
                        break
                
                if duration >= self.config['magnetic']['min_disturbance_duration']:
                    anomalies.append({
                        'time': measurement.time,
                        'magnitude': magnitude,
                        'duration': duration,
                        'location': measurement.location,
                        'confidence': measurement.confidence
                    })
        
        return anomalies
    
    def _analyze_doppler_tracks(self, data: List[SDRMeasurement]) -> List[Dict[str, Any]]:
        """Analyze Doppler shift patterns for moving objects."""
        if not data:
            return []
            
        tracks = []
        doppler_threshold = self.config['sdr']['doppler_threshold']
        
        for i, measurement in enumerate(data):
            if abs(measurement.doppler_shift) > doppler_threshold:
                # Track duration and characteristics
                duration = 1
                max_shift = abs(measurement.doppler_shift)
                shift_pattern = [measurement.doppler_shift]
                
                for j in range(i + 1, len(data)):
                    if abs(data[j].doppler_shift) > doppler_threshold:
                        duration += 1
                        max_shift = max(max_shift, abs(data[j].doppler_shift))
                        shift_pattern.append(data[j].doppler_shift)
                    else:
                        break
                
                if duration >= self.config['sdr']['min_track_duration']:
                    # Calculate velocity from Doppler shift
                    freq = measurement.frequency
                    velocity_radial = max_shift * 3e8 / freq  # v = Δf * c / f
                    
                    tracks.append({
                        'time': measurement.time,
                        'frequency': freq,
                        'max_shift': max_shift,
                        'duration': duration,
                        'velocity_radial': velocity_radial,
                        'location': measurement.location,
                        'confidence': measurement.confidence,
                        'shift_pattern': shift_pattern
                    })
        
        return tracks
    
    def _analyze_power_variations(self, data: List[SDRMeasurement]) -> List[Dict[str, Any]]:
        """Analyze signal power variations for moving objects."""
        if not data:
            return []
            
        variations = []
        power_threshold = self.config['sdr']['power_threshold']
        
        for i, measurement in enumerate(data):
            if measurement.power > power_threshold:
                # Track power variation pattern
                duration = 1
                power_pattern = [measurement.power]
                max_power = measurement.power
                
                for j in range(i + 1, len(data)):
                    if data[j].power > power_threshold:
                        duration += 1
                        power_pattern.append(data[j].power)
                        max_power = max(max_power, data[j].power)
                    else:
                        break
                
                if duration >= self.config['sdr']['min_track_duration']:
                    variations.append({
                        'time': measurement.time,
                        'max_power': max_power,
                        'duration': duration,
                        'location': measurement.location,
                        'confidence': measurement.confidence,
                        'power_pattern': power_pattern
                    })
        
        return variations
    
    def _correlate_geophysical_anomalies(self,
                                        tec_anomalies: List[Dict[str, Any]],
                                        magnetic_anomalies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Correlate TEC and magnetic field anomalies."""
        if not tec_anomalies or not magnetic_anomalies:
            return None
            
        max_time_diff = self.config['correlation']['max_time_difference']
        max_pos_diff = self.config['correlation']['max_position_difference']
        
        for tec in tec_anomalies:
            for mag in magnetic_anomalies:
                # Check temporal correlation
                time_diff = abs((tec['time'] - mag['time']).total_seconds())
                if time_diff > max_time_diff:
                    continue
                
                # Check spatial correlation
                pos1 = tec['location']
                pos2 = mag['location']
                pos_diff = np.sqrt(
                    (pos1['lat'] - pos2['lat'])**2 +
                    (pos1['lon'] - pos2['lon'])**2 +
                    (pos1['alt'] - pos2['alt'])**2
                )
                
                if pos_diff > max_pos_diff:
                    continue
                
                # Calculate combined confidence
                confidence = min(tec['confidence'], mag['confidence'])
                if confidence < self.config['correlation']['min_correlation_score']:
                    continue
                
                return {
                    'time': max(tec['time'], mag['time']),
                    'location': tec['location'],  # Use TEC location as reference
                    'tec_magnitude': tec['magnitude'],
                    'magnetic_magnitude': mag['magnitude'],
                    'confidence': confidence
                }
        
        return None
    
    def _correlate_sdr_tracks(self,
                             doppler_tracks: List[Dict[str, Any]],
                             power_tracks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Correlate Doppler and power tracks."""
        if not doppler_tracks or not power_tracks:
            return None
            
        max_time_diff = self.config['correlation']['max_time_difference']
        
        for doppler in doppler_tracks:
            for power in power_tracks:
                # Check temporal correlation
                time_diff = abs((doppler['time'] - power['time']).total_seconds())
                if time_diff > max_time_diff:
                    continue
                
                # Check if tracks have similar durations
                duration_ratio = min(doppler['duration'], power['duration']) / max(doppler['duration'], power['duration'])
                if duration_ratio < 0.8:  # At least 80% overlap
                    continue
                
                # Calculate combined confidence
                confidence = min(doppler['confidence'], power['confidence'])
                if confidence < self.config['correlation']['min_correlation_score']:
                    continue
                
                return {
                    'time': max(doppler['time'], power['time']),
                    'frequency': doppler['frequency'],
                    'doppler_shift': doppler['max_shift'],
                    'power': power['max_power'],
                    'velocity_radial': doppler['velocity_radial'],
                    'location': doppler['location'],
                    'confidence': confidence
                }
        
        return None
    
    def _correlate_detections(self,
                            geo_detections: List[Dict[str, Any]],
                            sdr_detections: List[Dict[str, Any]]) -> List[TransitObject]:
        """Correlate detections across data sources."""
        if not geo_detections or not sdr_detections:
            return []
            
        transit_objects = []
        max_time_diff = self.config['correlation']['max_time_difference']
        max_pos_diff = self.config['correlation']['max_position_difference']
        
        for geo in geo_detections:
            for sdr in sdr_detections:
                # Check temporal correlation
                time_diff = abs((geo['time'] - sdr['time']).total_seconds())
                if time_diff > max_time_diff:
                    continue
                
                # Check spatial correlation
                pos1 = geo['location']
                pos2 = sdr['location']
                pos_diff = np.sqrt(
                    (pos1['lat'] - pos2['lat'])**2 +
                    (pos1['lon'] - pos2['lon'])**2 +
                    (pos1['alt'] - pos2['alt'])**2
                )
                
                if pos_diff > max_pos_diff:
                    continue
                
                # Calculate combined confidence
                confidence = min(geo['confidence'], sdr['confidence'])
                if confidence < self.config['correlation']['min_correlation_score']:
                    continue
                
                # Create transit object
                transit_obj = TransitObject(
                    time_first_detection=min(geo['time'], sdr['time']),
                    location=geo['location'],  # Use geophysical location as reference
                    velocity={
                        'vx': sdr['velocity_radial'] * 0.707,  # Approximate 3D velocity
                        'vy': sdr['velocity_radial'] * 0.707,
                        'vz': 0.0  # Will be updated by trajectory analysis
                    },
                    transit_type=TransitType.UNKNOWN,  # Will be classified later
                    confidence=confidence
                )
                
                transit_objects.append(transit_obj)
        
        return transit_objects
    
    def _classify_transit_type(self, obj: TransitObject) -> TransitType:
        """Classify the type of transit object."""
        # Calculate vertical velocity component
        if 'vz' in obj.velocity:
            vz = obj.velocity['vz']
            
            # Upward motion indicates launch
            if vz > 0 and obj.location['alt'] < 100:  # Below 100km
                return TransitType.SPACE_LAUNCH
                
            # Downward motion indicates reentry
            if vz < 0 and obj.location['alt'] > 80:  # Above 80km
                return TransitType.REENTRY
        
        return TransitType.UNKNOWN
    
    def _predict_impact(self, obj: TransitObject) -> Dict[str, Any]:
        """Predict impact location and time for Earth-bound objects.
        
        Args:
            obj: Transit object with current state
            
        Returns:
            Dictionary containing impact prediction
        """
        # Convert location to Cartesian coordinates
        lat_rad = np.radians(obj.location['lat'])
        lon_rad = np.radians(obj.location['lon'])
        alt = obj.location['alt'] * 1000  # Convert to meters
        
        # Calculate position vector
        R = 6.371e6  # Earth radius in meters
        r = (R + alt)
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        
        # Convert velocity to m/s
        vx = obj.velocity['vx'] * 1000
        vy = obj.velocity['vy'] * 1000
        vz = obj.velocity['vz'] * 1000
        
        # Create initial state vector
        initial_state = np.array([x, y, z, vx, vy, vz])
        
        # Estimate object mass based on radar cross section or default value
        mass = self.config.get('default_mass', 1000.0)  # kg
        
        # Get impact prediction from trajectory predictor
        impact = self.trajectory_predictor.predict_impact(
            initial_state=initial_state,
            mass=mass,
            time_step=1.0,  # 1 second time step
            max_time=3600.0  # Look ahead up to 1 hour
        )
        
        if impact is None:
            # If no impact predicted, return low confidence prediction
            return {
                'time': datetime.utcnow() + timedelta(hours=1),
                'location': {
                    'lat': obj.location['lat'],
                    'lon': obj.location['lon']
                },
                'uncertainty_radius_km': 500.0,
                'confidence': 0.1
            }
            
        return impact

class IonosphericAnalyzer:
    """Analyzer for ionospheric perturbations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the analyzer."""
        self.config = config
    
    def detect_perturbations(self, tec_data: List[float],
                           timestamps: List[datetime]) -> List[Dict[str, Any]]:
        """Detect significant perturbations in TEC data."""
        # Implementation for TEC perturbation detection
        return []

class MagneticFieldAnalyzer:
    """Analyzer for magnetic field disturbances."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the analyzer."""
        self.config = config
    
    def detect_disturbances(self, field_data: List[Dict[str, float]],
                          timestamps: List[datetime]) -> List[Dict[str, Any]]:
        """Detect significant magnetic field disturbances."""
        # Implementation for magnetic disturbance detection
        return []

class DopplerTracker:
    """Tracker for Doppler shift patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the tracker."""
        self.config = config
    
    def track_signals(self, frequencies: List[float],
                     timestamps: List[datetime]) -> List[Dict[str, Any]]:
        """Track Doppler-shifted signals over time."""
        # Implementation for Doppler tracking
        return []

class TrajectoryPredictor:
    """Predictor for object trajectories and impacts."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the predictor."""
        self.config = config
    
    def predict_trajectory(self, initial_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict object trajectory from initial state."""
        # Implementation for trajectory prediction
        return []
    
    def estimate_impact(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate impact location and time from trajectory."""
        # Implementation for impact estimation
        return {} 