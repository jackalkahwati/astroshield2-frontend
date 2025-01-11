import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class RawTelemetry:
    spacecraft_id: str
    timestamp: datetime
    sensor_data: Dict[str, Union[float, List[float]]]
    system_status: Dict[str, Union[str, float]]
    environmental_data: Dict[str, Union[float, List[float]]]

@dataclass
class ProcessedTelemetry:
    spacecraft_id: str
    timestamp: datetime
    sensor_readings: Dict[str, float]
    resource_status: Dict[str, float]
    threat_indicators: Dict[str, float]
    quality_metrics: Dict[str, float]

class TelemetryProcessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'sampling_rate': 10,  # Hz
            'smoothing_window': 5,  # samples
            'noise_threshold': 0.1,
            'thread_pool_size': 4
        }
        self.executor = ThreadPoolExecutor(max_workers=self.config['thread_pool_size'])
        self.telemetry_buffer = []
        self.running = False

    async def start_processing(self):
        """
        Starts the telemetry processing loop.
        """
        self.running = True
        while self.running:
            try:
                await self._process_telemetry_batch()
                await asyncio.sleep(1.0 / self.config['sampling_rate'])
            except Exception as e:
                print(f"Error in telemetry processing loop: {str(e)}")
                await asyncio.sleep(1)  # Error recovery delay

    async def stop_processing(self):
        """
        Stops the telemetry processing loop.
        """
        self.running = False
        self.executor.shutdown(wait=True)

    async def process_telemetry(self, raw_telemetry: RawTelemetry) -> ProcessedTelemetry:
        """
        Processes raw telemetry data into a standardized format for analysis.
        """
        try:
            # Process different aspects of telemetry concurrently
            sensor_task = asyncio.create_task(self._process_sensor_data(raw_telemetry.sensor_data))
            status_task = asyncio.create_task(self._process_system_status(raw_telemetry.system_status))
            environ_task = asyncio.create_task(self._process_environmental_data(raw_telemetry.environmental_data))

            # Wait for all processing to complete
            sensor_readings, resource_status, threat_indicators = await asyncio.gather(
                sensor_task, status_task, environ_task
            )

            # Calculate quality metrics based on raw data
            total_fields = (
                len(raw_telemetry.sensor_data) +
                len(raw_telemetry.system_status) +
                len(raw_telemetry.environmental_data)
            )
            processed_fields = (
                len(sensor_readings) +
                len(resource_status) +
                len(threat_indicators)
            )

            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                sensor_readings, resource_status, threat_indicators,
                total_fields, processed_fields
            )

            processed = ProcessedTelemetry(
                spacecraft_id=raw_telemetry.spacecraft_id,
                timestamp=raw_telemetry.timestamp,
                sensor_readings=sensor_readings,
                resource_status=resource_status,
                threat_indicators=threat_indicators,
                quality_metrics=quality_metrics
            )

            # Store in buffer for batch processing
            self.telemetry_buffer.append(processed)

            return processed

        except Exception as e:
            print(f"Error processing telemetry: {str(e)}")
            raise

    async def _process_sensor_data(self, sensor_data: Dict) -> Dict[str, float]:
        """
        Processes raw sensor data, applying noise reduction and calibration.
        """
        processed_data = {}
        
        def process_sensor_reading(key: str, value: Union[float, List[float]]) -> float:
            if isinstance(value, list):
                # Apply moving average for noise reduction
                smoothed = np.convolve(
                    value,
                    np.ones(self.config['smoothing_window'])/self.config['smoothing_window'],
                    mode='valid'
                )
                return float(smoothed[-1])
            return float(value)

        # Process sensor readings concurrently
        tasks = []
        for key, value in sensor_data.items():
            if value is not None:
                task = asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    process_sensor_reading,
                    key,
                    value
                )
                tasks.append((key, task))

        # Collect results
        for key, task in tasks:
            processed_data[key] = await task

        return processed_data

    async def _process_system_status(self, system_status: Dict) -> Dict[str, float]:
        """
        Processes system status information into numerical metrics.
        """
        resource_status = {}
        
        # Convert status indicators to numerical values
        status_mapping = {
            'OPTIMAL': 1.0,
            'NOMINAL': 0.8,
            'DEGRADED': 0.5,
            'CRITICAL': 0.2
        }

        for key, value in system_status.items():
            if isinstance(value, str):
                resource_status[key] = status_mapping.get(value.upper(), 0.0)
            else:
                resource_status[key] = float(value)

        return resource_status

    async def _process_environmental_data(self, environmental_data: Dict) -> Dict[str, float]:
        """
        Processes environmental data to extract threat indicators.
        """
        threat_indicators = {}
        
        # Process environmental readings for threat assessment
        for key, value in environmental_data.items():
            if isinstance(value, list):
                # Calculate rate of change
                changes = np.diff(value)
                avg_change = np.mean(np.abs(changes))
                threat_indicators[f"{key}_volatility"] = min(avg_change, 1.0)
            else:
                # Normalize single values
                threat_indicators[key] = min(float(value), 1.0)

        return threat_indicators

    async def _calculate_quality_metrics(
        self,
        sensor_readings: Dict[str, float],
        resource_status: Dict[str, float],
        threat_indicators: Dict[str, float],
        total_fields: int,
        processed_fields: int
    ) -> Dict[str, float]:
        """
        Calculates quality metrics for the processed telemetry data.
        """
        quality_metrics = {
            'completeness': 0.0,
            'reliability': 0.0,
            'accuracy': 0.0
        }

        # Calculate data completeness based on raw vs processed fields
        quality_metrics['completeness'] = processed_fields / total_fields if total_fields > 0 else 0.0

        # Calculate data reliability based on sensor readings consistency
        if sensor_readings:
            variations = [
                abs(v - np.mean(list(sensor_readings.values())))
                for v in sensor_readings.values()
                if v is not None
            ]
            quality_metrics['reliability'] = 1.0 - min(np.mean(variations), 1.0) if variations else 0.0

        # Calculate accuracy based on threat indicator confidence
        if threat_indicators:
            confidence_values = [v for v in threat_indicators.values() if v is not None and 0 <= v <= 1]
            quality_metrics['accuracy'] = np.mean(confidence_values) if confidence_values else 0.0

        return quality_metrics

    async def _process_telemetry_batch(self):
        """
        Processes accumulated telemetry data in batches for efficiency.
        """
        if not self.telemetry_buffer:
            return

        try:
            # Process batch
            batch = self.telemetry_buffer.copy()
            self.telemetry_buffer.clear()

            # Aggregate batch statistics
            aggregated_stats = {
                'sensor_readings': {},
                'resource_status': {},
                'threat_indicators': {}
            }

            for telemetry in batch:
                for category in aggregated_stats:
                    data = getattr(telemetry, category)
                    for key, value in data.items():
                        if key not in aggregated_stats[category]:
                            aggregated_stats[category][key] = []
                        aggregated_stats[category][key].append(value)

            # Calculate batch metrics
            for category in aggregated_stats:
                for key in aggregated_stats[category]:
                    values = aggregated_stats[category][key]
                    aggregated_stats[category][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }

            return aggregated_stats

        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            raise
