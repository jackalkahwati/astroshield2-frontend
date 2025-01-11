import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import asyncio
from dataclasses import dataclass

@dataclass
class TelemetryData:
    spacecraft_id: str
    timestamp: datetime
    sensor_readings: Dict[str, float]
    deployed_strategy: Dict
    resource_status: Dict[str, float]

@dataclass
class StrategyEffectiveness:
    overall_score: float
    metrics: Dict[str, float]
    recommendations: List[Dict]
    confidence: float

class StrategyEvaluator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'evaluation_window': 3600,  # 1 hour
            'min_confidence_threshold': 0.75,
            'metric_weights': {
                'success_rate': 0.4,
                'resource_efficiency': 0.3,
                'response_time': 0.3
            }
        }
        self.evaluation_history = []

    async def evaluate_strategy(self, telemetry: TelemetryData) -> StrategyEffectiveness:
        """
        Evaluates the effectiveness of currently deployed CCDM strategy based on telemetry data.
        """
        try:
            # Calculate base metrics
            success_rate = await self._calculate_success_rate(telemetry)
            resource_efficiency = await self._analyze_resource_efficiency(telemetry)
            response_time = await self._measure_response_time(telemetry)

            # Compute weighted score
            overall_score = (
                success_rate * self.config['metric_weights']['success_rate'] +
                resource_efficiency * self.config['metric_weights']['resource_efficiency'] +
                response_time * self.config['metric_weights']['response_time']
            )

            # Generate recommendations
            recommendations = await self._generate_recommendations(
                telemetry,
                {
                    'success_rate': success_rate,
                    'resource_efficiency': resource_efficiency,
                    'response_time': response_time
                }
            )

            # Calculate confidence
            confidence = self._calculate_confidence(telemetry, overall_score)

            effectiveness = StrategyEffectiveness(
                overall_score=overall_score,
                metrics={
                    'success_rate': success_rate,
                    'resource_efficiency': resource_efficiency,
                    'response_time': response_time
                },
                recommendations=recommendations,
                confidence=confidence
            )

            # Store evaluation result
            self.evaluation_history.append({
                'timestamp': datetime.now(),
                'effectiveness': effectiveness,
                'telemetry': telemetry
            })

            return effectiveness

        except Exception as e:
            print(f"Error in strategy evaluation: {str(e)}")
            raise

    async def _calculate_success_rate(self, telemetry: TelemetryData) -> float:
        """
        Calculates the success rate of the current strategy based on threat avoidance
        and mission objective maintenance.
        """
        threat_avoidance = telemetry.sensor_readings.get('threat_detection_confidence', 0.0)
        position_accuracy = telemetry.sensor_readings.get('position_accuracy', 0.0)
        
        # Weight the components
        success_rate = 0.7 * threat_avoidance + 0.3 * position_accuracy
        return min(max(success_rate, 0.0), 1.0)

    async def _analyze_resource_efficiency(self, telemetry: TelemetryData) -> float:
        """
        Analyzes how efficiently resources are being used by the current strategy.
        """
        resource_usage = telemetry.resource_status.get('utilization', 0.0)
        fuel_efficiency = telemetry.resource_status.get('fuel_efficiency', 1.0)
        
        # Calculate efficiency score
        efficiency = (1 - resource_usage) * fuel_efficiency
        return min(max(efficiency, 0.0), 1.0)

    async def _measure_response_time(self, telemetry: TelemetryData) -> float:
        """
        Measures the system's response time to threats and changes in the environment.
        """
        baseline_response = 1000.0  # milliseconds
        actual_response = telemetry.sensor_readings.get('response_time', baseline_response)
        
        # Normalize response time (lower is better)
        normalized_response = baseline_response / max(actual_response, 1.0)
        return min(max(normalized_response, 0.0), 1.0)

    async def _generate_recommendations(
        self,
        telemetry: TelemetryData,
        metrics: Dict[str, float]
    ) -> List[Dict]:
        """
        Generates strategy adjustment recommendations based on evaluation metrics.
        """
        recommendations = []

        # Check for low success rate
        if metrics['success_rate'] < 0.8:
            recommendations.append({
                'component': telemetry.deployed_strategy['type'],
                'adjustment': 'INCREASE_INTENSITY',
                'confidence': 0.85,
                'reason': 'Low success rate detected'
            })

        # Check resource efficiency
        if metrics['resource_efficiency'] < 0.6:
            recommendations.append({
                'component': 'RESOURCE_MANAGEMENT',
                'adjustment': 'OPTIMIZE_USAGE',
                'confidence': 0.9,
                'reason': 'Poor resource efficiency'
            })

        # Check response time
        if metrics['response_time'] < 0.7:
            recommendations.append({
                'component': 'RESPONSE_SYSTEM',
                'adjustment': 'REDUCE_LATENCY',
                'confidence': 0.8,
                'reason': 'Slow response time'
            })

        return recommendations

    def _calculate_confidence(self, telemetry: TelemetryData, overall_score: float) -> float:
        """
        Calculates the confidence level of the evaluation based on data quality and coverage.
        """
        # Check data quality
        sensor_confidence = telemetry.sensor_readings.get('threat_detection_confidence', 0.0)
        data_completeness = sum(1 for v in telemetry.sensor_readings.values() if v is not None) / len(telemetry.sensor_readings)
        
        # Calculate final confidence
        confidence = (sensor_confidence + data_completeness + overall_score) / 3
        return min(max(confidence, 0.0), 1.0)

    async def get_historical_performance(
        self,
        spacecraft_id: str,
        time_window: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieves historical performance data for analysis and trending.
        """
        time_window = time_window or self.config['evaluation_window']
        current_time = datetime.now()
        
        return [
            entry for entry in self.evaluation_history
            if (
                entry['telemetry'].spacecraft_id == spacecraft_id and
                (current_time - entry['timestamp']).total_seconds() <= time_window
            )
        ]

    async def analyze_trends(self, spacecraft_id: str) -> Dict:
        """
        Analyzes performance trends to identify patterns and long-term effectiveness.
        """
        history = await self.get_historical_performance(spacecraft_id)
        if not history:
            return {'trend': 'INSUFFICIENT_DATA'}

        scores = [entry['effectiveness'].overall_score for entry in history]
        trend_slope = np.polyfit(range(len(scores)), scores, 1)[0]

        return {
            'trend': 'IMPROVING' if trend_slope > 0.1 else 'DEGRADING' if trend_slope < -0.1 else 'STABLE',
            'slope': trend_slope,
            'average_score': np.mean(scores),
            'volatility': np.std(scores)
        }
