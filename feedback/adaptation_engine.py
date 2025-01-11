import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from .strategy_evaluator import StrategyEffectiveness
from .telemetry_processor import ProcessedTelemetry

@dataclass
class AdaptationRule:
    condition: str  # Python expression to evaluate
    action: str    # Strategy adjustment to apply
    priority: int  # Higher number = higher priority
    cooldown: int  # Minimum time (seconds) between applications

@dataclass
class AdaptationResult:
    success: bool
    adapted_strategy: Dict
    adaptation_reason: str
    confidence: float
    timestamp: datetime

class AdaptationEngine:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'min_adaptation_confidence': 0.7,
            'max_adaptations_per_hour': 10,
            'adaptation_cooldown': 300,  # 5 minutes
            'learning_rate': 0.1
        }
        self.adaptation_history = []
        self.active_rules = self._initialize_rules()
        self.last_adaptation = None
        self.learning_buffer = []

    def _initialize_rules(self) -> List[AdaptationRule]:
        """
        Initializes the base set of adaptation rules.
        """
        return [
            AdaptationRule(
                condition="effectiveness.overall_score < 0.6",
                action="INCREASE_STRATEGY_INTENSITY",
                priority=3,
                cooldown=300
            ),
            AdaptationRule(
                condition="telemetry.threat_indicators.get('threat_level', 0) > 0.8",
                action="ACTIVATE_EMERGENCY_PROTOCOL",
                priority=5,
                cooldown=60
            ),
            AdaptationRule(
                condition="telemetry.resource_status.get('fuel_efficiency', 1) < 0.3",
                action="OPTIMIZE_RESOURCE_USAGE",
                priority=4,
                cooldown=600
            ),
            AdaptationRule(
                condition="effectiveness.metrics.get('response_time', 1) > 0.8",
                action="REDUCE_LATENCY",
                priority=2,
                cooldown=300
            )
        ]

    async def adapt_strategy(
        self,
        current_strategy: Dict,
        effectiveness: StrategyEffectiveness,
        telemetry: ProcessedTelemetry
    ) -> AdaptationResult:
        """
        Adapts the current CCDM strategy based on effectiveness evaluation and telemetry data.
        """
        try:
            # Check adaptation frequency
            if not self._can_adapt():
                return AdaptationResult(
                    success=False,
                    adapted_strategy=current_strategy,
                    adaptation_reason="COOLDOWN_ACTIVE",
                    confidence=1.0,
                    timestamp=datetime.now()
                )

            # Evaluate rules and get highest priority adaptation
            applicable_rules = await self._evaluate_rules(effectiveness, telemetry)
            if not applicable_rules:
                return AdaptationResult(
                    success=False,
                    adapted_strategy=current_strategy,
                    adaptation_reason="NO_ADAPTATION_NEEDED",
                    confidence=1.0,
                    timestamp=datetime.now()
                )

            # Apply adaptation
            adapted_strategy = await self._apply_adaptation(
                current_strategy,
                applicable_rules[0],
                effectiveness,
                telemetry
            )

            # Calculate adaptation confidence
            confidence = self._calculate_adaptation_confidence(
                adapted_strategy,
                current_strategy,
                effectiveness
            )

            # Record adaptation
            result = AdaptationResult(
                success=True,
                adapted_strategy=adapted_strategy,
                adaptation_reason=applicable_rules[0].action,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            self.adaptation_history.append(result)
            self.last_adaptation = datetime.now()
            
            # Update learning buffer
            self.learning_buffer.append({
                'previous_strategy': current_strategy,
                'adaptation': result,
                'effectiveness': effectiveness
            })

            return result

        except Exception as e:
            print(f"Error in strategy adaptation: {str(e)}")
            raise

    async def _evaluate_rules(
        self,
        effectiveness: StrategyEffectiveness,
        telemetry: ProcessedTelemetry
    ) -> List[AdaptationRule]:
        """
        Evaluates adaptation rules and returns applicable ones in priority order.
        """
        applicable_rules = []
        
        for rule in self.active_rules:
            try:
                # Create evaluation context
                context = {
                    'effectiveness': effectiveness,
                    'telemetry': telemetry
                }
                
                # Evaluate rule condition
                if eval(rule.condition, {}, context):
                    # Check cooldown
                    last_application = next(
                        (a for a in reversed(self.adaptation_history)
                         if a.adaptation_reason == rule.action),
                        None
                    )
                    
                    if not last_application or \
                       (datetime.now() - last_application.timestamp).total_seconds() >= rule.cooldown:
                        applicable_rules.append(rule)
            
            except Exception as e:
                print(f"Error evaluating rule {rule.action}: {str(e)}")
                continue

        # Sort by priority (highest first)
        return sorted(applicable_rules, key=lambda r: r.priority, reverse=True)

    async def _apply_adaptation(
        self,
        current_strategy: Dict,
        rule: AdaptationRule,
        effectiveness: StrategyEffectiveness,
        telemetry: ProcessedTelemetry
    ) -> Dict:
        """
        Applies the selected adaptation rule to the current strategy.
        """
        adapted_strategy = current_strategy.copy()
        
        if rule.action == "INCREASE_STRATEGY_INTENSITY":
            # Increase the intensity of current CCDM components
            for component in adapted_strategy.get('components', []):
                component['intensity'] = min(
                    component.get('intensity', 0.5) * 1.5,
                    1.0
                )

        elif rule.action == "ACTIVATE_EMERGENCY_PROTOCOL":
            adapted_strategy['mode'] = 'EMERGENCY'
            adapted_strategy['priority'] = 'HIGH'
            adapted_strategy['components'] = self._get_emergency_components(
                current_strategy.get('components', [])
            )

        elif rule.action == "OPTIMIZE_RESOURCE_USAGE":
            # Optimize resource allocation across components
            total_resources = sum(
                c.get('resource_allocation', 0.1)
                for c in adapted_strategy.get('components', [])
            )
            
            if total_resources > 0:
                for component in adapted_strategy.get('components', []):
                    effectiveness_contribution = component.get('effectiveness', 0.5)
                    component['resource_allocation'] = (
                        effectiveness_contribution / total_resources
                    )

        elif rule.action == "REDUCE_LATENCY":
            # Optimize for faster response time
            adapted_strategy['execution_mode'] = 'RAPID_RESPONSE'
            adapted_strategy['preprocessing_level'] = 'MINIMAL'
            
            for component in adapted_strategy.get('components', []):
                if component.get('type') in ['DECEPTION', 'MANEUVER']:
                    component['response_priority'] = 'HIGH'

        return adapted_strategy

    def _get_emergency_components(self, current_components: List[Dict]) -> List[Dict]:
        """
        Generates emergency protocol components based on current configuration.
        """
        emergency_components = []
        
        for component in current_components:
            emergency_component = component.copy()
            emergency_component.update({
                'intensity': 1.0,
                'response_priority': 'HIGH',
                'resource_allocation': 1.0
            })
            emergency_components.append(emergency_component)
            
        return emergency_components

    def _calculate_adaptation_confidence(
        self,
        adapted_strategy: Dict,
        current_strategy: Dict,
        effectiveness: StrategyEffectiveness
    ) -> float:
        """
        Calculates confidence in the adaptation based on historical performance.
        """
        # Base confidence on effectiveness evaluation
        base_confidence = effectiveness.confidence

        # Adjust based on historical success with similar adaptations
        similar_adaptations = [
            a for a in self.adaptation_history
            if a.adaptation_reason == adapted_strategy.get('type')
        ]
        
        if similar_adaptations:
            historical_success = sum(
                1 for a in similar_adaptations
                if a.success
            ) / len(similar_adaptations)
            
            # Weighted average of base confidence and historical success
            confidence = 0.7 * base_confidence + 0.3 * historical_success
        else:
            confidence = base_confidence

        return min(max(confidence, 0.0), 1.0)

    def _can_adapt(self) -> bool:
        """
        Checks if adaptation is allowed based on frequency and cooldown constraints.
        """
        if not self.last_adaptation:
            return True

        # Check cooldown period
        time_since_last = (datetime.now() - self.last_adaptation).total_seconds()
        if time_since_last < self.config['adaptation_cooldown']:
            return False

        # Check adaptation frequency
        recent_adaptations = len([
            a for a in self.adaptation_history
            if (datetime.now() - a.timestamp).total_seconds() <= 3600
        ])
        
        return recent_adaptations < self.config['max_adaptations_per_hour']

    async def learn_from_adaptations(self):
        """
        Analyzes adaptation history to improve future decisions.
        """
        if len(self.learning_buffer) < 2:
            return

        try:
            # Calculate success rates for different adaptation types
            adaptation_success = {}
            
            for entry in self.learning_buffer:
                adaptation_type = entry['adaptation'].adaptation_reason
                success_score = entry['effectiveness'].overall_score
                
                if adaptation_type not in adaptation_success:
                    adaptation_success[adaptation_type] = []
                    
                adaptation_success[adaptation_type].append(success_score)

            # Update rule priorities based on success rates
            for rule in self.active_rules:
                if rule.action in adaptation_success:
                    success_rate = np.mean(adaptation_success[rule.action])
                    
                    # Adjust rule priority based on success rate
                    if success_rate > 0.8:
                        rule.priority = min(rule.priority + 1, 5)
                    elif success_rate < 0.4:
                        rule.priority = max(rule.priority - 1, 1)

            # Clear learning buffer
            self.learning_buffer = self.learning_buffer[-100:]  # Keep last 100 entries

        except Exception as e:
            print(f"Error in adaptation learning: {str(e)}")
            raise
