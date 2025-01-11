const axios = require('axios');

jest.mock('axios');

describe('Adaptation Engine', () => {
  describe('Strategy Modification Logic', () => {
    test('should adapt strategy based on effectiveness evaluation', () => {
      const mockCurrentStrategy = {
        type: 'MULTI_LAYER_CCDM',
        components: [
          {
            type: 'MANEUVER',
            intensity: 0.5,
            resource_allocation: 0.4
          },
          {
            type: 'DECEPTION',
            intensity: 0.3,
            resource_allocation: 0.3
          }
        ],
        mode: 'NORMAL',
        priority: 'MEDIUM'
      };

      const mockEffectiveness = {
        overall_score: 0.4,  // Below threshold
        metrics: {
          success_rate: 0.5,
          resource_efficiency: 0.6,
          response_time: 0.4
        },
        recommendations: [
          {
            component: 'MANEUVER',
            adjustment: 'INCREASE_INTENSITY',
            confidence: 0.8
          }
        ]
      };

      const adaptedStrategy = {
        type: 'MULTI_LAYER_CCDM',
        components: [
          {
            type: 'MANEUVER',
            intensity: 0.7,  // Increased
            resource_allocation: 0.5  // Adjusted
          },
          {
            type: 'DECEPTION',
            intensity: 0.3,
            resource_allocation: 0.2  // Adjusted
          }
        ],
        mode: 'ENHANCED',
        priority: 'HIGH'
      };

      expect(adaptedStrategy.components[0].intensity).toBeGreaterThan(
        mockCurrentStrategy.components[0].intensity
      );
      expect(adaptedStrategy.mode).toBe('ENHANCED');
      expect(adaptedStrategy.priority).toBe('HIGH');
    });

    test('should handle multiple adaptation rules', () => {
      const mockRules = [
        {
          condition: 'effectiveness.overall_score < 0.6',
          action: 'INCREASE_STRATEGY_INTENSITY',
          priority: 3
        },
        {
          condition: 'telemetry.threat_indicators.threat_level > 0.8',
          action: 'ACTIVATE_EMERGENCY_PROTOCOL',
          priority: 5
        }
      ];

      const mockTelemetry = {
        threat_indicators: {
          threat_level: 0.9
        }
      };

      const ruleEvaluation = {
        selected_rule: mockRules[1],  // Higher priority rule
        evaluation_metrics: {
          condition_match: true,
          priority_score: 5,
          applicability: 0.9
        }
      };

      expect(ruleEvaluation.selected_rule.action).toBe('ACTIVATE_EMERGENCY_PROTOCOL');
      expect(ruleEvaluation.evaluation_metrics.priority_score).toBe(5);
    });
  });

  describe('Resource Optimization', () => {
    test('should optimize resource allocation across components', () => {
      const mockResources = {
        fuel: 75,
        power: 90,
        computational_capacity: 80
      };

      const mockComponents = [
        {
          type: 'MANEUVER',
          resource_requirements: {
            fuel: 30,
            power: 20
          }
        },
        {
          type: 'DECEPTION',
          resource_requirements: {
            power: 30,
            computational_capacity: 40
          }
        }
      ];

      const optimization = {
        allocations: [
          {
            component: 'MANEUVER',
            resources: {
              fuel: 25,
              power: 15
            }
          },
          {
            component: 'DECEPTION',
            resources: {
              power: 25,
              computational_capacity: 35
            }
          }
        ],
        efficiency_metrics: {
          resource_utilization: 0.85,
          balance_score: 0.9,
          overhead: 0.1
        }
      };

      expect(optimization.efficiency_metrics.resource_utilization).toBeGreaterThan(0.8);
      expect(optimization.efficiency_metrics.overhead).toBeLessThan(0.2);
    });

    test('should handle resource constraints', () => {
      const mockConstraints = {
        min_fuel_reserve: 20,
        max_power_usage: 95,
        min_computational_capacity: 30
      };

      const resourceAllocation = {
        is_feasible: true,
        allocations: {
          fuel_usage: 55,  // Leaves required reserve
          power_usage: 90,  // Within max limit
          computational_usage: 45  // Above minimum
        },
        constraint_satisfaction: {
          fuel_constraint: true,
          power_constraint: true,
          computational_constraint: true
        }
      };

      expect(resourceAllocation.is_feasible).toBe(true);
      expect(resourceAllocation.allocations.fuel_usage).toBeLessThanOrEqual(80);  // 100 - min_reserve
      expect(resourceAllocation.allocations.power_usage).toBeLessThanOrEqual(mockConstraints.max_power_usage);
    });
  });

  describe('Safety Constraints', () => {
    test('should validate safety constraints for adaptations', () => {
      const mockSafetyConstraints = {
        min_separation_distance: 100,
        max_delta_v: 2.0,
        min_fuel_reserve: 20,
        max_risk_level: 0.3
      };

      const mockAdaptation = {
        maneuver: {
          delta_v: 1.5,
          resulting_separation: 150
        },
        resource_impact: {
          fuel_consumption: 15,
          risk_level: 0.2
        }
      };

      const safetyValidation = {
        is_safe: true,
        constraint_checks: [
          { name: 'separation_check', passed: true },
          { name: 'delta_v_check', passed: true },
          { name: 'fuel_check', passed: true },
          { name: 'risk_check', passed: true }
        ],
        safety_metrics: {
          separation_margin: 50,
          velocity_margin: 0.5,
          risk_margin: 0.1
        }
      };

      expect(safetyValidation.is_safe).toBe(true);
      expect(safetyValidation.constraint_checks.every(check => check.passed)).toBe(true);
      expect(safetyValidation.safety_metrics.separation_margin).toBeGreaterThan(0);
    });

    test('should handle safety constraint violations', () => {
      const mockUnsafeAdaptation = {
        maneuver: {
          delta_v: 2.5,  // Exceeds max_delta_v
          resulting_separation: 80  // Below min_separation
        },
        resource_impact: {
          fuel_consumption: 25,  // Violates min_reserve
          risk_level: 0.4  // Exceeds max_risk
        }
      };

      const safetyEvaluation = {
        is_safe: false,
        violations: [
          {
            constraint: 'max_delta_v',
            value: 2.5,
            limit: 2.0
          },
          {
            constraint: 'min_separation',
            value: 80,
            limit: 100
          }
        ],
        recommended_adjustments: [
          {
            parameter: 'delta_v',
            current: 2.5,
            recommended: 1.8
          },
          {
            parameter: 'trajectory',
            adjustment: 'INCREASE_SEPARATION'
          }
        ]
      };

      expect(safetyEvaluation.is_safe).toBe(false);
      expect(safetyEvaluation.violations.length).toBeGreaterThan(0);
      expect(safetyEvaluation.recommended_adjustments).toBeDefined();
    });

    test('should maintain safety during emergency adaptations', () => {
      const mockEmergencyScenario = {
        threat_level: 'HIGH',
        time_critical: true,
        current_resources: {
          fuel: 30,
          power: 85
        }
      };

      const emergencyAdaptation = {
        is_safe: true,
        strategy: {
          type: 'EMERGENCY_MANEUVER',
          parameters: {
            delta_v: 1.8,  // Within safety limits
            fuel_usage: 10  // Maintains reserve
          }
        },
        safety_checks: {
          basic_constraints_met: true,
          emergency_constraints_met: true,
          risk_assessment: {
            level: 'ACCEPTABLE',
            confidence: 0.9
          }
        }
      };

      expect(emergencyAdaptation.is_safe).toBe(true);
      expect(emergencyAdaptation.strategy.parameters.delta_v).toBeLessThanOrEqual(2.0);
      expect(emergencyAdaptation.safety_checks.basic_constraints_met).toBe(true);
      expect(emergencyAdaptation.safety_checks.emergency_constraints_met).toBe(true);
    });
  });
});
