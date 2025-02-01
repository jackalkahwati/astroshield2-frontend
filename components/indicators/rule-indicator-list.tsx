import { RuleIndicator } from "./rule-indicator"

interface IndicatorsData {
  udl: {
    satellites: Array<{
      id: string;
      indicators: Record<string, any>;
    }>;
  };
  stability: {
    status: string;
    metrics: Record<string, any>;
  };
}

interface RuleIndicatorListProps {
  data: IndicatorsData | null;
}

export function RuleIndicatorList({ data }: RuleIndicatorListProps) {
  // Transform UDL data into rule indicators
  const ruleIndicators = data?.udl.satellites.flatMap(satellite => {
    const indicators = []

    // Eclipse Analysis Rule
    if (satellite.indicators.eclipse) {
      indicators.push({
        title: "Eclipse Analysis",
        version: "Rule v1.0.0",
        description: "Analyzes satellite behavior during eclipse periods",
        ruleSet: "Eclipse Behavior Rules",
        passCriteria: "All eclipse behaviors match predictions",
        features: [
          "Eclipse entry/exit times",
          "Power system behavior",
          "Maneuver detection"
        ]
      })
    }

    // Orbit Analysis Rule
    if (satellite.indicators.orbit) {
      indicators.push({
        title: "Orbit Analysis",
        version: "Rule v1.1.0",
        description: "Evaluates orbital parameters and behavior",
        ruleSet: "Orbital Parameters Rules",
        passCriteria: "All orbital parameters within expected ranges",
        features: [
          "Orbital elements",
          "Trajectory analysis",
          "Safety distances"
        ]
      })
    }

    // CCDM Analysis Rule
    if (satellite.indicators.ccdm) {
      indicators.push({
        title: "CCDM Analysis",
        version: "Rule v1.2.0",
        description: "Conjunction and collision risk assessment",
        ruleSet: "Collision Avoidance Rules",
        passCriteria: "No high-risk conjunctions or collisions detected",
        features: [
          "Conjunction analysis",
          "Safety margins",
          "Avoidance maneuvers"
        ]
      })
    }

    return indicators
  }) || []

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {ruleIndicators.map((indicator, index) => (
        <RuleIndicator key={index} {...indicator} />
      ))}
    </div>
  )
}

