import { ThresholdIndicator } from "./threshold-indicator"

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

interface ThresholdIndicatorListProps {
  data: IndicatorsData | null;
}

export function ThresholdIndicatorList({ data }: ThresholdIndicatorListProps) {
  // Transform UDL data into threshold indicators
  const thresholdIndicators = data?.udl.satellites.flatMap(satellite => {
    const indicators = []

    // Stability Threshold
    if (satellite.indicators.stability) {
      indicators.push({
        title: "Stability Threshold",
        description: "Monitors stability metrics against defined thresholds",
        threshold: `${(satellite.indicators.stability.threshold || 0.8) * 100}%`,
        passCriteria: "Stability score must be above threshold",
        features: [
          "Stability metrics",
          "Historical trends",
          "Anomaly detection"
        ]
      })
    }

    // Maneuver Frequency Threshold
    if (satellite.indicators.maneuvers) {
      indicators.push({
        title: "Maneuver Frequency",
        description: "Tracks frequency of orbital maneuvers",
        threshold: `${satellite.indicators.maneuvers.maxFrequency || 5} per week`,
        passCriteria: "Maneuver frequency must be below threshold",
        features: [
          "Maneuver detection",
          "Frequency tracking",
          "Pattern analysis"
        ]
      })
    }

    // RF Power Threshold
    if (satellite.indicators.rf) {
      indicators.push({
        title: "RF Power Level",
        description: "Monitors RF transmission power levels",
        threshold: `${satellite.indicators.rf.maxPower || 100} dBW`,
        passCriteria: "RF power must be below maximum threshold",
        features: [
          "Power measurement",
          "Frequency bands",
          "Transmission patterns"
        ]
      })
    }

    return indicators
  }) || []

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {thresholdIndicators.map((indicator, index) => (
        <ThresholdIndicator key={index} {...indicator} />
      ))}
    </div>
  )
}

