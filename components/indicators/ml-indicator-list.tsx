import { MLIndicator } from "./ml-indicator"

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

interface MLIndicatorListProps {
  data: IndicatorsData | null;
}

export function MLIndicatorList({ data }: MLIndicatorListProps) {
  // Transform UDL data into ML indicators
  const mlIndicators = data?.udl.satellites.flatMap(satellite => {
    const indicators = []

    // Object Stability Indicator
    if (satellite.indicators.stability) {
      indicators.push({
        title: "Object Stability",
        version: "ML v2.3.1",
        description: "Evaluates if the object is maintaining stable orbit and behavior",
        confidenceScore: `${(satellite.indicators.stability.confidence * 100).toFixed(1)}%`,
        trainingAccuracy: "96.0%",
        algorithm: "LSTM Neural Network",
        passCriteria: "No significant deviations from expected orbital parameters",
        features: ["Orbital Elements", "Historical Stability", "Attitude Data"],
      })
    }

    // Maneuvers Indicator
    if (satellite.indicators.maneuvers) {
      indicators.push({
        title: "Maneuvers Detected",
        version: "ML v3.0.1",
        description: "Identifies and classifies orbital maneuvers",
        confidenceScore: `${(satellite.indicators.maneuvers.confidence * 100).toFixed(1)}%`,
        trainingAccuracy: "95.0%",
        algorithm: "Bi-LSTM with Attention",
        passCriteria: "All maneuvers match declared operations",
        features: ["Trajectory Data", "Historical Maneuvers"],
      })
    }

    // RF Detection Indicator
    if (satellite.indicators.rf) {
      indicators.push({
        title: "RF Detected",
        version: "ML v2.2.0",
        description: "Detects and characterizes RF emissions",
        confidenceScore: `${(satellite.indicators.rf.confidence * 100).toFixed(1)}%`,
        trainingAccuracy: "94.0%",
        algorithm: "Deep Neural Network",
        passCriteria: "RF emissions match declared capabilities",
        features: ["RF Spectrum", "Signal Characteristics"],
      })
    }

    return indicators
  }) || []

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {mlIndicators.map((indicator, index) => (
        <MLIndicator key={index} {...indicator} />
      ))}
    </div>
  )
}

