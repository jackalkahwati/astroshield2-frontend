import { ThresholdIndicator } from "./threshold-indicator"

const thresholdIndicators = [
  {
    title: "High Radiation Orbit",
    threshold: "1000rad/day",
    description: "Monitors if object is in high radiation environment",
    passCriteria: "Radiation levels within acceptable range",
    features: ["Orbit Parameters", "Space Weather"],
  },
]

export function ThresholdIndicatorList() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {thresholdIndicators.map((indicator, index) => (
        <ThresholdIndicator key={index} {...indicator} />
      ))}
    </div>
  )
}

