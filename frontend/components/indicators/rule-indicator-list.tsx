import { RuleIndicator } from "./rule-indicator"

const ruleIndicators = [
  {
    title: "ITU/FCC Violation",
    version: "Rule v1.9.0",
    description: "Checks compliance with ITU and FCC filings",
    ruleSet: "ITU/FCC Compliance Rules",
    passCriteria: "All RF activities comply with filings",
    features: ["RF Parameters", "Filing Data"],
  },
  {
    title: "Unoccupied Orbit",
    version: "Rule v1.1.0",
    description: "Checks if object is in relatively unoccupied orbit",
    ruleSet: "Orbit Population Rules",
    passCriteria: "Object density below threshold",
    features: ["Space Object Catalog", "Orbit Statistics"],
  },
  {
    title: "Threat Launch Origin",
    version: "Rule v1.2.1",
    description: "Identifies if object came from known threat launch site",
    ruleSet: "Launch Site Classification",
    passCriteria: "Launch site not associated with threats",
    features: ["Launch Site Data", "Historical Launches"],
  },
  {
    title: "UN Registry",
    version: "Rule v2.1.0",
    description: "Checks if object is in UN satellite registry",
    ruleSet: "UN Registry Compliance",
    passCriteria: "Object properly registered",
    features: ["UN Registry Data", "Object Identity"],
  },
]

export function RuleIndicatorList() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {ruleIndicators.map((indicator, index) => (
        <RuleIndicator key={index} {...indicator} />
      ))}
    </div>
  )
}

