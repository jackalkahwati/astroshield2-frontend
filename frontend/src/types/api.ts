// Common Types
export interface BaseResponse {
  success: boolean
  message?: string
}

// Tracking Types
export interface TrackingData {
  satelliteId: string
  timestamp: number
  position: {
    x: number
    y: number
    z: number
  }
  velocity: {
    x: number
    y: number
    z: number
  }
}

export interface TrackingEvaluation extends BaseResponse {
  confidence: number
  predictions: TrackingData[]
}

// Stability Types
export interface StabilityMetrics {
  attitudeStability: number
  orbitStability: number
  overallScore: number
  recommendations: string[]
}

export interface StabilityAnalysis extends BaseResponse {
  metrics: StabilityMetrics
}

// Maneuver Types
export interface ManeuverOption {
  id: string
  type: string
  deltaV: number
  fuelCost: number
  successProbability: number
}

export interface ManeuverPlan extends BaseResponse {
  recommendedOptions: ManeuverOption[]
  optimalPath: ManeuverOption
}

// Physical Properties Types
export interface PhysicalProperties {
  mass: number
  dimensions: {
    length: number
    width: number
    height: number
  }
  centerOfMass: {
    x: number
    y: number
    z: number
  }
}

export interface PhysicalAnalysis extends BaseResponse {
  properties: PhysicalProperties
  structuralIntegrity: number
}

// Environmental Types
export interface EnvironmentalConditions {
  solarActivity: number
  debrisRisk: number
  radiationLevel: number
  temperature: number
}

export interface EnvironmentalAssessment extends BaseResponse {
  conditions: EnvironmentalConditions
  riskLevel: 'Low' | 'Medium' | 'High'
  mitigationStrategies: string[]
}

// Launch Types
export interface LaunchParameters {
  launchWindow: {
    start: string
    end: string
  }
  weather: {
    temperature: number
    windSpeed: number
    precipitation: number
  }
  trajectory: {
    azimuth: number
    elevation: number
    velocity: number
  }
}

export interface LaunchEvaluation extends BaseResponse {
  parameters: LaunchParameters
  feasibility: number
  constraints: string[]
} 