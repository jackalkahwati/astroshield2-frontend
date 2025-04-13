// API Response Types
export interface ApiError {
  message: string
  code?: string
  details?: any
}

export interface ApiResponse<T> {
  data: T | null
  status?: number
  message?: string
  error?: ApiError
  retryAfter?: number
}

// Domain Types
export interface ManeuverData {
  id: string
  satellite_id: string
  type: string
  status: string
  scheduledTime: string
  completedTime?: string
  created_by?: string
  created_at?: string
  details: {
    delta_v?: number
    duration?: number
    fuel_required?: number
    fuel_used?: number
    target_orbit?: {
      altitude?: number
      inclination?: number
    }
  }
}

export interface ComprehensiveData {
  metrics: Record<string, number>
  status: string
  alerts: string[]
  timestamp: string
}

export interface SatelliteData {
  id: string
  name: string
  status: string
  orbit: {
    altitude: number
    inclination: number
    period: number
  }
  telemetry: {
    battery: number
    temperature: number
    signal_strength: number
  }
  lastContact: string
}

export interface SystemHealth {
  status: string
  timestamp: string
  services: {
    database: string
    api: string
    telemetry: string
  }
}

export interface TelemetryData {
  satellite_id: string
  timestamp: string
  data: {
    power: {
      battery_level: number
      solar_panel_output: number
      power_consumption: number
    }
    thermal: {
      internal_temp: number
      external_temp: number
      heating_power: number
    }
    communication: {
      signal_strength: number
      bit_error_rate: number
      latency: number
    }
  }
}

// Trajectory Types
export interface TrajectoryPoint {
  time: number
  position: [number, number, number]  // [longitude, latitude, altitude]
  velocity: [number, number, number]  // [x, y, z] velocity components
}

export interface BreakupEvent {
  altitude: number
  fragments: number
  time: string
}

export interface ImpactPrediction {
  time: string
  location: {
    lat: number
    lon: number
  }
  velocity: {
    magnitude: number
    direction: {
      x: number
      y: number
      z: number
    }
  }
  uncertainty_radius_km: number
  confidence: number
}

// API-specific Types
export interface RateLimitError extends ApiError {
  retryAfter: number
}

export interface NetworkError extends ApiError {
  isNetworkError: true
  originalError: Error
}

export interface ValidationError extends ApiError {
  fieldErrors: Record<string, string[]>
}

export type EnhancedApiError = ApiError | RateLimitError | NetworkError | ValidationError

// Monitoring Types
export interface SecurityHeaders {
  'Content-Security-Policy': string
  'Strict-Transport-Security': string
  'X-Content-Type-Options': string
  'X-Frame-Options': string
  'X-XSS-Protection': string
  [key: string]: string
}

export interface PerformanceMetrics {
  endpoint: string
  method: string
  duration: number
  status: number
  timestamp: string
}

export interface MonitoringMetrics extends PerformanceMetrics {
  cacheHits: number
  cacheMisses: number
  deduplicationHits: number
  backgroundRefreshes: number
  memoryUsage: number
} 