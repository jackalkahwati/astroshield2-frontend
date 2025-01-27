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
  type: string
  status: string
  scheduledTime: string
  completedTime?: string
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