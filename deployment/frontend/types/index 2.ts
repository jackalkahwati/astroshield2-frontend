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

export interface ComprehensiveData {
  metrics: Record<string, number>
  status: string
  alerts: string[]
  timestamp: string
}