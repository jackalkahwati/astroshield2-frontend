import axios, { AxiosError, AxiosResponse } from 'axios'
import { toast } from '@/components/ui/use-toast'
import type { 
  ApiResponse,
  ApiError,
  RateLimitError,
  NetworkError,
  ValidationError,
  EnhancedApiError,
  ComprehensiveData,
  SatelliteData,
  SystemHealth,
  TelemetryData,
  ManeuverData,
  SecurityHeaders as SecurityHeadersType,
  MonitoringMetrics as MonitoringMetricsType
} from './types'
import { 
  API_CONFIG, 
  RATE_LIMIT_CONFIG, 
  SECURITY_CONFIG, 
  MONITORING_CONFIG 
} from './api-config'

const MAX_RETRIES = 3
const RETRY_DELAY = 1000

// Rate limiting state
let requestCount = 0
let windowStart = Date.now()

// Add performance monitoring types
interface PerformanceMetrics {
  endpoint: string
  method: string
  duration: number
  status: number
  timestamp: string
}

interface CircuitBreakerMetrics {
  state: 'CLOSED' | 'OPEN' | 'HALF_OPEN'
  failureCount: number
  lastFailureTime: number | null
  successCount: number
  totalRequests: number
  errorRate: number
}

// API endpoints with strong typing
export const getSatellites = async (): Promise<ApiResponse<SatelliteData[]>> => {
  try {
    const response = await apiClient.get<SatelliteData[]>('/api/v1/satellites')
    return {
      data: response.data,
      status: response.status
    }
  } catch (error) {
    console.error('Error fetching satellites:', error)
    return {
      data: null,
      error: {
        message: 'Failed to fetch satellites'
      }
    }
  }
}

export const getSystemHealth = async (): Promise<ApiResponse<SystemHealth>> => {
  try {
    const response = await apiClient.get<SystemHealth>('/api/v1/health')
    return {
      data: response.data,
      status: response.status
    }
  } catch (error) {
    console.error('Error fetching system health:', error)
    return {
      data: null,
      error: {
        message: 'Failed to fetch system health'
      }
    }
  }
}

export async function getComprehensiveData(): Promise<ApiResponse<ComprehensiveData>> {
  try {
    const response = await apiClient.get<ComprehensiveData>('/api/v1/comprehensive')
    return {
      data: response.data,
      status: response.status
    }
  } catch (error) {
    console.error('Error fetching comprehensive data:', error)
    return {
      data: null,
      error: {
        message: 'Failed to fetch comprehensive data'
      }
    }
  }
}

export async function getManeuvers(): Promise<ManeuverData[]> {
  try {
    const response = await axios.get('/api/maneuvers')
    return response.data
  } catch (error) {
    console.error('Error fetching maneuvers:', error)
    throw error
  }
}

export async function createManeuver(maneuverData: Partial<ManeuverData>): Promise<ManeuverData> {
  try {
    const response = await axios.post('/api/maneuvers', maneuverData)
    return response.data
  } catch (error) {
    console.error('Error creating maneuver:', error)
    throw error
  }
}

// Initialize API client with configuration
const apiClient = axios.create(API_CONFIG)

// Add request interceptor for authentication
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default apiClient 