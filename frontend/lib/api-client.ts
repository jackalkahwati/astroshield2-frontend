// API client for the AstroShield backend
import { API_CONFIG } from './api-config';
import type { ManeuverData } from './types';
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
  SecurityHeaders as SecurityHeadersType,
  MonitoringMetrics as MonitoringMetricsType,
  TrajectoryRequest,
  TrajectoryResult,
  TrajectoryConfig
} from './types'
import { 
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

// Helper function to handle fetch responses
const handleResponse = async (response: Response) => {
  try {
    const data = await response.json();
    if (!response.ok) {
      return {
        data: null,
        error: { message: data.message || `Error: ${response.status}` }
      };
    }
    return { data, status: response.status };
  } catch (error) {
    return {
      data: null,
      error: { message: 'Failed to parse response' }
    };
  }
};

// Initialize API client with configuration
const apiClient = axios.create({
  ...API_CONFIG,
  withCredentials: false // Disable credentials for mock server
})

// Add request interceptor with debugging
apiClient.interceptors.request.use(
  (config) => {
    console.log('Making request to:', config.url)
    return config
  },
  (error) => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

// Add response interceptor with debugging
apiClient.interceptors.response.use(
  (response) => {
    console.log('Received response from:', response.config.url, response.status)
    return response
  },
  async (error) => {
    console.error('Response error:', error.message, error?.response?.status, error?.config?.url)
    return Promise.reject(error)
  }
)

// API endpoints with strong typing
export const getSatellites = async (): Promise<ApiResponse<SatelliteData[]>> => {
  try {
    const response = await apiClient.get<SatelliteData[]>('/satellites')
    return {
      data: response.data,
      status: response.status
    }
  } catch (error) {
    console.error('Error fetching satellites:', error)
    return {
      data: null,
      error: {
        message: 'Failed to fetch satellites data'
      }
    }
  }
}

export const getSystemHealth = async (): Promise<ApiResponse<SystemHealth>> => {
  try {
    const response = await apiClient.get<SystemHealth>('/health')
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

export const getComprehensiveData = async (): Promise<ApiResponse<ComprehensiveData>> => {
  try {
    const response = await apiClient.get<ComprehensiveData>('/comprehensive')
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

export async function getManeuvers(): Promise<ApiResponse<ManeuverData[]>> {
  try {
    // Use direct fetch for simpler debugging
    console.log('Fetching maneuvers via direct fetch')
    const url = `${API_CONFIG.baseURL}/maneuvers`
    console.log('Fetch URL:', url)
    
    const response = await fetch(url)
    console.log('Fetch response status:', response.status)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json()
    console.log('Maneuvers data received:', data)
    
    return {
      data: data,
      status: response.status
    }
  } catch (error) {
    console.error('Error fetching maneuvers:', error)
    return {
      data: null,
      error: {
        message: `Failed to fetch maneuvers: ${error instanceof Error ? error.message : 'Unknown error'}`
      }
    }
  }
}

export async function createManeuver(maneuverData: Partial<ManeuverData>): Promise<ApiResponse<ManeuverData>> {
  try {
    const response = await apiClient.post<ManeuverData>('/maneuvers', maneuverData)
    return {
      data: response.data,
      status: response.status
    }
  } catch (error) {
    console.error('Error creating maneuver:', error)
    return {
      data: null,
      error: {
        message: 'Failed to create maneuver'
      }
    }
  }
}

// Event API calls
export const eventsApi = {
  // Get dashboard data with optional date range
  getDashboardData: async (startTime?: string, endTime?: string) => {
    let url = `${API_CONFIG.baseURL}/events/dashboard`;
    
    // Add query parameters if provided
    if (startTime || endTime) {
      const params = new URLSearchParams();
      if (startTime) params.append('start_time', startTime);
      if (endTime) params.append('end_time', endTime);
      url += `?${params.toString()}`;
    }
    
    const response = await fetch(url);
    return handleResponse(response);
  },
  
  // Query events with filters
  queryEvents: async (filters: any) => {
    const response = await fetch(`${API_CONFIG.baseURL}/events/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(filters),
    });
    return handleResponse(response);
  },
  
  // Get a specific event by ID
  getEvent: async (eventId: string) => {
    const response = await fetch(`${API_CONFIG.baseURL}/events/${eventId}`);
    return handleResponse(response);
  },
  
  // Submit sensor data for event detection
  detectEvent: async (sensorData: any) => {
    const response = await fetch(`${API_CONFIG.baseURL}/events/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(sensorData),
    });
    return handleResponse(response);
  },
  
  // Process pending events
  processPendingEvents: async () => {
    const response = await fetch(`${API_CONFIG.baseURL}/events/process-pending`, {
      method: 'POST',
    });
    return handleResponse(response);
  },
  
  // Get all event types
  getEventTypes: async () => {
    const response = await fetch(`${API_CONFIG.baseURL}/events/types`);
    return handleResponse(response);
  },
  
  // Get all event status values
  getEventStatuses: async () => {
    const response = await fetch(`${API_CONFIG.baseURL}/events/status`);
    return handleResponse(response);
  },
  
  // Get all threat levels
  getThreatLevels: async () => {
    const response = await fetch(`${API_CONFIG.baseURL}/events/threat-levels`);
    return handleResponse(response);
  },
};

// Export a default API client that contains all sub-APIs
// Trajectory API functions
export const trajectoryApi = {
  // Analyze trajectory with given configuration and initial state
  analyzeTrajectory: async (request: TrajectoryRequest): Promise<ApiResponse<TrajectoryResult>> => {
    try {
      const response = await apiClient.post<TrajectoryResult>('/trajectory/analyze', request)
      return {
        data: response.data,
        status: response.status
      }
    } catch (error) {
      console.error('Error analyzing trajectory:', error)
      
      const errorMsg = error instanceof Error ? error.message : 'Unknown error'
      
      // Show toast notification for error
      toast({
        title: "Trajectory Analysis Failed",
        description: `Could not analyze trajectory: ${errorMsg}`,
        variant: "destructive",
      })
      
      return {
        data: null,
        error: {
          message: `Failed to analyze trajectory: ${errorMsg}`
        }
      }
    }
  },
  
  // Get saved trajectories
  getSavedTrajectories: async (): Promise<ApiResponse<any[]>> => {
    try {
      const response = await apiClient.get<any[]>('/trajectories/')
      return {
        data: response.data,
        status: response.status
      }
    } catch (error) {
      console.error('Error fetching saved trajectories:', error)
      return {
        data: null,
        error: {
          message: 'Failed to fetch saved trajectories'
        }
      }
    }
  },
  
  // Get trajectory by ID
  getTrajectoryById: async (id: number): Promise<ApiResponse<any>> => {
    try {
      const response = await apiClient.get<any>(`/trajectories/${id}`)
      return {
        data: response.data,
        status: response.status
      }
    } catch (error) {
      console.error(`Error fetching trajectory ${id}:`, error)
      return {
        data: null,
        error: {
          message: `Failed to fetch trajectory ${id}`
        }
      }
    }
  },
  
  // Save trajectory
  saveTrajectory: async (data: any): Promise<ApiResponse<any>> => {
    try {
      const response = await apiClient.post<any>('/trajectories/', data)
      return {
        data: response.data,
        status: response.status
      }
    } catch (error) {
      console.error('Error saving trajectory:', error)
      return {
        data: null,
        error: {
          message: 'Failed to save trajectory'
        }
      }
    }
  },
  
  // Update trajectory
  updateTrajectory: async (id: number, data: any): Promise<ApiResponse<any>> => {
    try {
      const response = await apiClient.put<any>(`/trajectories/${id}`, data)
      return {
        data: response.data,
        status: response.status
      }
    } catch (error) {
      console.error(`Error updating trajectory ${id}:`, error)
      return {
        data: null,
        error: {
          message: `Failed to update trajectory ${id}`
        }
      }
    }
  },
  
  // Delete trajectory
  deleteTrajectory: async (id: number): Promise<ApiResponse<void>> => {
    try {
      const response = await apiClient.delete(`/trajectories/${id}`)
      return {
        data: null,
        status: response.status
      }
    } catch (error) {
      console.error(`Error deleting trajectory ${id}:`, error)
      return {
        data: null,
        error: {
          message: `Failed to delete trajectory ${id}`
        }
      }
    }
  },
  
  // Compare trajectories
  compareTrajectories: async (data: any): Promise<ApiResponse<any>> => {
    try {
      const response = await apiClient.post<any>('/trajectory/compare', data)
      return {
        data: response.data,
        status: response.status
      }
    } catch (error) {
      console.error('Error comparing trajectories:', error)
      return {
        data: null,
        error: {
          message: 'Failed to compare trajectories'
        }
      }
    }
  }
}

export default apiClient