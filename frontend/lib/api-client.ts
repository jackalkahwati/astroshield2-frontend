import axios from 'axios'

const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Satellite Data
export const getSatellites = async () => {
  const response = await apiClient.get('/satellites')
  return response.data
}

export const getSatelliteById = async (id: string) => {
  const response = await apiClient.get(`/satellites/${id}`)
  return response.data
}

// Telemetry Data
export const getTelemetryData = async (satelliteId: string) => {
  const response = await apiClient.get(`/telemetry/${satelliteId}`)
  return response.data
}

// Maneuvers
export const getManeuvers = async () => {
  const response = await apiClient.get('/maneuvers')
  return response.data
}

export const createManeuver = async (data: any) => {
  const response = await apiClient.post('/maneuvers', data)
  return response.data
}

// Analytics
export const getAnalytics = async () => {
  const response = await apiClient.get('/analytics')
  return response.data
}

// System Health
export const getSystemHealth = async () => {
  const response = await apiClient.get('/health')
  return response.data
}

// Indicators
export const getIndicators = async () => {
  const response = await apiClient.get('/indicators')
  return response.data
}

// Stability Analysis
export const getStabilityAnalysis = async (satelliteId: string) => {
  const response = await apiClient.get(`/stability/${satelliteId}`)
  return response.data
}

// Error handling interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export default apiClient 