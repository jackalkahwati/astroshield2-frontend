import axios, { AxiosError, AxiosResponse } from 'axios'
import { toast } from '@/components/ui/use-toast'

const MAX_RETRIES = 3
const RETRY_DELAY = 1000

// Rate limiting configuration
const RATE_LIMIT = {
  maxRequests: 50,
  windowMs: 60000, // 1 minute
  retryAfter: 5000 // 5 seconds
}

// Rate limiting state
let requestCount = 0
let windowStart = Date.now()

interface ApiError {
  message: string
  code: string
  details?: any
}

// Enhanced error types
interface RateLimitError extends ApiError {
  retryAfter: number
}

interface NetworkError extends ApiError {
  isNetworkError: true
  originalError: Error
}

interface ValidationError extends ApiError {
  fieldErrors: Record<string, string[]>
}

type EnhancedApiError = ApiError | RateLimitError | NetworkError | ValidationError

interface ApiResponse<T> {
  data: T | null
  error?: EnhancedApiError
  retryAfter?: number
}

// Add security and monitoring enhancements
interface SecurityHeaders {
  'Content-Security-Policy': string
  'Strict-Transport-Security': string
  'X-Content-Type-Options': string
  'X-Frame-Options': string
  'X-XSS-Protection': string
  [key: string]: string
}

interface MonitoringMetrics extends PerformanceMetrics {
  cacheHits: number
  cacheMisses: number
  deduplicationHits: number
  backgroundRefreshes: number
  memoryUsage: number
}

class MetricsCollector {
  private metrics: MonitoringMetrics[] = []
  private readonly maxMetrics = 1000
  private cacheHits = 0
  private cacheMisses = 0
  private deduplicationHits = 0
  private backgroundRefreshes = 0

  record(metric: Partial<MonitoringMetrics>) {
    this.metrics.push({
      ...metric,
      timestamp: new Date().toISOString(),
      cacheHits: this.cacheHits,
      cacheMisses: this.cacheMisses,
      deduplicationHits: this.deduplicationHits,
      backgroundRefreshes: this.backgroundRefreshes,
      memoryUsage: this.getMemoryUsage()
    } as MonitoringMetrics)

    if (this.metrics.length > this.maxMetrics) {
      this.metrics.shift()
    }
  }

  incrementCacheHits() { this.cacheHits++ }
  incrementCacheMisses() { this.cacheMisses++ }
  incrementDeduplicationHits() { this.deduplicationHits++ }
  incrementBackgroundRefreshes() { this.backgroundRefreshes++ }

  getMetrics(timeWindow?: number): MonitoringMetrics[] {
    if (!timeWindow) return this.metrics

    const cutoff = Date.now() - timeWindow
    return this.metrics.filter(m => 
      new Date(m.timestamp).getTime() > cutoff
    )
  }

  private getMemoryUsage(): number {
    if (typeof window !== 'undefined' && window.performance?.memory) {
      return window.performance.memory.usedJSHeapSize
    }
    return 0
  }
}

// Initialize metrics collector
const metricsCollector = new MetricsCollector()

// Add security monitoring
interface SecurityMetrics {
  httpsPercentage: number
  cspViolations: number
  blockedRequests: number
  rateLimited: number
  sanitizedErrors: number
  potentialLeaks: number
  timestamp: string
}

class SecurityMonitor {
  private metrics: SecurityMetrics[] = []
  private violations = {
    csp: 0,
    blocked: 0,
    rateLimit: 0,
    sanitized: 0,
    leaks: 0
  }

  record() {
    const httpsPercentage = typeof window !== 'undefined' ? 
      window.location.protocol === 'https:' ? 100 : 0 : 0

    this.metrics.push({
      httpsPercentage,
      cspViolations: this.violations.csp,
      blockedRequests: this.violations.blocked,
      rateLimited: this.violations.rateLimit,
      sanitizedErrors: this.violations.sanitized,
      potentialLeaks: this.violations.leaks,
      timestamp: new Date().toISOString()
    })

    if (this.metrics.length > 1000) {
      this.metrics.shift()
    }
  }

  incrementViolation(type: keyof typeof this.violations) {
    this.violations[type]++
    this.record()
  }

  getMetrics(timeWindow: number = 300000) {
    const cutoff = Date.now() - timeWindow
    return this.metrics.filter(m => 
      new Date(m.timestamp).getTime() > cutoff
    )
  }
}

const securityMonitor = new SecurityMonitor()

// Update security headers
const securityHeaders: SecurityHeaders = {
  'Content-Security-Policy': [
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: https:",
    "font-src 'self'",
    "connect-src 'self'",
    "report-uri /api/csp-report"
  ].join('; '),
  'Strict-Transport-Security': 'max-age=63072000; includeSubDomains; preload',
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'X-XSS-Protection': '1; mode=block',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
}

// Update API client
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.astroshield.com'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    ...securityHeaders
  },
  timeout: 10000,
  withCredentials: true
})

// Enhanced error sanitization
const sanitizeError = (error: any): ApiError => {
  // Remove sensitive information
  const sanitized: ApiError = {
    code: error.code || 'UNKNOWN_ERROR',
    message: 'An error occurred',
    details: undefined
  }

  // Only include safe error messages in production
  if (process.env.NODE_ENV !== 'production') {
    sanitized.message = error.message
    sanitized.details = error.details
  }

  return sanitized
}

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError<ApiError>) => {
    const originalRequest = error.config
    if (!originalRequest) {
      return Promise.reject(error)
    }

    // Handle network errors
    if (!error.response) {
      toast({
        variant: "destructive",
        title: "Network Error",
        description: "Please check your internet connection."
      })
      return Promise.reject(error)
    }

    // Retry logic for network errors or 5xx responses
    if (
      (error.response.status >= 500 || error.response.status === 0) &&
      originalRequest._retry !== MAX_RETRIES
    ) {
      originalRequest._retry = (originalRequest._retry || 0) + 1
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * originalRequest._retry))
      return apiClient(originalRequest)
    }

    // Handle specific error cases
    switch (error.response.status) {
      case 401:
        // Unauthorized - clear auth and redirect to login
        localStorage.removeItem('auth_token')
        window.location.href = '/login'
        break
      case 403:
        toast({
          variant: "destructive",
          title: "Access Denied",
          description: error.response.data?.message || "You don't have permission to perform this action."
        })
        break
      case 429:
        toast({
          variant: "destructive",
          title: "Rate Limited",
          description: "Please try again later."
        })
        break
      default:
        toast({
          variant: "destructive",
          title: "Error",
          description: error.response.data?.message || "An unexpected error occurred."
        })
    }

    return Promise.reject(error)
  }
)

// API endpoints with strong typing
export interface SatelliteData {
  id: string
  name: string
  status: string
  orbit: {
    altitude: number
    inclination: number
    period: number
  }
  health: {
    power: number
    thermal: number
    communication: number
  }
  telemetry?: {
    lastUpdate: string
    signalStrength: number
    temperature: number
    batteryLevel: number
  }
}

export interface ManeuverData {
  id: string
  type: string
  status: string
  scheduledTime: string
  completedTime?: string
  details: {
    deltaV: number
    duration: number
    fuel_required: number
    rotation_angle: number
    fuel_used?: number
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

export interface IndicatorData {
  timestamp: string
  indicators: {
    orbit_stability: {
      value: number
      trend: 'stable' | 'improving' | 'degrading'
      alerts: string[]
    }
    power_management: {
      value: number
      trend: 'stable' | 'improving' | 'degrading'
      alerts: string[]
    }
    thermal_control: {
      value: number
      trend: 'stable' | 'improving' | 'degrading'
      alerts: string[]
    }
    communication_quality: {
      value: number
      trend: 'stable' | 'improving' | 'degrading'
      alerts: string[]
    }
  }
}

export interface StabilityData {
  satellite_id: string
  timestamp: string
  analysis: {
    attitude_stability: {
      value: number
      confidence: number
      trend: string
    }
    orbit_stability: {
      value: number
      confidence: number
      trend: string
    }
    thermal_stability: {
      value: number
      confidence: number
      trend: string
    }
  }
  recommendations: string[]
}

export interface AnalyticsData {
  summary: {
    total_conjunctions_analyzed: number
    threats_detected: number
    threats_mitigated: number
    average_response_time: number
    protection_coverage: number
  }
  current_metrics: {
    [key: string]: {
      value: number
      trend: string
      status: string
    }
  }
  trends: {
    daily: Array<{
      timestamp: string
      conjunction_count: number
      threat_level: number
      protection_coverage: number
      response_time: number
      mitigation_success: number
    }>
    weekly_summary: {
      average_threat_level: number
      total_conjunctions: number
      mitigation_success_rate: number
      average_response_time: number
    }
    monthly_summary: {
      average_threat_level: number
      total_conjunctions: number
      mitigation_success_rate: number
      average_response_time: number
    }
  }
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

export interface ComprehensiveData {
  metrics: Record<string, number>
  status: string
  alerts: string[]
  timestamp: string
}

// Rate limiting check
function checkRateLimit(): boolean {
  const now = Date.now()
  if (now - windowStart >= RATE_LIMIT.windowMs) {
    requestCount = 0
    windowStart = now
  }
  
  if (requestCount >= RATE_LIMIT.maxRequests) {
    return false
  }
  
  requestCount++
  return true
}

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

// Update circuit breaker with metrics
class CircuitBreaker {
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED'
  private failureCount = 0
  private lastFailureTime = 0
  private readonly threshold = 5
  private readonly resetTimeout = 30000
  private readonly halfOpenMaxAttempts = 3
  private halfOpenSuccesses = 0
  private successCount = 0
  private totalRequests = 0
  private metrics: PerformanceMetrics[] = []
  private readonly maxMetricsLength = 1000

  constructor(
    private readonly onStateChange?: (state: 'CLOSED' | 'OPEN' | 'HALF_OPEN') => void
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    this.totalRequests++
    const startTime = Date.now()
    let status = 0

    try {
      if (this.state === 'OPEN') {
        if (Date.now() - this.lastFailureTime >= this.resetTimeout) {
          this.transitionToHalfOpen()
        } else {
          throw new Error('Circuit breaker is OPEN')
        }
      }

      const result = await fn()
      
      this.successCount++
      if (this.state === 'HALF_OPEN') {
        this.halfOpenSuccesses++
        if (this.halfOpenSuccesses >= this.halfOpenMaxAttempts) {
          this.reset()
        }
      }

      // Capture success metrics
      if (result && 'status' in result) {
        status = result.status
      }
      
      return result
    } catch (error) {
      this.handleFailure()
      throw error
    } finally {
      // Record metrics
      this.recordMetrics({
        endpoint: fn.toString(),
        method: fn.toString().split(' ')[0],
        duration: Date.now() - startTime,
        status,
        timestamp: new Date().toISOString()
      })
    }
  }

  private handleFailure() {
    this.failureCount++
    this.lastFailureTime = Date.now()
    
    if (this.state === 'HALF_OPEN' || this.failureCount >= this.threshold) {
      this.trip()
    }
  }

  private transitionToHalfOpen() {
    this.state = 'HALF_OPEN'
    this.halfOpenSuccesses = 0
    this.onStateChange?.(this.state)
  }

  private trip() {
    this.state = 'OPEN'
    this.onStateChange?.(this.state)
  }

  private reset() {
    this.failureCount = 0
    this.state = 'CLOSED'
    this.halfOpenSuccesses = 0
    this.onStateChange?.(this.state)
  }

  getState() {
    return this.state
  }

  private recordMetrics(metrics: PerformanceMetrics) {
    this.metrics.push(metrics)
    if (this.metrics.length > this.maxMetricsLength) {
      this.metrics.shift()
    }
  }

  getMetrics(): CircuitBreakerMetrics {
    return {
      state: this.state,
      failureCount: this.failureCount,
      lastFailureTime: this.lastFailureTime,
      successCount: this.successCount,
      totalRequests: this.totalRequests,
      errorRate: this.failureCount / (this.totalRequests || 1)
    }
  }

  getPerformanceMetrics(timeWindow?: number): PerformanceMetrics[] {
    if (!timeWindow) {
      return this.metrics
    }

    const cutoff = Date.now() - timeWindow
    return this.metrics.filter(m => 
      new Date(m.timestamp).getTime() > cutoff
    )
  }
}

// Create circuit breaker instance
const circuitBreaker = new CircuitBreaker((state) => {
  if (state === 'OPEN') {
    toast({
      variant: "warning",
      title: "Service Degraded",
      description: "Some services are currently unavailable. Retrying shortly."
    })
  } else if (state === 'CLOSED') {
    toast({
      variant: "success",
      title: "Service Restored",
      description: "All services are now operational."
    })
  }
})

// Create monitoring instance
const monitoring = {
  circuitBreaker: null as CircuitBreaker | null,
  
  initialize(breaker: CircuitBreaker) {
    this.circuitBreaker = breaker
  },

  getHealthMetrics() {
    if (!this.circuitBreaker) {
      return null
    }

    const metrics = this.circuitBreaker.getMetrics()
    const performance = this.circuitBreaker.getPerformanceMetrics(300000) // Last 5 minutes

    const avgResponseTime = performance.reduce(
      (sum, m) => sum + m.duration, 0
    ) / (performance.length || 1)

    return {
      ...metrics,
      avgResponseTime,
      recentErrors: performance.filter(m => m.status >= 400).length,
      availability: (metrics.successCount / metrics.totalRequests) * 100
    }
  }
}

// Initialize monitoring
monitoring.initialize(circuitBreaker)

// Add request batching and caching
interface BatchRequest<T> {
  key: string
  method: 'get' | 'post' | 'put' | 'delete'
  url: string
  data?: any
  resolve: (value: ApiResponse<T>) => void
  reject: (error: any) => void
}

interface CacheEntry<T> {
  data: T
  timestamp: number
  staleAfter: number
}

class RequestBatcher {
  private batchTimeout: NodeJS.Timeout | null = null
  private batchQueue: BatchRequest<any>[] = []
  private readonly batchDelay = 50 // ms
  private readonly maxBatchSize = 10

  async add<T>(request: Omit<BatchRequest<T>, 'resolve' | 'reject'>): Promise<ApiResponse<T>> {
    return new Promise((resolve, reject) => {
      this.batchQueue.push({
        ...request,
        resolve,
        reject
      })

      if (this.batchQueue.length >= this.maxBatchSize) {
        this.processBatch()
      } else if (!this.batchTimeout) {
        this.batchTimeout = setTimeout(() => this.processBatch(), this.batchDelay)
      }
    })
  }

  private async processBatch() {
    if (this.batchTimeout) {
      clearTimeout(this.batchTimeout)
      this.batchTimeout = null
    }

    const batch = this.batchQueue.splice(0, this.maxBatchSize)
    if (batch.length === 0) return

    try {
      const responses = await Promise.all(
        batch.map(request => 
          apiCall(request.method, request.url, request.data)
        )
      )

      batch.forEach((request, index) => {
        request.resolve(responses[index])
      })
    } catch (error) {
      batch.forEach(request => request.reject(error))
    }
  }
}

class CacheManager {
  private cache: Map<string, CacheEntry<any>> = new Map()
  private readonly defaultStaleAfter = 5 * 60 * 1000 // 5 minutes

  set<T>(key: string, data: T, staleAfter = this.defaultStaleAfter) {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      staleAfter
    })
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key)
    if (!entry) return null

    if (Date.now() - entry.timestamp > entry.staleAfter) {
      this.cache.delete(key)
      return null
    }

    return entry.data
  }

  clear() {
    this.cache.clear()
  }

  clearStale() {
    const now = Date.now()
    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > entry.staleAfter) {
        this.cache.delete(key)
      }
    }
  }

  async setWithBackgroundRefresh<T>(
    key: string,
    data: T,
    options: {
      staleAfter?: number
      refreshInterval?: number
      refresh: () => Promise<T>
    }
  ) {
    this.set(key, data, options.staleAfter)

    if (options.refreshInterval) {
      backgroundRefresh.schedule(
        key,
        async () => {
          try {
            const freshData = await options.refresh()
            this.set(key, freshData, options.staleAfter)
          } catch (error) {
            console.error('Background refresh failed:', error)
          }
        },
        options.refreshInterval
      )
    }
  }
}

const batcher = new RequestBatcher()
const cache = new CacheManager()

// Add request deduplication and background refresh
interface PendingRequest<T> {
  promise: Promise<ApiResponse<T>>
  timestamp: number
}

class RequestDeduplicator {
  private pendingRequests = new Map<string, PendingRequest<any>>()
  private readonly pendingTimeout = 5000 // 5 seconds

  async execute<T>(
    key: string,
    fn: () => Promise<ApiResponse<T>>
  ): Promise<ApiResponse<T>> {
    // Clean up old pending requests
    this.cleanup()

    // Check for pending request
    const pending = this.pendingRequests.get(key)
    if (pending && Date.now() - pending.timestamp < this.pendingTimeout) {
      return pending.promise
    }

    // Create new request
    const promise = fn()
    this.pendingRequests.set(key, {
      promise,
      timestamp: Date.now()
    })

    try {
      const result = await promise
      this.pendingRequests.delete(key)
      return result
    } catch (error) {
      this.pendingRequests.delete(key)
      throw error
    }
  }

  private cleanup() {
    const now = Date.now()
    for (const [key, request] of this.pendingRequests.entries()) {
      if (now - request.timestamp >= this.pendingTimeout) {
        this.pendingRequests.delete(key)
      }
    }
  }
}

class BackgroundRefreshManager {
  private refreshTimers = new Map<string, NodeJS.Timeout>()
  private refreshing = new Set<string>()

  schedule(
    key: string,
    fn: () => Promise<void>,
    refreshInterval: number
  ) {
    if (this.refreshTimers.has(key)) {
      return
    }

    const timer = setInterval(async () => {
      if (this.refreshing.has(key)) {
        return
      }

      this.refreshing.add(key)
      try {
        await fn()
      } finally {
        this.refreshing.delete(key)
      }
    }, refreshInterval)

    this.refreshTimers.set(key, timer)
  }

  cancel(key: string) {
    const timer = this.refreshTimers.get(key)
    if (timer) {
      clearInterval(timer)
      this.refreshTimers.delete(key)
    }
  }

  clear() {
    for (const timer of this.refreshTimers.values()) {
      clearInterval(timer)
    }
    this.refreshTimers.clear()
    this.refreshing.clear()
  }
}

const deduplicator = new RequestDeduplicator()
const backgroundRefresh = new BackgroundRefreshManager()

// Update API call function with security monitoring
async function apiCall<T>(
  method: 'get' | 'post' | 'put' | 'delete',
  url: string,
  data?: any,
  options: {
    bypassCache?: boolean
    staleAfter?: number
    bypassBatch?: boolean
    refreshInterval?: number
  } = {}
): Promise<ApiResponse<T>> {
  const startTime = Date.now()
  const cacheKey = `${method}:${url}:${JSON.stringify(data)}`

  try {
    // Check cache for GET requests
    if (method === 'get' && !options.bypassCache) {
      const cached = cache.get<T>(cacheKey)
      if (cached) {
        metricsCollector.incrementCacheHits()
        return { data: cached }
      }
      metricsCollector.incrementCacheMisses()
    }

    // Check rate limit
    if (!checkRateLimit()) {
      securityMonitor.incrementViolation('rateLimit')
      return {
        data: null,
        error: {
          code: 'RATE_LIMIT_EXCEEDED',
          message: 'Too many requests',
          retryAfter: RATE_LIMIT.retryAfter
        } as RateLimitError,
        retryAfter: RATE_LIMIT.retryAfter
      }
    }

    // Deduplicate requests
    return deduplicator.execute(cacheKey, async () => {
      try {
        const response = await circuitBreaker.execute(async () => {
          const response = await apiClient({
            method,
            url,
            data,
          })
          
          if (method === 'get' && response.data && !options.bypassCache) {
            await cache.setWithBackgroundRefresh(cacheKey, response.data, {
              staleAfter: options.staleAfter,
              refreshInterval: options.refreshInterval,
              refresh: async () => {
                metricsCollector.incrementBackgroundRefreshes()
                const fresh = await apiCall<T>(method, url, data, {
                  bypassCache: true,
                  bypassBatch: true
                })
                return fresh.data!
              }
            })
          }

          return response
        })

        return { data: response.data }
      } catch (error) {
        if (error instanceof AxiosError) {
          // Check for sensitive data in error
          const errorStr = JSON.stringify(error.response?.data || error)
          if (
            /password|token|key|secret|credential/i.test(errorStr)
          ) {
            securityMonitor.incrementViolation('leaks')
          }

          const sanitizedError = sanitizeError(error.response?.data || error)
          securityMonitor.incrementViolation('sanitized')
          return {
            data: null,
            error: sanitizedError
          }
        }
        throw error
      } finally {
        metricsCollector.record({
          endpoint: url,
          method,
          duration: Date.now() - startTime,
          status: 200 // Updated by error handlers if needed
        })
      }
    })
  } catch (error) {
    securityMonitor.incrementViolation('blocked')
    const sanitizedError = sanitizeError(error)
    return {
      data: null,
      error: sanitizedError
    }
  }
}

// Strongly typed API functions
export const getSatellites = (options?: { bypassCache?: boolean }) => 
  apiCall<SatelliteData[]>('get', '/api/satellites', undefined, options)

export const getSatelliteById = (id: string) => 
  apiCall<SatelliteData>('get', `/api/satellites/${id}`)

export const getTelemetryData = (
  satelliteId: string,
  options?: { staleAfter?: number }
) => apiCall<TelemetryData>(
  'get',
  `/api/telemetry/${satelliteId}`,
  undefined,
  {
    staleAfter: options?.staleAfter || 30000, // 30s
    refreshInterval: 15000 // Refresh every 15s
  }
)

export const getManeuvers = () => apiCall<ManeuverData[]>('get', '/api/v1/maneuvers')

export const createManeuver = (data: Omit<ManeuverData, 'id'>) => 
  apiCall<ManeuverData>('post', '/api/v1/maneuvers', data)

export const getAnalytics = () => 
  apiCall<AnalyticsData>('get', '/api/v1/analytics/data')

export const getSystemHealth = () => apiCall<SystemHealth>(
  'get',
  '/api/v1/health',
  undefined,
  {
    staleAfter: 60000, // 1 minute
    refreshInterval: 30000 // Refresh every 30s
  }
)

export const getIndicators = () => 
  apiCall<IndicatorData>('get', '/api/v1/indicators')

export const getStabilityAnalysis = (satelliteId: string) => 
  apiCall<StabilityData>('get', `/api/v1/stability/${satelliteId}`)

export const getComprehensiveData = () => 
  apiCall<ComprehensiveData>('get', '/api/v1/comprehensive/data')

// Export enhanced monitoring functions
export const getApiMetrics = () => ({
  ...monitoring.getHealthMetrics(),
  performance: metricsCollector.getMetrics(300000), // Last 5 minutes
  cacheEfficiency: {
    hits: metricsCollector.getMetrics()[0]?.cacheHits || 0,
    misses: metricsCollector.getMetrics()[0]?.cacheMisses || 0,
    ratio: (hits: number, misses: number) => 
      hits / (hits + misses) || 0
  }
})

// Add cache management to exports
export const clearApiCache = () => cache.clear()
export const clearStaleCache = () => cache.clearStale()

// Add cleanup on window unload
if (typeof window !== 'undefined') {
  window.addEventListener('unload', () => {
    backgroundRefresh.clear()
    cache.clear()
  })
}

// Export security metrics
export const getSecurityMetrics = () => ({
  ...getApiMetrics(),
  security: {
    current: securityMonitor.getMetrics(),
    violations: securityMonitor.violations
  }
})

export default apiClient 