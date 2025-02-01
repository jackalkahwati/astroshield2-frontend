// API Configuration
export const API_CONFIG = {
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  },
  withCredentials: true
}

// Rate limiting configuration
export const RATE_LIMIT_CONFIG = {
  maxRequests: 50,
  windowMs: 60000, // 1 minute
  retryAfter: 5000 // 5 seconds
}

// Cache configuration
export const CACHE_CONFIG = {
  defaultStaleAfter: 60000, // 1 minute
  defaultRefreshInterval: 30000, // 30 seconds
  maxCacheSize: 100 // Maximum number of cached responses
}

// Security configuration
export const SECURITY_CONFIG = {
  headers: {
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
}

// Monitoring configuration
export const MONITORING_CONFIG = {
  maxMetricsHistory: 1000,
  metricsWindowMs: 300000, // 5 minutes
  errorThreshold: 0.1, // 10% error rate threshold
  circuitBreakerTimeout: 60000 // 1 minute timeout for circuit breaker
}

export const WS_CONFIG = {
  url: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001/ws',
  reconnectInterval: 1000,
  maxReconnectAttempts: 5,
} 