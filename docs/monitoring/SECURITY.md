# Security Monitoring System

## Overview
The security monitoring system provides real-time tracking of security-related metrics and alerts for potential security issues. The system consists of a metrics collector service, REST API endpoints, and a frontend dashboard.

## Metrics Tracked

| Metric | Description | Warning Threshold | Critical Threshold |
|--------|-------------|------------------|-------------------|
| HTTPS Usage | Percentage of HTTPS requests | N/A | < 95% |
| CSP Violations | Content Security Policy violations | 5 | 10 |
| Blocked Requests | Suspicious requests blocked | 10 | 20 |
| Rate Limited | Number of rate-limited requests | 20 | 50 |
| Sanitized Errors | Count of sanitized error responses | 50 | 100 |
| Potential Leaks | Detected potential data leaks | 1 | 5 |

## API Endpoints

### Get Security Metrics
```
GET /api/security/metrics
Response: {
  "security": {
    "current": [{
      "httpsPercentage": number,
      "cspViolations": number,
      "blockedRequests": number,
      "rateLimited": number,
      "sanitizedErrors": number,
      "potentialLeaks": number,
      "timestamp": string
    }]
  }
}
```

### Get Active Alerts
```
GET /api/security/alerts
Response: [{
  "type": string,
  "severity": "warning" | "critical",
  "value": number,
  "threshold": number,
  "timestamp": string
}]
```

## Alert System

Alerts are generated when metrics exceed defined thresholds. The system checks for alerts every 5 minutes and metrics are reset hourly.

### Alert Severities
- **Warning**: Initial threshold breach
- **Critical**: Severe threshold breach requiring immediate attention

### Response Actions
1. **Warning Alerts**
   - Log alert details
   - Display in dashboard
   - Notify on-call team via Slack

2. **Critical Alerts**
   - Immediate notification to security team
   - Incident creation in tracking system
   - Automatic mitigation steps where applicable

## Dashboard

The security dashboard provides:
- Real-time metrics visualization
- Active alerts display
- Historical trends
- Threat level assessment
- Security score calculation

## Maintenance

1. **Daily Tasks**
   - Review alert history
   - Verify metrics collection
   - Check dashboard functionality

2. **Weekly Tasks**
   - Analyze trends
   - Update thresholds if needed
   - Review false positives

3. **Monthly Tasks**
   - Full system audit
   - Threshold optimization
   - Documentation update 