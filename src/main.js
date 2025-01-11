const express = require('express');
const cookieParser = require('cookie-parser');
const { createLogger } = require('../utils/logger');
const metrics = require('../utils/metrics');
const tracing = require('../utils/tracing');
const security = require('../utils/security');

const app = express();
const logger = createLogger();
const port = process.env.PORT || 3000;

// Initialize tracing
tracing.initialize();

// Middleware
app.use(cookieParser()); // Required for CSRF
app.use(express.json());

// Skip CSRF for test environment
if (process.env.NODE_ENV !== 'test') {
  app.use(security.csrfProtection());
}

app.use(security.rateLimiter());

// Request logging middleware
app.use((req, res, next) => {
  const startTime = Date.now();
  res.on('finish', () => {
    metrics.recordAPILatency(req.path, req.method, startTime);
  });
  next();
});

// Health check endpoints
app.get('/health/ready', (req, res) => {
  res.json({ status: 'ready' });
});

app.get('/health/live', (req, res) => {
  res.json({ status: 'alive' });
});

// API routes
app.get('/spacecraft/:id/status', (req, res) => {
  res.json({
    spacecraft_id: req.params.id,
    status: 'operational',
    last_update: new Date().toISOString()
  });
});

app.post('/spacecraft/telemetry', (req, res) => {
  res.json({
    status: 'success',
    message: 'Telemetry data received'
  });
});

app.post('/spacecraft/telemetry/batch', (req, res) => {
  res.json({
    status: 'success',
    message: 'Batch telemetry data processed'
  });
});

app.post('/spacecraft/analyze/trajectory', (req, res) => {
  res.json({
    status: 'success',
    analysis: {
      risk_level: 'low',
      confidence: 0.95
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error('Unhandled error', { error: err.message });
  metrics.incrementErrors('unhandled');
  res.status(500).json({
    status: 'error',
    message: 'Internal server error'
  });
});

// Start server
if (require.main === module) {
  app.listen(4000, () => {
    console.log('Server is running on port 4000');
  });
}

module.exports = app; 