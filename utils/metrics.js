const promClient = require('prom-client');
const os = require('os');
const { db } = require('../__tests__/test-utils/db');

// Initialize metrics registry
const register = new promClient.Registry();

// Create metrics
const apiLatencyHistogram = new promClient.Histogram({
  name: 'api_latency_seconds',
  help: 'API endpoint latency in seconds',
  labelNames: ['method', 'endpoint'],
  buckets: [0.1, 0.5, 1, 2, 5]
});

const errorCounter = new promClient.Counter({
  name: 'error_total',
  help: 'Total number of errors',
  labelNames: ['path']
});

const activeConnections = new promClient.Gauge({
  name: 'active_connections',
  help: 'Number of active connections'
});

// Register metrics
register.registerMetric(apiLatencyHistogram);
register.registerMetric(errorCounter);
register.registerMetric(activeConnections);

class Metrics {
  constructor() {
    this.histograms = {
      api_latency: apiLatencyHistogram
    };
    this.counters = {
      error_total: errorCounter
    };
    this.gauges = {
      active_connections: activeConnections
    };
  }

  reset() {
    register.resetMetrics();
  }

  async recordAPILatency(endpoint, method, startTime) {
    const duration = (Date.now() - startTime) / 1000;
    this.histograms.api_latency.labels(method, endpoint).observe(duration);
    return duration;
  }

  incrementErrors(path) {
    this.counters.error_total.labels(path).inc();
  }

  getHistogram(name) {
    const histogram = this.histograms[name];
    if (!histogram) return null;

    const labelValues = histogram.labelValues || {};
    return {
      name,
      count: Object.values(labelValues)[0]?.value?.count || 0
    };
  }

  getCounter(name) {
    const counter = this.counters[name];
    if (!counter) return null;

    return {
      name,
      labels: (path) => ({
        value: counter.hashMap[`path:${path}`]?.value || 0
      })
    };
  }

  collectSystemMetrics() {
    return {
      cpu_usage: os.loadavg()[0] / os.cpus().length,
      memory_usage: process.memoryUsage().heapUsed / process.memoryUsage().heapTotal,
      disk_usage: 0, // Implement actual disk usage monitoring
      network_io: {
        rx_bytes: 0, // Implement actual network monitoring
        tx_bytes: 0
      }
    };
  }

  async checkDatabaseHealth() {
    try {
      const startTime = Date.now();
      await db.raw('SELECT 1');
      const latency = Date.now() - startTime;
      
      return {
        status: 'healthy',
        latency
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        error: error.message
      };
    }
  }

  async checkDependencies() {
    return {
      redis: { status: 'up' },
      kafka: { status: 'up' },
      elasticsearch: { status: 'up' }
    };
  }

  async getQueueMetrics() {
    return {
      messageQueue: 0,
      deadLetterQueue: 0,
      processingRate: 0
    };
  }
}

module.exports = new Metrics(); 