const { createLogger } = require('./logger');

class AlertManager {
  constructor() {
    this.logger = createLogger();
    this.thresholds = {
      error_rate: 10, // errors per minute
      latency: 1000, // ms
      memory_usage: 0.9, // 90% of max
      cpu_usage: 0.8 // 80% of max
    };
    this.alerts = new Map();
  }

  checkErrorRate(errors, timeWindow) {
    const errorsPerMinute = (errors / timeWindow) * 60000;
    if (errorsPerMinute > this.thresholds.error_rate) {
      this.triggerAlert('error_rate', {
        current: errorsPerMinute,
        threshold: this.thresholds.error_rate,
        message: `Error rate of ${errorsPerMinute.toFixed(2)} errors/min exceeds threshold`
      });
    }
  }

  checkLatency(latency) {
    if (latency > this.thresholds.latency) {
      this.triggerAlert('latency', {
        current: latency,
        threshold: this.thresholds.latency,
        message: `Latency of ${latency}ms exceeds threshold`
      });
    }
  }

  checkResourceUsage(metrics) {
    if (metrics.memory_usage > this.thresholds.memory_usage) {
      this.triggerAlert('memory', {
        current: metrics.memory_usage,
        threshold: this.thresholds.memory_usage,
        message: 'Memory usage exceeds threshold'
      });
    }

    if (metrics.cpu_usage > this.thresholds.cpu_usage) {
      this.triggerAlert('cpu', {
        current: metrics.cpu_usage,
        threshold: this.thresholds.cpu_usage,
        message: 'CPU usage exceeds threshold'
      });
    }
  }

  triggerAlert(type, data) {
    const alert = {
      type,
      timestamp: new Date().toISOString(),
      ...data
    };

    this.alerts.set(type, alert);
    this.logger.warn('Alert triggered', alert);

    // In production, you would send this to your alerting system
    if (process.env.NODE_ENV === 'production') {
      this.notifyTeam(alert);
    }
  }

  clearAlert(type) {
    if (this.alerts.has(type)) {
      this.alerts.delete(type);
      this.logger.info(`Alert cleared: ${type}`);
    }
  }

  getActiveAlerts() {
    return Array.from(this.alerts.values());
  }

  async notifyTeam(alert) {
    // Implement your notification logic here
    // This could be sending emails, Slack messages, etc.
    this.logger.info('Team notification would be sent here', alert);
  }
}

module.exports = new AlertManager(); 