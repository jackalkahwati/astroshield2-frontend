const winston = require('winston');
const { createLogger } = require('../../utils/logger');
const metrics = require('../../utils/metrics');
const tracing = require('../../utils/tracing');

describe('Monitoring & Logging', () => {
  describe('Logging System', () => {
    let logger;

    beforeEach(() => {
      logger = createLogger();
    });

    test('logs are properly formatted', () => {
      const logMessage = 'Test log message';
      const logData = { key: 'value' };
      
      const logEntry = logger.info(logMessage, logData);
      
      expect(logEntry).toMatchObject({
        level: 'info',
        message: logMessage,
        timestamp: expect.any(String),
        metadata: expect.objectContaining(logData)
      });
    });

    test('error logs include stack traces', () => {
      const error = new Error('Test error');
      const logEntry = logger.error('Error occurred', { error });
      
      expect(logEntry.error.stack).toBeDefined();
      expect(logEntry.metadata.errorCode).toBeDefined();
    });

    test('sensitive data is masked in logs', () => {
      const sensitiveData = {
        password: 'secret123',
        apiKey: 'sk_test_123',
        creditCard: '4242-4242-4242-4242'
      };
      
      const logEntry = logger.info('User data', sensitiveData);
      
      expect(logEntry.metadata.password).toBe('[REDACTED]');
      expect(logEntry.metadata.apiKey).toBe('[REDACTED]');
      expect(logEntry.metadata.creditCard).toBe('[REDACTED]');
    });
  });

  describe('Metrics Collection', () => {
    beforeEach(() => {
      metrics.reset();
    });

    test('records API response times', async () => {
      const startTime = Date.now();
      await metrics.recordAPILatency('/api/test', 'GET', startTime);
      
      const histogram = metrics.getHistogram('api_latency');
      expect(histogram.count).toBe(1);
    });

    test('tracks error rates', () => {
      metrics.incrementErrors('api_error', { endpoint: '/api/test' });
      metrics.incrementErrors('api_error', { endpoint: '/api/test' });
      
      const counter = metrics.getCounter('error_total');
      expect(counter.labels('/api/test').value).toBe(2);
    });

    test('monitors system resources', () => {
      const resources = metrics.collectSystemMetrics();
      
      expect(resources).toMatchObject({
        cpu_usage: expect.any(Number),
        memory_usage: expect.any(Number),
        disk_usage: expect.any(Number),
        network_io: expect.any(Object)
      });
    });
  });

  describe('Alert System', () => {
    test('triggers alerts on error threshold', () => {
      const alertManager = require('../../utils/alerts');
      
      // Simulate errors
      for (let i = 0; i < 10; i++) {
        metrics.incrementErrors('critical_error');
      }
      
      expect(alertManager.getActiveAlerts()).toContainEqual(
        expect.objectContaining({
          type: 'error_threshold',
          severity: 'critical'
        })
      );
    });

    test('triggers alerts on high latency', async () => {
      const alertManager = require('../../utils/alerts');
      
      // Simulate slow requests
      for (let i = 0; i < 5; i++) {
        await metrics.recordAPILatency('/api/slow', 'GET', Date.now() - 5000);
      }
      
      expect(alertManager.getActiveAlerts()).toContainEqual(
        expect.objectContaining({
          type: 'high_latency',
          severity: 'warning'
        })
      );
    });
  });

  describe('Distributed Tracing', () => {
    test('generates trace IDs for requests', () => {
      const trace = tracing.startTrace('test-operation');
      expect(trace.traceId).toMatch(/^[0-9a-f]{32}$/);
    });

    test('propagates context between services', async () => {
      const parentTrace = tracing.startTrace('parent-operation');
      const childTrace = await tracing.startChildSpan('child-operation', parentTrace);
      
      expect(childTrace.parentId).toBe(parentTrace.spanId);
      expect(childTrace.traceId).toBe(parentTrace.traceId);
    });

    test('records span attributes', () => {
      const trace = tracing.startTrace('test-operation');
      trace.setAttribute('test.key', 'value');
      
      expect(trace.attributes['test.key']).toBe('value');
    });
  });

  describe('Health Monitoring', () => {
    test('checks database connectivity', async () => {
      const health = await metrics.checkDatabaseHealth();
      expect(health.status).toBe('healthy');
      expect(health.latency).toBeLessThan(100);
    });

    test('checks external service dependencies', async () => {
      const dependencies = await metrics.checkDependencies();
      
      expect(dependencies).toEqual(
        expect.objectContaining({
          redis: expect.objectContaining({ status: 'up' }),
          kafka: expect.objectContaining({ status: 'up' }),
          elasticsearch: expect.objectContaining({ status: 'up' })
        })
      );
    });

    test('monitors queue depths', async () => {
      const queueMetrics = await metrics.getQueueMetrics();
      
      expect(queueMetrics).toMatchObject({
        messageQueue: expect.any(Number),
        deadLetterQueue: expect.any(Number),
        processingRate: expect.any(Number)
      });
    });
  });
}); 