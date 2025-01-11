const { performance } = require('perf_hooks');
const { api } = require('../test-utils/config');

describe('Performance Benchmarks', () => {
  describe('API Response Time', () => {
    test('spacecraft status endpoint should respond within 200ms', async () => {
      const startTime = performance.now();
      
      await api.get('/spacecraft/test-1/status');
      
      const endTime = performance.now();
      const responseTime = endTime - startTime;
      
      expect(responseTime).toBeLessThan(200);
    });
  });

  describe('Throughput Tests', () => {
    test('should handle 100 concurrent status requests', async () => {
      const startTime = performance.now();
      
      // Generate 100 concurrent requests
      const requests = Array(100).fill().map(() => 
        api.get('/spacecraft/test-1/status')
      );
      
      await Promise.all(requests);
      
      const endTime = performance.now();
      const totalTime = endTime - startTime;
      const requestsPerSecond = (requests.length / totalTime) * 1000;
      
      expect(requestsPerSecond).toBeGreaterThan(50); // At least 50 req/sec
    });
  });

  describe('Resource Usage', () => {
    test('memory usage should stay within limits during load test', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Simulate heavy load
      const requests = Array(1000).fill().map(() => ({
        spacecraft_id: `test-${Math.random()}`,
        data: new Array(1000).fill(Math.random())
      }));
      
      for (const request of requests) {
        await api.post('/spacecraft/telemetry', request);
      }
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = (finalMemory - initialMemory) / 1024 / 1024; // MB
      
      expect(memoryIncrease).toBeLessThan(50); // Less than 50MB increase
    });

    test('CPU usage should remain reasonable under load', async () => {
      const startUsage = process.cpuUsage();
      
      // Perform compute-intensive operation
      await api.post('/spacecraft/analyze/trajectory', {
        data_points: Array(10000).fill().map((_, i) => ({
          time: i,
          position: [Math.random(), Math.random(), Math.random()]
        }))
      });
      
      const endUsage = process.cpuUsage(startUsage);
      const cpuTimeMS = (endUsage.user + endUsage.system) / 1000; // ms
      
      expect(cpuTimeMS).toBeLessThan(1000); // Should process within 1 second
    });
  });

  describe('Data Processing Performance', () => {
    test('should process large telemetry batch within time limit', async () => {
      const batchSize = 10000;
      const telemetryBatch = Array(batchSize).fill().map((_, i) => ({
        timestamp: new Date(Date.now() + i * 1000).toISOString(),
        readings: {
          position: [Math.random() * 1000, Math.random() * 1000, Math.random() * 1000],
          velocity: [Math.random() * 10, Math.random() * 10, Math.random() * 10]
        }
      }));

      const startTime = performance.now();
      
      await api.post('/spacecraft/telemetry/batch', {
        spacecraft_id: 'test-1',
        data: telemetryBatch
      });
      
      const processingTime = performance.now() - startTime;
      const recordsPerSecond = batchSize / (processingTime / 1000);
      
      expect(recordsPerSecond).toBeGreaterThan(1000); // Should process >1000 records/sec
    });
  });
}); 