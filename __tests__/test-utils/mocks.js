class MockExternalServices {
  constructor() {
    this.services = new Map();
    this.defaultResponses = new Map([
      ['trajectory-service', { status: 'success', data: { position: [0, 0, 0], velocity: [0, 0, 0] } }],
      ['telemetry-service', { status: 'success', data: { temperature: 293, pressure: 101.325 } }],
      ['command-service', { status: 'success', message: 'Command executed successfully' }]
    ]);
  }

  start() {
    this.services.clear();
    this.defaultResponses.forEach((response, service) => {
      this.services.set(service, {
        status: 'running',
        response
      });
    });
  }

  stop() {
    this.services.clear();
  }

  fail(serviceName) {
    if (this.services.has(serviceName)) {
      this.services.set(serviceName, {
        status: 'failed',
        response: { status: 'error', message: 'Service unavailable' }
      });
    }
  }

  restore(serviceName) {
    if (this.defaultResponses.has(serviceName)) {
      this.services.set(serviceName, {
        status: 'running',
        response: this.defaultResponses.get(serviceName)
      });
    }
  }

  getResponse(serviceName) {
    const service = this.services.get(serviceName);
    if (!service) {
      throw new Error(`Service ${serviceName} not found`);
    }
    return service.response;
  }

  isRunning(serviceName) {
    const service = this.services.get(serviceName);
    return service && service.status === 'running';
  }
}

const mockExternalServices = new MockExternalServices();

module.exports = {
  mockExternalServices
}; 