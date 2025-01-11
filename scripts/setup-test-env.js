// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.PORT = process.env.TEST_PORT || 3000;
process.env.API_URL = `http://localhost:${process.env.PORT}`;

// Configure test timeouts
jest.setTimeout(30000); // 30 seconds

// Mock external services
const { mockExternalServices } = require('../__tests__/test-utils/mocks');
mockExternalServices.start(); 