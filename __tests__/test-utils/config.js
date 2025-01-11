const axios = require('axios');

// Configure axios for tests
const api = axios.create({
  baseURL: process.env.API_URL || 'http://localhost:3000',
  validateStatus: null, // Don't throw on any status
  headers: {
    'Content-Type': 'application/json'
  }
});

// Add request interceptor to handle CSRF tokens
api.interceptors.request.use(config => {
  // Skip CSRF for test environment
  if (process.env.NODE_ENV === 'test') {
    return config;
  }

  const token = document.cookie.replace(/(?:(?:^|.*;\s*)_csrf\s*=\s*([^;]*).*$)|^.*$/, '$1');
  if (token) {
    config.headers['CSRF-Token'] = token;
  }
  return config;
});

// Test database configuration
const dbConfig = {
  host: process.env.TEST_DB_HOST || 'localhost',
  port: process.env.TEST_DB_PORT || 5432,
  database: process.env.TEST_DB_NAME || 'astroshield_test',
  user: process.env.TEST_DB_USER || 'postgres',
  password: process.env.TEST_DB_PASSWORD || 'postgres'
};

// Test JWT configuration
const jwtConfig = {
  secret: process.env.TEST_JWT_SECRET || 'test-secret-key',
  expiresIn: '1h'
};

module.exports = {
  api,
  dbConfig,
  jwtConfig
}; 