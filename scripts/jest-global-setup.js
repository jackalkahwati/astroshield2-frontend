const { setupTestDatabase } = require('../__tests__/test-utils/db');
const server = require('./start-test-server');

module.exports = async () => {
  // Set up test database
  await setupTestDatabase();

  // Store server reference for teardown
  global.__TEST_SERVER__ = server;
}; 