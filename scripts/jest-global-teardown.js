const { teardownTestDatabase } = require('../__tests__/test-utils/db');

module.exports = async () => {
  // Close server
  if (global.__TEST_SERVER__) {
    await new Promise((resolve) => {
      global.__TEST_SERVER__.close(resolve);
    });
  }

  // Clean up database
  await teardownTestDatabase();
}; 