const app = require('../src/main');
const { createLogger } = require('../utils/logger');

const logger = createLogger();
const port = process.env.TEST_PORT || 3000;

const server = app.listen(port, () => {
  logger.info(`Test server started on port ${port}`);
});

// Handle cleanup
process.on('SIGTERM', () => {
  logger.info('Shutting down test server');
  server.close(() => {
    process.exit(0);
  });
});

module.exports = server; 