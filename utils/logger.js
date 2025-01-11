const winston = require('winston');

const sensitiveFields = ['password', 'apiKey', 'token', 'secret', 'creditCard', 'ssn'];

function maskSensitiveData(info) {
  if (typeof info === 'object') {
    const masked = { ...info };
    for (const field of sensitiveFields) {
      if (masked[field]) {
        masked[field] = '[REDACTED]';
      }
      if (masked.metadata && masked.metadata[field]) {
        masked.metadata[field] = '[REDACTED]';
      }
    }
    return masked;
  }
  return info;
}

function createLogger(options = {}) {
  const logger = winston.createLogger({
    level: options.level || process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.errors({ stack: true }),
      winston.format(info => maskSensitiveData(info))(),
      winston.format.json()
    ),
    defaultMeta: { service: 'astroshield-service' },
    transports: [
      new winston.transports.Console({
        format: winston.format.combine(
          winston.format.colorize(),
          winston.format.simple()
        )
      })
    ]
  });

  // Add file transports in production
  if (process.env.NODE_ENV === 'production') {
    logger.add(new winston.transports.File({
      filename: 'logs/error.log',
      level: 'error',
      maxsize: 5242880, // 5MB
      maxFiles: 5
    }));

    logger.add(new winston.transports.File({
      filename: 'logs/combined.log',
      maxsize: 5242880,
      maxFiles: 5
    }));
  }

  // Override the logger methods to return the log entry
  const originalMethods = {
    error: logger.error,
    warn: logger.warn,
    info: logger.info,
    debug: logger.debug
  };

  Object.keys(originalMethods).forEach(level => {
    logger[level] = (...args) => {
      const logEntry = {
        level,
        message: args[0],
        metadata: args[1] || {},
        timestamp: new Date().toISOString()
      };
      originalMethods[level].apply(logger, args);
      return logEntry;
    };
  });

  return logger;
}

module.exports = {
  createLogger
}; 