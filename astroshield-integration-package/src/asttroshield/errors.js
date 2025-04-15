/**
 * Custom errors for the AstroShield system
 */

/**
 * Base error class for AstroShield errors
 */
class AstroShieldError extends Error {
  /**
   * Create a new AstroShield error
   * @param {string} message - Error message
   * @param {Error} [cause] - Optional cause of the error
   */
  constructor(message, cause) {
    super(message);
    this.name = this.constructor.name;
    this.cause = cause;
    
    // Capture stack trace
    Error.captureStackTrace(this, this.constructor);
  }
}

/**
 * Error for validation failures
 */
class ValidationError extends AstroShieldError {
  /**
   * Create a new validation error
   * @param {string} message - Error message
   * @param {Error} [cause] - Optional cause of the error
   */
  constructor(message, cause) {
    super(message, cause);
  }
}

/**
 * Error for configuration issues
 */
class ConfigError extends AstroShieldError {
  /**
   * Create a new configuration error
   * @param {string} message - Error message
   * @param {Error} [cause] - Optional cause of the error
   */
  constructor(message, cause) {
    super(message, cause);
  }
}

/**
 * Error for connection issues
 */
class ConnectionError extends AstroShieldError {
  /**
   * Create a new connection error
   * @param {string} message - Error message
   * @param {Error} [cause] - Optional cause of the error
   */
  constructor(message, cause) {
    super(message, cause);
  }
}

/**
 * Error for timeout issues
 */
class TimeoutError extends AstroShieldError {
  /**
   * Create a new timeout error
   * @param {string} message - Error message
   * @param {Error} [cause] - Optional cause of the error
   */
  constructor(message, cause) {
    super(message, cause);
  }
}

/**
 * Error for authentication/authorization issues
 */
class AuthError extends AstroShieldError {
  /**
   * Create a new authentication error
   * @param {string} message - Error message
   * @param {Error} [cause] - Optional cause of the error
   */
  constructor(message, cause) {
    super(message, cause);
  }
}

/**
 * Error for service unavailability
 */
class ServiceError extends AstroShieldError {
  /**
   * Create a new service error
   * @param {string} message - Error message
   * @param {Error} [cause] - Optional cause of the error
   */
  constructor(message, cause) {
    super(message, cause);
  }
}

module.exports = {
  AstroShieldError,
  ValidationError,
  ConfigError,
  ConnectionError,
  TimeoutError,
  AuthError,
  ServiceError
}; 