/**
 * Validation module for AstroShield-Vantiq integration
 * Validates message formats before processing
 */
const { ValidationError } = require('../../errors') || class ValidationError extends Error {};

/**
 * Validate basic message structure
 * @param {Object} message - Message to validate
 * @throws {ValidationError} - If message is invalid
 * @private
 */
function validateBasicStructure(message) {
  if (!message) {
    throw new ValidationError('Invalid message format: message is null or undefined');
  }

  if (typeof message !== 'object') {
    throw new ValidationError('Invalid message format: message must be an object');
  }

  if (!message.metadata || typeof message.metadata !== 'object') {
    throw new ValidationError('Invalid message format: metadata is required');
  }

  if (!message.data || typeof message.data !== 'object') {
    throw new ValidationError('Invalid message format: data is required');
  }
}

/**
 * Validate maneuver message
 * @param {Object} message - Message to validate
 * @returns {boolean} - True if valid
 * @throws {ValidationError} - If message is invalid
 */
function validateManeuverMessage(message) {
  validateBasicStructure(message);

  const { data } = message;

  if (!data.spacecraftId) {
    throw new ValidationError('Invalid maneuver message: spacecraftId is required');
  }

  if (!data.maneuverType) {
    throw new ValidationError('Invalid maneuver message: maneuverType is required');
  }

  // All checks passed
  return true;
}

/**
 * Validate object details message
 * @param {Object} message - Message to validate
 * @returns {boolean} - True if valid
 * @throws {ValidationError} - If message is invalid
 */
function validateObjectDetailsMessage(message) {
  validateBasicStructure(message);

  const { data } = message;

  if (!data.objectId) {
    throw new ValidationError('Invalid object details message: objectId is required');
  }

  if (!data.objectType) {
    throw new ValidationError('Invalid object details message: objectType is required');
  }

  // All checks passed
  return true;
}

/**
 * Validate observation message
 * @param {Object} message - Message to validate
 * @returns {boolean} - True if valid
 * @throws {ValidationError} - If message is invalid
 */
function validateObservationMessage(message) {
  validateBasicStructure(message);

  const { data } = message;

  if (!data.sensorId) {
    throw new ValidationError('Invalid observation message: sensorId is required');
  }

  if (!data.objectId) {
    throw new ValidationError('Invalid observation message: objectId is required');
  }

  if (!data.measurementType) {
    throw new ValidationError('Invalid observation message: measurementType is required');
  }

  // All checks passed
  return true;
}

/**
 * Validate Vantiq formatted message
 * @param {Object} message - Message to validate
 * @returns {boolean} - True if valid
 * @throws {ValidationError} - If message is invalid
 */
function validateVantiqMessage(message) {
  if (!message) {
    throw new ValidationError('Invalid Vantiq message format: message is null or undefined');
  }

  if (typeof message !== 'object') {
    throw new ValidationError('Invalid Vantiq message format: message must be an object');
  }

  // Minimal validation - Vantiq message formats may vary
  if (!message.id) {
    throw new ValidationError('Invalid Vantiq message format: id is required');
  }

  // All checks passed
  return true;
}

/**
 * Validate message based on type
 * @param {Object} message - Message to validate
 * @param {string} messageType - Type of message (maneuver, objectDetails, observation)
 * @returns {boolean} - True if valid
 * @throws {ValidationError} - If message is invalid
 */
function validateMessage(message, messageType) {
  if (!message) {
    throw new ValidationError('Invalid message format: message is null or undefined');
  }

  if (!messageType) {
    throw new ValidationError('Message type is required for validation');
  }

  switch (messageType) {
    case 'maneuver':
      return validateManeuverMessage(message);
    case 'objectDetails':
      return validateObjectDetailsMessage(message);
    case 'observation':
      return validateObservationMessage(message);
    case 'vantiq':
      return validateVantiqMessage(message);
    default:
      throw new ValidationError(`Unsupported message type: ${messageType}`);
  }
}

module.exports = {
  validateMessage,
  validateManeuverMessage,
  validateObjectDetailsMessage,
  validateObservationMessage,
  validateVantiqMessage
}; 