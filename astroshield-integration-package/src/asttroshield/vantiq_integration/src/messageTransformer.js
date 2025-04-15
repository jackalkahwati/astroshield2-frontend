/**
 * Message transformer functions to convert AstroShield message formats to Vantiq formats
 */

/**
 * Transforms a maneuver detection message to Vantiq format
 * @param {Object} message - AstroShield maneuver detection message
 * @returns {Object} - Transformed message for Vantiq
 */
function transformManeuverDetection(message) {
  const { header, payload } = message;
  
  // Create metadata from header
  const metadata = {
    messageId: header.messageId,
    sourceTimestamp: header.timestamp,
    source: header.source
  };
  
  // Add optional traceId if present
  if (header.traceId) {
    metadata.traceId = header.traceId;
  }
  
  // Create transformed message with payload and metadata
  return {
    ...payload,
    metadata
  };
}

/**
 * Transforms an object details message to Vantiq format
 * @param {Object} message - AstroShield object details message
 * @returns {Object} - Transformed message for Vantiq
 */
function transformObjectDetails(message) {
  const { header, payload } = message;
  
  // Create metadata from header
  const metadata = {
    messageId: header.messageId,
    sourceTimestamp: header.timestamp,
    source: header.source
  };
  
  // Add optional traceId if present
  if (header.traceId) {
    metadata.traceId = header.traceId;
  }
  
  // Create transformed message with payload and metadata
  return {
    ...payload,
    metadata
  };
}

/**
 * Transforms an observation window message to Vantiq format
 * @param {Object} message - AstroShield observation window message
 * @returns {Object} - Transformed message for Vantiq
 */
function transformObservationWindow(message) {
  const { header, payload } = message;
  
  // Create metadata from header
  const metadata = {
    messageId: header.messageId,
    sourceTimestamp: header.timestamp,
    source: header.source
  };
  
  // Add optional traceId if present
  if (header.traceId) {
    metadata.traceId = header.traceId;
  }
  
  // Create transformed message with payload and metadata
  return {
    ...payload,
    metadata
  };
}

/**
 * Transforms a message based on its type
 * @param {Object} message - AstroShield message
 * @returns {Object} - Transformed message for Vantiq
 * @throws {Error} - If message type is not supported
 */
function transformMessage(message) {
  const { header } = message;
  
  switch (header.messageType) {
    case 'maneuver-detected':
      return transformManeuverDetection(message);
    case 'object-details':
      return transformObjectDetails(message);
    case 'observation-window':
      return transformObservationWindow(message);
    default:
      throw new Error(`Unsupported message type: ${header.messageType}`);
  }
}

module.exports = {
  transformManeuverDetection,
  transformObjectDetails,
  transformObservationWindow,
  transformMessage
}; 