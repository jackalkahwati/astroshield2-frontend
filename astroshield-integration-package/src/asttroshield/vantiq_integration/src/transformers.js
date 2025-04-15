/**
 * Message transformation functions for Vantiq integration
 */
const { ValidationError } = require('../../errors');

/**
 * Transform AstroShield maneuver message to Vantiq format
 * @param {Object} message - The original AstroShield maneuver message
 * @returns {Object} Transformed message in Vantiq format
 */
function transformManeuverToVantiq(message) {
  if (!message || !message.data || !message.metadata) {
    throw new ValidationError('Invalid maneuver message format');
  }

  const { data, metadata } = message;
  
  return {
    id: metadata.id || generateId(),
    timestamp: metadata.timestamp || new Date().toISOString(),
    spacecraft_id: data.spacecraftId,
    maneuver_type: data.maneuverType,
    parameters: data.parameters || {},
    origin: {
      system: 'AstroShield',
      subsystem: metadata.source || 'unknown',
      version: metadata.version || '1.0.0'
    },
    priority: data.priority || 'normal',
    execution_window: {
      start: data.executionWindow?.start,
      end: data.executionWindow?.end
    },
    status: data.status || 'pending'
  };
}

/**
 * Transform AstroShield object details message to Vantiq format
 * @param {Object} message - The original AstroShield object details message
 * @returns {Object} Transformed message in Vantiq format
 */
function transformObjectDetailsToVantiq(message) {
  if (!message || !message.data || !message.metadata) {
    throw new ValidationError('Invalid object details message format');
  }

  const { data, metadata } = message;
  
  return {
    id: metadata.id || generateId(),
    timestamp: metadata.timestamp || new Date().toISOString(),
    object_id: data.objectId,
    object_type: data.objectType,
    classification: data.classification,
    observed_parameters: data.parameters || {},
    state_vector: data.stateVector || {},
    origin: {
      system: 'AstroShield',
      subsystem: metadata.source || 'unknown',
      version: metadata.version || '1.0.0'
    },
    confidence: data.confidence || 1.0,
    additional_properties: data.additionalProperties || {}
  };
}

/**
 * Transform AstroShield observation message to Vantiq format
 * @param {Object} message - The original AstroShield observation message
 * @returns {Object} Transformed message in Vantiq format
 */
function transformObservationToVantiq(message) {
  if (!message || !message.data || !message.metadata) {
    throw new ValidationError('Invalid observation message format');
  }

  const { data, metadata } = message;
  
  return {
    id: metadata.id || generateId(),
    timestamp: metadata.timestamp || new Date().toISOString(),
    sensor_id: data.sensorId,
    object_id: data.objectId,
    measurement_type: data.measurementType,
    measurement_values: data.values || {},
    origin: {
      system: 'AstroShield',
      subsystem: metadata.source || 'unknown',
      version: metadata.version || '1.0.0'
    },
    quality: data.quality || 1.0,
    observation_time: data.observationTime || metadata.timestamp || new Date().toISOString()
  };
}

/**
 * Transform Vantiq message to AstroShield format (generic transformation)
 * 
 * Note: This would need to be expanded for specific message types from Vantiq
 * when bidirectional communication is implemented
 * 
 * @param {Object} message - The original Vantiq message
 * @param {string} type - The message type
 * @returns {Object} Transformed message in AstroShield format
 */
function transformVantiqToAstroShield(message, type) {
  if (!message) {
    throw new ValidationError('Invalid message format');
  }
  
  // This is a placeholder for future bidirectional integration
  // Would need to be expanded based on specific Vantiq message formats
  
  return {
    metadata: {
      id: message.id || generateId(),
      timestamp: message.timestamp || new Date().toISOString(),
      source: 'vantiq_integration',
      type: type,
      version: '1.0.0'
    },
    data: {
      ...message,
      // Transform specific fields here based on message type
    }
  };
}

/**
 * Generate a unique ID
 * @returns {string} - Unique ID
 * @private
 */
function generateId() {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
}

/**
 * Transformers module for converting between AstroShield and Vantiq message formats
 */

/**
 * Transform an AstroShield message to Vantiq format
 * @param {Object} message - AstroShield message
 * @param {string} messageType - Type of message (e.g., 'maneuver', 'object', 'observation')
 * @returns {Object} - Vantiq formatted message
 */
function transformAstroShieldToVantiq(message, messageType) {
  if (!message) {
    throw new Error('Message is required');
  }
  
  if (!messageType) {
    throw new Error('Message type is required');
  }

  // Common metadata for all Vantiq messages
  const vantiqMessage = {
    source: 'astroshield',
    timestamp: message.timestamp || new Date().toISOString(),
    metadata: {
      messageType,
      version: '1.0',
      sourceSystem: 'AstroShield'
    }
  };

  // Transform based on message type
  switch (messageType.toLowerCase()) {
    case 'maneuver':
      return transformManeuver(message, vantiqMessage);
    case 'object':
      return transformObject(message, vantiqMessage);
    case 'observation':
      return transformObservation(message, vantiqMessage);
    case 'alert':
      return transformAlert(message, vantiqMessage);
    case 'status':
      return transformStatus(message, vantiqMessage);
    default:
      // Generic transformation
      return {
        ...vantiqMessage,
        data: message
      };
  }
}

/**
 * Transform a Vantiq message to AstroShield format
 * @param {Object} message - Vantiq message
 * @param {string} messageType - Type of message
 * @param {string} topic - Kafka topic
 * @returns {Object} - AstroShield formatted message
 */
function transformVantiqToAstroShield(message, messageType, topic) {
  if (!message) {
    throw new Error('Message is required');
  }
  
  if (!messageType) {
    throw new Error('Message type is required');
  }

  // Common metadata for all AstroShield messages
  const astroShieldMessage = {
    id: message.id || generateId(),
    timestamp: message.timestamp || new Date().toISOString(),
    source: 'vantiq',
    type: messageType
  };

  // Transform based on message type
  switch (messageType.toLowerCase()) {
    case 'maneuver':
      return transformVantiqManeuver(message, astroShieldMessage);
    case 'object':
      return transformVantiqObject(message, astroShieldMessage);
    case 'observation':
      return transformVantiqObservation(message, astroShieldMessage);
    case 'command':
      return transformVantiqCommand(message, astroShieldMessage);
    case 'configuration':
      return transformVantiqConfiguration(message, astroShieldMessage);
    default:
      // Generic transformation
      return {
        ...astroShieldMessage,
        data: message.data || message
      };
  }
}

/**
 * Transform AstroShield maneuver message to Vantiq format
 * @param {Object} message - AstroShield maneuver message
 * @param {Object} baseMessage - Base Vantiq message
 * @returns {Object} - Vantiq formatted maneuver message
 * @private
 */
function transformManeuver(message, baseMessage) {
  return {
    ...baseMessage,
    data: {
      maneuverType: message.maneuverType,
      satelliteId: message.satelliteId,
      startTime: message.startTime,
      endTime: message.endTime,
      deltaV: message.deltaV,
      initialState: message.initialState,
      finalState: message.finalState,
      confidence: message.confidence,
      purpose: message.purpose,
      status: message.status
    }
  };
}

/**
 * Transform AstroShield object message to Vantiq format
 * @param {Object} message - AstroShield object message
 * @param {Object} baseMessage - Base Vantiq message
 * @returns {Object} - Vantiq formatted object message
 * @private
 */
function transformObject(message, baseMessage) {
  return {
    ...baseMessage,
    data: {
      objectId: message.objectId,
      name: message.name,
      type: message.type,
      state: {
        position: message.state?.position,
        velocity: message.state?.velocity,
        epoch: message.state?.epoch
      },
      orbit: message.orbit,
      physicalProperties: message.physicalProperties,
      metadata: message.metadata
    }
  };
}

/**
 * Transform AstroShield observation message to Vantiq format
 * @param {Object} message - AstroShield observation message
 * @param {Object} baseMessage - Base Vantiq message
 * @returns {Object} - Vantiq formatted observation message
 * @private
 */
function transformObservation(message, baseMessage) {
  return {
    ...baseMessage,
    data: {
      observationId: message.observationId,
      objectId: message.objectId,
      sensorId: message.sensorId,
      measurementType: message.measurementType,
      measurementValue: message.measurementValue,
      uncertainty: message.uncertainty,
      time: message.time,
      metadata: message.metadata
    }
  };
}

/**
 * Transform AstroShield alert message to Vantiq format
 * @param {Object} message - AstroShield alert message
 * @param {Object} baseMessage - Base Vantiq message
 * @returns {Object} - Vantiq formatted alert message
 * @private
 */
function transformAlert(message, baseMessage) {
  return {
    ...baseMessage,
    data: {
      alertId: message.alertId,
      alertType: message.alertType,
      severity: message.severity,
      title: message.title,
      description: message.description,
      objectIds: message.objectIds,
      timestamp: message.timestamp,
      metadata: message.metadata
    }
  };
}

/**
 * Transform AstroShield status message to Vantiq format
 * @param {Object} message - AstroShield status message
 * @param {Object} baseMessage - Base Vantiq message
 * @returns {Object} - Vantiq formatted status message
 * @private
 */
function transformStatus(message, baseMessage) {
  return {
    ...baseMessage,
    data: {
      statusId: message.statusId,
      component: message.component,
      status: message.status,
      details: message.details,
      timestamp: message.timestamp
    }
  };
}

/**
 * Transform Vantiq maneuver message to AstroShield format
 * @param {Object} message - Vantiq maneuver message
 * @param {Object} baseMessage - Base AstroShield message
 * @returns {Object} - AstroShield formatted maneuver message
 * @private
 */
function transformVantiqManeuver(message, baseMessage) {
  const data = message.data || {};
  
  return {
    ...baseMessage,
    maneuverType: data.maneuverType,
    satelliteId: data.satelliteId,
    startTime: data.startTime,
    endTime: data.endTime,
    deltaV: data.deltaV,
    initialState: data.initialState,
    finalState: data.finalState,
    confidence: data.confidence,
    purpose: data.purpose,
    status: data.status
  };
}

/**
 * Transform Vantiq object message to AstroShield format
 * @param {Object} message - Vantiq object message
 * @param {Object} baseMessage - Base AstroShield message
 * @returns {Object} - AstroShield formatted object message
 * @private
 */
function transformVantiqObject(message, baseMessage) {
  const data = message.data || {};
  
  return {
    ...baseMessage,
    objectId: data.objectId,
    name: data.name,
    type: data.type,
    state: {
      position: data.state?.position,
      velocity: data.state?.velocity,
      epoch: data.state?.epoch
    },
    orbit: data.orbit,
    physicalProperties: data.physicalProperties,
    metadata: data.metadata
  };
}

/**
 * Transform Vantiq observation message to AstroShield format
 * @param {Object} message - Vantiq observation message
 * @param {Object} baseMessage - Base AstroShield message
 * @returns {Object} - AstroShield formatted observation message
 * @private
 */
function transformVantiqObservation(message, baseMessage) {
  const data = message.data || {};
  
  return {
    ...baseMessage,
    observationId: data.observationId,
    objectId: data.objectId,
    sensorId: data.sensorId,
    measurementType: data.measurementType,
    measurementValue: data.measurementValue,
    uncertainty: data.uncertainty,
    time: data.time,
    metadata: data.metadata
  };
}

/**
 * Transform Vantiq command message to AstroShield format
 * @param {Object} message - Vantiq command message
 * @param {Object} baseMessage - Base AstroShield message
 * @returns {Object} - AstroShield formatted command message
 * @private
 */
function transformVantiqCommand(message, baseMessage) {
  const data = message.data || {};
  
  return {
    ...baseMessage,
    commandId: data.commandId,
    targetSystem: data.targetSystem,
    commandType: data.commandType,
    parameters: data.parameters,
    priority: data.priority,
    requesterId: data.requesterId
  };
}

/**
 * Transform Vantiq configuration message to AstroShield format
 * @param {Object} message - Vantiq configuration message
 * @param {Object} baseMessage - Base AstroShield message
 * @returns {Object} - AstroShield formatted configuration message
 * @private
 */
function transformVantiqConfiguration(message, baseMessage) {
  const data = message.data || {};
  
  return {
    ...baseMessage,
    configId: data.configId,
    component: data.component,
    parameters: data.parameters,
    version: data.version,
    effectiveFrom: data.effectiveFrom,
    effectiveTo: data.effectiveTo
  };
}

module.exports = {
  transformAstroShieldToVantiq,
  transformVantiqToAstroShield
}; 