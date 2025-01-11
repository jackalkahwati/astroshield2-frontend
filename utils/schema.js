const Ajv = require('ajv');
const addFormats = require('ajv-formats');

const ajv = new Ajv({
  allErrors: true,
  removeAdditional: true,
  useDefaults: true,
  coerceTypes: true,
  strict: false
});

addFormats(ajv);

// Common schema definitions
const definitions = {
  coordinates: {
    type: 'array',
    items: { type: 'number' },
    minItems: 3,
    maxItems: 3
  },
  timestamp: {
    type: 'string',
    format: 'date-time'
  }
};

// Schema for spacecraft commands
const commandSchema = {
  type: 'object',
  required: ['command_type', 'spacecraft_id', 'parameters'],
  properties: {
    command_type: { type: 'string', enum: ['maneuver', 'telemetry', 'configuration'] },
    spacecraft_id: { type: 'string' },
    parameters: {
      type: 'object',
      required: ['timestamp'],
      properties: {
        timestamp: { $ref: '#/definitions/timestamp' },
        delta_v: { type: 'number' },
        direction: { $ref: '#/definitions/coordinates' },
        duration: { type: 'number' }
      }
    }
  }
};

// Schema for telemetry data
const telemetrySchema = {
  type: 'object',
  required: ['spacecraft_id', 'timestamp', 'measurements'],
  properties: {
    spacecraft_id: { type: 'string' },
    timestamp: { $ref: '#/definitions/timestamp' },
    measurements: {
      type: 'object',
      required: ['position', 'velocity'],
      properties: {
        position: { $ref: '#/definitions/coordinates' },
        velocity: { $ref: '#/definitions/coordinates' },
        temperature: { type: 'number' },
        pressure: { type: 'number' }
      }
    }
  }
};

// Add schemas to validator
ajv.addSchema(commandSchema, 'command');
ajv.addSchema(telemetrySchema, 'telemetry');

function validateSchema(data, schemaName) {
  const validate = ajv.getSchema(schemaName);
  if (!validate) {
    throw new Error(`Schema ${schemaName} not found`);
  }

  const valid = validate(data);
  return {
    valid,
    errors: validate.errors
  };
}

module.exports = {
  validateSchema,
  definitions,
  commandSchema,
  telemetrySchema
}; 