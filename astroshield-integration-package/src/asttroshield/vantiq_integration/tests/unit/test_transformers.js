/**
 * Unit tests for the Vantiq transformers module
 */
const assert = require('assert');
const testConfig = require('../test_config');
const { 
  transformManeuverToVantiq,
  transformObjectDetailsToVantiq,
  transformObservationToVantiq,
  transformAstroShieldToVantiq,
  transformVantiqToAstroShield
} = require('../../src/transformers');

describe('Transformers Module', () => {
  // Test for basic message validation
  describe('Input Validation', () => {
    it('should throw error for null/undefined maneuver message', () => {
      assert.throws(() => transformManeuverToVantiq(null), /Invalid maneuver message format/);
      assert.throws(() => transformManeuverToVantiq({}), /Invalid maneuver message format/);
    });

    it('should throw error for null/undefined object details message', () => {
      assert.throws(() => transformObjectDetailsToVantiq(null), /Invalid object details message format/);
      assert.throws(() => transformObjectDetailsToVantiq({}), /Invalid object details message format/);
    });

    it('should throw error for null/undefined observation message', () => {
      assert.throws(() => transformObservationToVantiq(null), /Invalid observation message format/);
      assert.throws(() => transformObservationToVantiq({}), /Invalid observation message format/);
    });
  });

  // Test maneuver transformation
  describe('transformManeuverToVantiq', () => {
    it('should transform maneuver messages correctly', () => {
      // Use the test helper to create a test message
      const inputMessage = testConfig.helpers.createTestMessage('maneuver');
      
      // Transform message
      const result = transformManeuverToVantiq(inputMessage);
      
      // Assert
      assert.strictEqual(result.id, inputMessage.metadata.id);
      assert.strictEqual(result.timestamp, inputMessage.metadata.timestamp);
      assert.strictEqual(result.spacecraft_id, inputMessage.data.spacecraftId);
      assert.strictEqual(result.maneuver_type, inputMessage.data.maneuverType);
      assert.deepStrictEqual(result.parameters, inputMessage.data.parameters);
      assert.strictEqual(result.priority, inputMessage.data.priority);
      assert.deepStrictEqual(result.execution_window, {
        start: inputMessage.data.executionWindow.start,
        end: inputMessage.data.executionWindow.end
      });
      assert.strictEqual(result.status, inputMessage.data.status);
      assert.deepStrictEqual(result.origin, {
        system: 'AstroShield',
        subsystem: inputMessage.metadata.source,
        version: inputMessage.metadata.version
      });
    });

    it('should handle missing optional fields in maneuver messages', () => {
      // Create a minimal test message
      const inputMessage = testConfig.helpers.createTestMessage('maneuver', {
        data: {
          // Remove optional fields
          parameters: undefined,
          priority: undefined,
          executionWindow: undefined,
          status: undefined
        }
      });

      // Transform message
      const result = transformManeuverToVantiq(inputMessage);
      
      // Assert required fields
      assert.strictEqual(result.id, inputMessage.metadata.id);
      assert.strictEqual(result.timestamp, inputMessage.metadata.timestamp);
      assert.strictEqual(result.spacecraft_id, inputMessage.data.spacecraftId);
      assert.strictEqual(result.maneuver_type, inputMessage.data.maneuverType);
      
      // Assert defaults for optional fields
      assert.deepStrictEqual(result.parameters, {});
      assert.strictEqual(result.priority, 'normal');
      assert.strictEqual(result.status, 'pending');
    });
  });

  // Test object details transformation
  describe('transformObjectDetailsToVantiq', () => {
    it('should transform object details messages correctly', () => {
      // Use the test helper to create a test message
      const inputMessage = testConfig.helpers.createTestMessage('objectDetails');

      // Transform message
      const result = transformObjectDetailsToVantiq(inputMessage);
      
      // Assert
      assert.strictEqual(result.id, inputMessage.metadata.id);
      assert.strictEqual(result.timestamp, inputMessage.metadata.timestamp);
      assert.strictEqual(result.object_id, inputMessage.data.objectId);
      assert.strictEqual(result.object_type, inputMessage.data.objectType);
      assert.strictEqual(result.classification, inputMessage.data.classification);
      assert.deepStrictEqual(result.observed_parameters, inputMessage.data.parameters);
      assert.deepStrictEqual(result.state_vector, inputMessage.data.stateVector);
      assert.strictEqual(result.confidence, inputMessage.data.confidence);
      assert.deepStrictEqual(result.additional_properties, inputMessage.data.additionalProperties);
      assert.deepStrictEqual(result.origin, {
        system: 'AstroShield',
        subsystem: inputMessage.metadata.source,
        version: inputMessage.metadata.version
      });
    });
  });

  // Test observation transformation
  describe('transformObservationToVantiq', () => {
    it('should transform observation messages correctly', () => {
      // Use the test helper to create a test message
      const inputMessage = testConfig.helpers.createTestMessage('observation');

      // Transform message
      const result = transformObservationToVantiq(inputMessage);
      
      // Assert
      assert.strictEqual(result.id, inputMessage.metadata.id);
      assert.strictEqual(result.timestamp, inputMessage.metadata.timestamp);
      assert.strictEqual(result.sensor_id, inputMessage.data.sensorId);
      assert.strictEqual(result.object_id, inputMessage.data.objectId);
      assert.strictEqual(result.measurement_type, inputMessage.data.measurementType);
      assert.deepStrictEqual(result.measurement_values, inputMessage.data.values);
      assert.strictEqual(result.quality, inputMessage.data.quality);
      assert.strictEqual(result.observation_time, inputMessage.data.observationTime);
      assert.deepStrictEqual(result.origin, {
        system: 'AstroShield',
        subsystem: inputMessage.metadata.source,
        version: inputMessage.metadata.version
      });
    });
  });

  // Test the higher-level transformation functions
  describe('transformAstroShieldToVantiq', () => {
    it('should transform messages by message type', () => {
      // This is a partial test since the function has nested dependencies
      // Would need proper mocking to fully test
      assert.throws(() => transformAstroShieldToVantiq(null, 'maneuver'), /Message is required/);
      assert.throws(() => transformAstroShieldToVantiq({}, null), /Message type is required/);
    });
  });

  describe('transformVantiqToAstroShield', () => {
    it('should transform Vantiq messages by message type', () => {
      // This is a partial test since the function has nested dependencies
      // Would need proper mocking to fully test
      assert.throws(() => transformVantiqToAstroShield(null, 'maneuver'), /Message is required/);
      assert.throws(() => transformVantiqToAstroShield({}, null), /Message type is required/);
    });
  });
}); 