/**
 * Unit tests for the validation module
 */
const assert = require('assert');
const testConfig = require('../test_config');
const { 
  validateMessage, 
  validateManeuverMessage,
  validateObjectDetailsMessage,
  validateObservationMessage
} = require('../../src/validation');

describe('Validation Module', () => {
  describe('validateManeuverMessage', () => {
    it('should validate a valid maneuver message', () => {
      // Create a valid maneuver message
      const message = testConfig.helpers.createTestMessage('maneuver');
      
      // Validate
      const result = validateManeuverMessage(message);
      
      // Should return true for valid messages
      assert.strictEqual(result, true);
    });
    
    it('should throw error for invalid maneuver message structure', () => {
      // Create invalid messages
      const noMetadata = { data: { spacecraftId: 'test', maneuverType: 'test' } };
      const noData = { metadata: { id: 'test', timestamp: new Date().toISOString() } };
      const emptyMessage = {};
      const nullMessage = null;
      
      // Validate each invalid case
      assert.throws(() => validateManeuverMessage(noMetadata), /Invalid message format/);
      assert.throws(() => validateManeuverMessage(noData), /Invalid message format/);
      assert.throws(() => validateManeuverMessage(emptyMessage), /Invalid message format/);
      assert.throws(() => validateManeuverMessage(nullMessage), /Invalid message format/);
    });
    
    it('should throw error for missing required fields', () => {
      // Valid base message
      const baseMessage = testConfig.helpers.createTestMessage('maneuver');
      
      // Create messages with missing required fields
      const noSpacecraftId = JSON.parse(JSON.stringify(baseMessage));
      delete noSpacecraftId.data.spacecraftId;
      
      const noManeuverType = JSON.parse(JSON.stringify(baseMessage));
      delete noManeuverType.data.maneuverType;
      
      // Validate each invalid case
      assert.throws(() => validateManeuverMessage(noSpacecraftId), /spacecraftId is required/);
      assert.throws(() => validateManeuverMessage(noManeuverType), /maneuverType is required/);
    });
  });
  
  describe('validateObjectDetailsMessage', () => {
    it('should validate a valid object details message', () => {
      // Create a valid object details message
      const message = testConfig.helpers.createTestMessage('objectDetails');
      
      // Validate
      const result = validateObjectDetailsMessage(message);
      
      // Should return true for valid messages
      assert.strictEqual(result, true);
    });
    
    it('should throw error for invalid object details message structure', () => {
      // Create invalid messages
      const noMetadata = { data: { objectId: 'test', objectType: 'test' } };
      const noData = { metadata: { id: 'test', timestamp: new Date().toISOString() } };
      const emptyMessage = {};
      const nullMessage = null;
      
      // Validate each invalid case
      assert.throws(() => validateObjectDetailsMessage(noMetadata), /Invalid message format/);
      assert.throws(() => validateObjectDetailsMessage(noData), /Invalid message format/);
      assert.throws(() => validateObjectDetailsMessage(emptyMessage), /Invalid message format/);
      assert.throws(() => validateObjectDetailsMessage(nullMessage), /Invalid message format/);
    });
    
    it('should throw error for missing required fields', () => {
      // Valid base message
      const baseMessage = testConfig.helpers.createTestMessage('objectDetails');
      
      // Create messages with missing required fields
      const noObjectId = JSON.parse(JSON.stringify(baseMessage));
      delete noObjectId.data.objectId;
      
      const noObjectType = JSON.parse(JSON.stringify(baseMessage));
      delete noObjectType.data.objectType;
      
      // Validate each invalid case
      assert.throws(() => validateObjectDetailsMessage(noObjectId), /objectId is required/);
      assert.throws(() => validateObjectDetailsMessage(noObjectType), /objectType is required/);
    });
  });
  
  describe('validateObservationMessage', () => {
    it('should validate a valid observation message', () => {
      // Create a valid observation message
      const message = testConfig.helpers.createTestMessage('observation');
      
      // Validate
      const result = validateObservationMessage(message);
      
      // Should return true for valid messages
      assert.strictEqual(result, true);
    });
    
    it('should throw error for invalid observation message structure', () => {
      // Create invalid messages
      const noMetadata = { data: { sensorId: 'test', objectId: 'test' } };
      const noData = { metadata: { id: 'test', timestamp: new Date().toISOString() } };
      const emptyMessage = {};
      const nullMessage = null;
      
      // Validate each invalid case
      assert.throws(() => validateObservationMessage(noMetadata), /Invalid message format/);
      assert.throws(() => validateObservationMessage(noData), /Invalid message format/);
      assert.throws(() => validateObservationMessage(emptyMessage), /Invalid message format/);
      assert.throws(() => validateObservationMessage(nullMessage), /Invalid message format/);
    });
    
    it('should throw error for missing required fields', () => {
      // Valid base message
      const baseMessage = testConfig.helpers.createTestMessage('observation');
      
      // Create messages with missing required fields
      const noSensorId = JSON.parse(JSON.stringify(baseMessage));
      delete noSensorId.data.sensorId;
      
      const noObjectId = JSON.parse(JSON.stringify(baseMessage));
      delete noObjectId.data.objectId;
      
      const noMeasurementType = JSON.parse(JSON.stringify(baseMessage));
      delete noMeasurementType.data.measurementType;
      
      // Validate each invalid case
      assert.throws(() => validateObservationMessage(noSensorId), /sensorId is required/);
      assert.throws(() => validateObservationMessage(noObjectId), /objectId is required/);
      assert.throws(() => validateObservationMessage(noMeasurementType), /measurementType is required/);
    });
  });
  
  describe('validateMessage', () => {
    it('should validate based on message type', () => {
      // Create valid messages of each type
      const maneuverMessage = testConfig.helpers.createTestMessage('maneuver');
      const objectDetailsMessage = testConfig.helpers.createTestMessage('objectDetails');
      const observationMessage = testConfig.helpers.createTestMessage('observation');
      
      // Validate each type
      assert.strictEqual(validateMessage(maneuverMessage, 'maneuver'), true);
      assert.strictEqual(validateMessage(objectDetailsMessage, 'objectDetails'), true);
      assert.strictEqual(validateMessage(observationMessage, 'observation'), true);
    });
    
    it('should throw error for unsupported message type', () => {
      const message = testConfig.helpers.createTestMessage('maneuver');
      
      assert.throws(() => validateMessage(message, 'unknownType'), /Unsupported message type/);
    });
    
    it('should throw error for empty or null message', () => {
      assert.throws(() => validateMessage(null, 'maneuver'), /Invalid message format/);
      assert.throws(() => validateMessage({}, 'maneuver'), /Invalid message format/);
    });
  });
}); 