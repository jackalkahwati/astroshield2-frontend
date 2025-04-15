/**
 * Unit tests for the ML feature extractors
 */
const assert = require('assert');
const { extractManeuverFeatures, extractObservationFeatures } = require('../ml/featureExtractors');

describe('Feature Extractors', () => {
  describe('extractManeuverFeatures', () => {
    it('should extract features from maneuver events correctly', () => {
      // Create a sample maneuver event
      const maneuverEvent = {
        catalogId: 'SATCAT-12345',
        deltaV: 0.25,
        confidence: 0.92,
        maneuverType: 'MAJOR_MANEUVER',
        detectionTime: '2023-09-01T11:45:00Z',
        source: 'astroshield-maneuver-detector',
        messageId: 'msg-123'
      };

      // Extract features
      const features = extractManeuverFeatures(maneuverEvent);
      
      // Assert basic features
      assert.strictEqual(features.deltaV, 0.25);
      assert.strictEqual(features.confidence, 0.92);
      
      // Assert categorical encoding
      assert.strictEqual(features.isStationkeeping, 0);
      assert.strictEqual(features.isOrbitMaintenance, 0);
      assert.strictEqual(features.isOrbitAdjustment, 0);
      assert.strictEqual(features.isMajorManeuver, 1);
      
      // Assert time-based features are numbers
      assert.strictEqual(typeof features.hourOfDay, 'number');
      assert.strictEqual(typeof features.dayOfWeek, 'number');
    });
  });

  describe('extractObservationFeatures', () => {
    it('should extract features from observation window events correctly', () => {
      // Create a sample observation window event
      const observationEvent = {
        location: {
          latitude: 32.9,
          longitude: -117.2,
          locationName: 'San Diego Observatory'
        },
        qualityScore: 0.85,
        qualityCategory: 'GOOD',
        recommendation: 'GO',
        observationWindow: {
          startTime: '2023-09-02T22:00:00Z',
          endTime: '2023-09-03T01:00:00Z',
          durationMinutes: 180
        },
        targetObject: {
          catalogId: 'SATCAT-54321',
          altitudeKm: 550
        }
      };

      // Extract features
      const features = extractObservationFeatures(observationEvent);
      
      // Assert basic features
      assert.strictEqual(features.qualityScore, 0.85);
      assert.strictEqual(features.latitude, 32.9);
      assert.strictEqual(features.longitude, -117.2);
      assert.strictEqual(features.durationMinutes, 180);
      assert.strictEqual(features.altitudeKm, 550);
      
      // Assert categorical encoding
      assert.strictEqual(features.isExcellent, 0);
      assert.strictEqual(features.isGood, 1);
      assert.strictEqual(features.isFair, 0);
      assert.strictEqual(features.isPoor, 0);
      
      // Assert time-based features are numbers
      assert.strictEqual(typeof features.startHourUTC, 'number');
    });
  });
}); 