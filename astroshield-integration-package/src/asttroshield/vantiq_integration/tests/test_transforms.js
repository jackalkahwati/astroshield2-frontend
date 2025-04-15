/**
 * Unit tests for the Vantiq message transformers
 */
const assert = require('assert');
const { transformManeuverDetection, transformObservationWindow } = require('../src/messageTransformer');

describe('Message Transformers', () => {
  describe('transformManeuverDetection', () => {
    it('should transform maneuver detection messages correctly', () => {
      // Sample input message
      const inputMessage = {
        header: {
          messageId: 'msg-123',
          timestamp: '2023-09-01T12:00:00Z',
          source: 'astroshield-maneuver-detector',
          messageType: 'maneuver-detected',
          traceId: 'trace-456'
        },
        payload: {
          catalogId: 'SATCAT-12345',
          deltaV: 0.25,
          confidence: 0.92,
          maneuverType: 'MAJOR_MANEUVER',
          detectionTime: '2023-09-01T11:45:00Z'
        }
      };

      // Expected output
      const expectedOutput = {
        catalogId: 'SATCAT-12345',
        deltaV: 0.25,
        confidence: 0.92,
        maneuverType: 'MAJOR_MANEUVER',
        detectionTime: '2023-09-01T11:45:00Z',
        source: 'astroshield-maneuver-detector',
        messageId: 'msg-123',
        traceId: 'trace-456'
      };

      // Transform message
      const result = transformManeuverDetection(inputMessage);
      
      // Assert
      assert.deepStrictEqual(result, expectedOutput);
    });

    it('should handle missing fields gracefully', () => {
      // Sample input with missing fields
      const inputMessage = {
        header: {
          messageId: 'msg-123',
          source: 'astroshield-maneuver-detector'
        },
        payload: {
          catalogId: 'SATCAT-12345',
          deltaV: 0.25
        }
      };

      // Transform message
      const result = transformManeuverDetection(inputMessage);
      
      // Assert fields were handled gracefully
      assert.strictEqual(result.catalogId, 'SATCAT-12345');
      assert.strictEqual(result.deltaV, 0.25);
      assert.strictEqual(result.messageId, 'msg-123');
      assert.strictEqual(result.source, 'astroshield-maneuver-detector');
      assert.strictEqual(result.confidence, undefined);
      assert.strictEqual(result.maneuverType, undefined);
    });
  });

  describe('transformObservationWindow', () => {
    it('should transform observation window messages correctly', () => {
      // Sample input message
      const inputMessage = {
        header: {
          messageId: 'msg-789',
          timestamp: '2023-09-02T10:00:00Z',
          source: 'astroshield-observation-planner',
          messageType: 'observation-window-recommended'
        },
        payload: {
          location: {
            latitude: 32.9,
            longitude: -117.2,
            locationName: 'San Diego Observatory'
          },
          qualityScore: 0.85,
          qualityCategory: 'GOOD',
          recommendation: 'GO',
          observationWindow: {
            start_time: '2023-09-02T22:00:00Z',
            end_time: '2023-09-03T01:00:00Z',
            duration_minutes: 180
          },
          targetObject: {
            catalog_id: 'SATCAT-54321',
            altitude_km: 550
          }
        }
      };

      // Expected output
      const expectedOutput = {
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

      // Transform message
      const result = transformObservationWindow(inputMessage);
      
      // Assert
      assert.deepStrictEqual(result, expectedOutput);
    });
  });
}); 