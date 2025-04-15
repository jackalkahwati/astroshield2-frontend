/**
 * Unit tests for message transformer functions
 */
const { 
  transformManeuverDetection,
  transformObjectDetails,
  transformObservationWindow
} = require('../../src/messageTransformer');

describe('Message Transformer', () => {
  describe('transformManeuverDetection', () => {
    it('should transform maneuver detection message correctly', () => {
      // Arrange
      const input = {
        header: {
          messageId: 'msg-123',
          timestamp: '2023-05-15T10:30:00Z',
          source: 'astroshield-subsystem3',
          messageType: 'maneuver-detected',
          traceId: 'trace-456'
        },
        payload: {
          catalogId: 'SATCAT-12345',
          deltaV: 0.5,
          confidence: 0.95,
          maneuverType: 'ORBIT_ADJUSTMENT',
          detectionTime: '2023-05-15T10:25:00Z'
        }
      };

      // Act
      const result = transformManeuverDetection(input);

      // Assert
      expect(result).toEqual({
        catalogId: 'SATCAT-12345',
        deltaV: 0.5,
        confidence: 0.95, 
        maneuverType: 'ORBIT_ADJUSTMENT',
        detectionTime: '2023-05-15T10:25:00Z',
        metadata: {
          messageId: 'msg-123',
          sourceTimestamp: '2023-05-15T10:30:00Z',
          source: 'astroshield-subsystem3',
          traceId: 'trace-456'
        }
      });
    });

    it('should handle missing optional fields', () => {
      // Arrange
      const input = {
        header: {
          messageId: 'msg-123',
          timestamp: '2023-05-15T10:30:00Z',
          source: 'astroshield-subsystem3',
          messageType: 'maneuver-detected'
          // Missing traceId
        },
        payload: {
          catalogId: 'SATCAT-12345',
          deltaV: 0.5,
          // Missing confidence
          maneuverType: 'ORBIT_ADJUSTMENT',
          detectionTime: '2023-05-15T10:25:00Z'
        }
      };

      // Act
      const result = transformManeuverDetection(input);

      // Assert
      expect(result).toEqual({
        catalogId: 'SATCAT-12345',
        deltaV: 0.5,
        maneuverType: 'ORBIT_ADJUSTMENT',
        detectionTime: '2023-05-15T10:25:00Z',
        metadata: {
          messageId: 'msg-123',
          sourceTimestamp: '2023-05-15T10:30:00Z',
          source: 'astroshield-subsystem3'
        }
      });
    });
  });

  describe('transformObjectDetails', () => {
    it('should transform object details message correctly', () => {
      // Arrange
      const input = {
        header: {
          messageId: 'msg-456',
          timestamp: '2023-05-15T11:30:00Z',
          source: 'astroshield-subsystem1',
          messageType: 'object-details',
          traceId: 'trace-789'
        },
        payload: {
          catalogId: 'SATCAT-67890',
          name: 'Test Satellite',
          type: 'PAYLOAD',
          country: 'USA',
          launchDate: '2020-01-15',
          orbitType: 'LEO',
          averageAltitude: 550,
          status: 'ACTIVE'
        }
      };

      // Act
      const result = transformObjectDetails(input);

      // Assert
      expect(result).toEqual({
        catalogId: 'SATCAT-67890',
        name: 'Test Satellite',
        type: 'PAYLOAD',
        country: 'USA',
        launchDate: '2020-01-15',
        orbitType: 'LEO',
        averageAltitude: 550,
        status: 'ACTIVE',
        metadata: {
          messageId: 'msg-456',
          sourceTimestamp: '2023-05-15T11:30:00Z',
          source: 'astroshield-subsystem1',
          traceId: 'trace-789'
        }
      });
    });
  });

  describe('transformObservationWindow', () => {
    it('should transform observation window message correctly', () => {
      // Arrange
      const input = {
        header: {
          messageId: 'msg-789',
          timestamp: '2023-05-15T12:30:00Z',
          source: 'astroshield-subsystem2',
          messageType: 'observation-window',
          traceId: 'trace-101112'
        },
        payload: {
          catalogId: 'SATCAT-13579',
          startTime: '2023-05-16T02:00:00Z',
          endTime: '2023-05-16T02:15:00Z',
          sensorId: 'SENSOR-123',
          maxElevation: 67.5,
          quality: 'HIGH'
        }
      };

      // Act
      const result = transformObservationWindow(input);

      // Assert
      expect(result).toEqual({
        catalogId: 'SATCAT-13579',
        startTime: '2023-05-16T02:00:00Z',
        endTime: '2023-05-16T02:15:00Z',
        sensorId: 'SENSOR-123',
        maxElevation: 67.5,
        quality: 'HIGH',
        metadata: {
          messageId: 'msg-789',
          sourceTimestamp: '2023-05-15T12:30:00Z',
          source: 'astroshield-subsystem2',
          traceId: 'trace-101112'
        }
      });
    });
  });
}); 