/**
 * Test configuration for AstroShield Vantiq integration
 * This file contains shared configuration for both unit and integration tests
 */

// Default test configuration
const testConfig = {
  // Kafka configuration
  kafka: {
    bootstrapServers: process.env.KAFKA_BOOTSTRAP_SERVERS || 'localhost:9092',
    clientId: 'astroshield-vantiq-test-client',
    groupId: 'astroshield-vantiq-test-group',
    topics: {
      astroShieldOutbound: 'astroshield-test-outbound',
      vantiqInbound: 'vantiq-test-inbound',
      vantiqOutbound: 'vantiq-test-outbound',
      astroShieldInbound: 'astroshield-test-inbound'
    }
  },
  
  // Test message templates
  messageTemplates: {
    // Maneuver message template for testing
    maneuver: {
      metadata: {
        id: 'test-maneuver-',  // Will be appended with unique ID
        timestamp: '', // Will be set to current time
        source: 'subsystem3_command_control',
        version: '1.0.0'
      },
      data: {
        spacecraftId: 'TEST-SAT-',  // Will be appended with unique ID
        maneuverType: 'STATION_KEEPING',
        parameters: {
          deltaV: 0.25,
          burnDuration: 30
        },
        priority: 'high',
        executionWindow: {
          start: '', // Will be set to future time
          end: ''    // Will be set to future time
        },
        status: 'scheduled'
      }
    },
    
    // Object details message template for testing
    objectDetails: {
      metadata: {
        id: 'test-object-',  // Will be appended with unique ID
        timestamp: '', // Will be set to current time
        source: 'subsystem1_target_modeling',
        version: '1.0.0'
      },
      data: {
        objectId: 'OBJ-',  // Will be appended with unique ID
        objectType: 'SATELLITE',
        classification: 'OPERATIONAL',
        parameters: {
          size: 'SMALL',
          rcs: 0.75
        },
        stateVector: {
          position: [7000, 0, 0],
          velocity: [0, 7.5, 0]
        },
        confidence: 0.95,
        additionalProperties: {
          owner: 'COMMERCIAL',
          launchDate: '2022-03-15'
        }
      }
    },
    
    // Observation message template for testing
    observation: {
      metadata: {
        id: 'test-observation-',  // Will be appended with unique ID
        timestamp: '', // Will be set to current time
        source: 'subsystem0_data_ingestion',
        version: '1.0.0'
      },
      data: {
        sensorId: 'SENSOR-',  // Will be appended with unique ID
        objectId: 'OBJ-',     // Will be appended with unique ID
        measurementType: 'OPTICAL',
        values: {
          azimuth: 178.5,
          elevation: 45.2,
          range: 800
        },
        quality: 0.87,
        observationTime: '' // Will be set to current time
      }
    }
  },
  
  // Helper functions for tests
  helpers: {
    /**
     * Create a test message based on a template
     * @param {string} type - The type of message (maneuver, objectDetails, observation)
     * @param {object} overrides - Fields to override in the template
     * @returns {object} A test message
     */
    createTestMessage: (type, overrides = {}) => {
      if (!testConfig.messageTemplates[type]) {
        throw new Error(`Unknown message type: ${type}`);
      }
      
      // Deep clone the template
      const message = JSON.parse(JSON.stringify(testConfig.messageTemplates[type]));
      
      // Set timestamp and ID fields
      const uniqueId = Date.now().toString(36) + Math.random().toString(36).substring(2, 7);
      message.metadata.id += uniqueId;
      message.metadata.timestamp = new Date().toISOString();
      
      // Set type-specific fields
      switch (type) {
        case 'maneuver':
          message.data.spacecraftId += uniqueId;
          message.data.executionWindow.start = new Date(Date.now() + 3600000).toISOString(); // 1 hour from now
          message.data.executionWindow.end = new Date(Date.now() + 7200000).toISOString(); // 2 hours from now
          break;
        case 'objectDetails':
          message.data.objectId += uniqueId;
          break;
        case 'observation':
          message.data.sensorId += uniqueId;
          message.data.objectId += uniqueId;
          message.data.observationTime = new Date().toISOString();
          break;
      }
      
      // Apply any overrides
      const mergeOverrides = (target, source) => {
        for (const key in source) {
          if (source[key] instanceof Object && !(source[key] instanceof Array) && target[key]) {
            mergeOverrides(target[key], source[key]);
          } else {
            target[key] = source[key];
          }
        }
      };
      
      mergeOverrides(message, overrides);
      
      return message;
    },
    
    /**
     * Wait for a condition to be true
     * @param {function} conditionFn - Function that returns true when condition is met
     * @param {number} timeout - Timeout in milliseconds
     * @param {number} interval - Check interval in milliseconds
     * @returns {Promise} Resolves when condition is met, rejects on timeout
     */
    waitForCondition: (conditionFn, timeout = 5000, interval = 100) => {
      return new Promise((resolve, reject) => {
        let elapsed = 0;
        
        const checkInterval = setInterval(() => {
          if (conditionFn()) {
            clearInterval(checkInterval);
            resolve();
            return;
          }
          
          elapsed += interval;
          if (elapsed >= timeout) {
            clearInterval(checkInterval);
            reject(new Error(`Timeout waiting for condition after ${timeout}ms`));
          }
        }, interval);
      });
    }
  }
};

module.exports = testConfig; 