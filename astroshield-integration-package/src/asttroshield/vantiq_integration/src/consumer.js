/**
 * Vantiq consumer module for receiving messages from Vantiq via Kafka
 */

const { Kafka } = require('kafkajs');
const { transformVantiqToAstroShield } = require('./transformers');

/**
 * Error class for configuration issues
 */
class ConfigError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ConfigError';
  }
}

/**
 * Error class for connection issues
 */
class ConnectionError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ConnectionError';
  }
}

/**
 * VantiqConsumer class for receiving messages from Vantiq
 */
class VantiqConsumer {
  /**
   * Create a new VantiqConsumer
   * @param {Object} config - Configuration object
   * @param {string} config.clientId - Kafka client ID
   * @param {Array<string>} config.brokers - Array of Kafka broker addresses
   * @param {string} config.groupId - Consumer group ID
   * @param {Object} [config.ssl] - SSL configuration
   * @param {Object} [config.sasl] - SASL authentication configuration
   * @param {Function} [config.messageHandler] - Function to handle received messages
   */
  constructor(config) {
    if (!config) {
      throw new ConfigError('Configuration is required');
    }

    if (!config.clientId) {
      throw new ConfigError('Client ID is required');
    }

    if (!config.brokers || !Array.isArray(config.brokers) || config.brokers.length === 0) {
      throw new ConfigError('At least one broker is required');
    }

    if (!config.groupId) {
      throw new ConfigError('Consumer group ID is required');
    }

    this.config = config;
    this.messageHandler = config.messageHandler;
    
    this.kafka = new Kafka({
      clientId: config.clientId,
      brokers: config.brokers,
      ssl: config.ssl || false,
      sasl: config.sasl || undefined
    });

    this.consumer = this.kafka.consumer({
      groupId: config.groupId
    });
    
    this.connected = false;
    this.subscriptions = new Set();
  }

  /**
   * Set message handler function
   * @param {Function} handler - Function to handle received messages
   */
  setMessageHandler(handler) {
    if (typeof handler !== 'function') {
      throw new ConfigError('Message handler must be a function');
    }
    this.messageHandler = handler;
  }

  /**
   * Connect to Kafka
   * @returns {Promise<void>}
   */
  async connect() {
    try {
      await this.consumer.connect();
      this.connected = true;
    } catch (error) {
      throw new ConnectionError(`Failed to connect to Kafka: ${error.message}`);
    }
  }

  /**
   * Disconnect from Kafka
   * @returns {Promise<void>}
   */
  async disconnect() {
    if (this.connected) {
      try {
        await this.consumer.disconnect();
        this.connected = false;
        this.subscriptions.clear();
      } catch (error) {
        throw new ConnectionError(`Failed to disconnect from Kafka: ${error.message}`);
      }
    }
  }

  /**
   * Subscribe to a topic
   * @param {string} topic - Kafka topic to subscribe to
   * @returns {Promise<void>}
   */
  async subscribe(topic) {
    if (!this.connected) {
      throw new ConnectionError('Not connected to Kafka');
    }

    if (!topic) {
      throw new ConfigError('Topic is required');
    }

    if (this.subscriptions.has(topic)) {
      return; // Already subscribed
    }

    try {
      await this.consumer.subscribe({ topic, fromBeginning: false });
      this.subscriptions.add(topic);
    } catch (error) {
      throw new Error(`Failed to subscribe to topic ${topic}: ${error.message}`);
    }
  }

  /**
   * Start consuming messages
   * @returns {Promise<void>}
   */
  async start() {
    if (!this.connected) {
      throw new ConnectionError('Not connected to Kafka');
    }

    if (this.subscriptions.size === 0) {
      throw new ConfigError('No topic subscriptions. Call subscribe() before start()');
    }

    if (!this.messageHandler) {
      throw new ConfigError('No message handler set. Set a message handler before starting consumption');
    }

    try {
      await this.consumer.run({
        eachMessage: async ({ topic, partition, message }) => {
          try {
            const value = message.value ? JSON.parse(message.value.toString()) : null;
            
            if (!value) {
              console.warn('Received empty message, skipping');
              return;
            }

            // Extract message type from headers or topic
            let messageType = null;
            if (message.headers && message.headers.messageType) {
              messageType = message.headers.messageType.toString();
            } else {
              // Try to infer type from topic
              messageType = this.extractMessageTypeFromTopic(topic);
            }

            if (!messageType) {
              console.warn(`Unable to determine message type for message on topic ${topic}`);
              return;
            }

            // Transform the message to AstroShield format
            const transformedMessage = transformVantiqToAstroShield(value, messageType, topic);
            
            // Pass the transformed message to the handler
            await this.messageHandler(transformedMessage, messageType, topic);
          } catch (error) {
            console.error(`Error processing message from topic ${topic}: ${error.message}`);
          }
        }
      });
    } catch (error) {
      throw new Error(`Failed to start consumer: ${error.message}`);
    }
  }

  /**
   * Stop consuming messages
   * @returns {Promise<void>}
   */
  async stop() {
    if (this.connected) {
      try {
        await this.consumer.stop();
      } catch (error) {
        throw new Error(`Failed to stop consumer: ${error.message}`);
      }
    }
  }

  /**
   * Extract message type from topic
   * @param {string} topic - Kafka topic
   * @returns {string|null} - Message type or null if not determined
   * @private
   */
  extractMessageTypeFromTopic(topic) {
    if (topic.includes('maneuver')) {
      return 'maneuver';
    } else if (topic.includes('object')) {
      return 'object';
    } else if (topic.includes('observation')) {
      return 'observation';
    }
    return null;
  }
}

module.exports = {
  VantiqConsumer,
  ConfigError,
  ConnectionError
}; 