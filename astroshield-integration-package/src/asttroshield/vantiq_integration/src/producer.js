/**
 * Vantiq producer module for sending messages to Vantiq via Kafka
 */

const { Kafka } = require('kafkajs');
const { transformAstroShieldToVantiq } = require('./transformers');

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
 * VantiqProducer class for sending messages to Vantiq
 */
class VantiqProducer {
  /**
   * Create a new VantiqProducer
   * @param {Object} config - Configuration object
   * @param {string} config.clientId - Kafka client ID
   * @param {Array<string>} config.brokers - Array of Kafka broker addresses
   * @param {Object} [config.ssl] - SSL configuration
   * @param {Object} [config.sasl] - SASL authentication configuration
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

    this.config = config;
    this.kafka = new Kafka({
      clientId: config.clientId,
      brokers: config.brokers,
      ssl: config.ssl || false,
      sasl: config.sasl || undefined
    });

    this.producer = this.kafka.producer();
    this.connected = false;
  }

  /**
   * Connect to Kafka
   * @returns {Promise<void>}
   */
  async connect() {
    try {
      await this.producer.connect();
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
        await this.producer.disconnect();
        this.connected = false;
      } catch (error) {
        throw new ConnectionError(`Failed to disconnect from Kafka: ${error.message}`);
      }
    }
  }

  /**
   * Send a message to a topic
   * @param {string} topic - Kafka topic to send message to
   * @param {Object} message - Message to send
   * @param {string} messageType - Type of message (maneuver, object, observation)
   * @returns {Promise<Object>} - Metadata about the sent message
   */
  async sendMessage(topic, message, messageType) {
    if (!this.connected) {
      throw new ConnectionError('Not connected to Kafka');
    }

    if (!topic) {
      throw new ConfigError('Topic is required');
    }

    if (!message) {
      throw new ConfigError('Message is required');
    }

    if (!messageType) {
      throw new ConfigError('Message type is required');
    }

    // Transform the message to Vantiq format
    const transformedMessage = transformAstroShieldToVantiq(message, messageType);

    try {
      return await this.producer.send({
        topic,
        messages: [
          {
            value: JSON.stringify(transformedMessage),
            headers: {
              source: 'ASTROSHIELD',
              timestamp: Date.now().toString(),
              messageType
            }
          }
        ]
      });
    } catch (error) {
      throw new Error(`Failed to send message: ${error.message}`);
    }
  }

  /**
   * Send a batch of messages to a topic
   * @param {string} topic - Kafka topic to send messages to
   * @param {Array<Object>} messages - Array of messages to send
   * @param {string} messageType - Type of messages (maneuver, object, observation)
   * @returns {Promise<Object>} - Metadata about the sent messages
   */
  async sendBatch(topic, messages, messageType) {
    if (!this.connected) {
      throw new ConnectionError('Not connected to Kafka');
    }

    if (!topic) {
      throw new ConfigError('Topic is required');
    }

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      throw new ConfigError('Messages array is required and must not be empty');
    }

    if (!messageType) {
      throw new ConfigError('Message type is required');
    }

    // Transform all messages to Vantiq format
    const kafkaMessages = messages.map(message => {
      const transformedMessage = transformAstroShieldToVantiq(message, messageType);
      return {
        value: JSON.stringify(transformedMessage),
        headers: {
          source: 'ASTROSHIELD',
          timestamp: Date.now().toString(),
          messageType
        }
      };
    });

    try {
      return await this.producer.send({
        topic,
        messages: kafkaMessages
      });
    } catch (error) {
      throw new Error(`Failed to send batch of messages: ${error.message}`);
    }
  }
}

module.exports = {
  VantiqProducer,
  ConfigError,
  ConnectionError
}; 