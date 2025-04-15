/**
 * Vantiq client for Kafka integration
 */
const { Kafka } = require('kafkajs');
const { transformMessage } = require('./messageTransformer');
const logger = require('../../common/logger');

class VantiqClient {
  /**
   * Creates a new Vantiq client
   * @param {Object} config - Configuration object
   * @param {string} config.brokers - Comma-separated list of Kafka brokers
   * @param {string} config.clientId - Client ID for Kafka
   * @param {string} config.maneuverTopic - Topic for maneuver messages
   * @param {string} config.objectDetailsTopic - Topic for object details
   * @param {string} config.observationTopic - Topic for observation windows
   */
  constructor(config) {
    this.config = config;
    this.kafka = new Kafka({
      clientId: config.clientId,
      brokers: config.brokers.split(','),
      ssl: config.ssl || true,
      sasl: config.sasl || undefined
    });
    
    this.producer = this.kafka.producer();
    this.isConnected = false;
    this.topicMap = {
      'maneuver-detected': config.maneuverTopic,
      'object-details': config.objectDetailsTopic,
      'observation-window': config.observationTopic
    };
  }
  
  /**
   * Connects to Kafka
   * @returns {Promise<void>}
   */
  async connect() {
    try {
      await this.producer.connect();
      this.isConnected = true;
      logger.info('Connected to Vantiq Kafka broker');
    } catch (error) {
      logger.error('Failed to connect to Vantiq Kafka broker', error);
      throw error;
    }
  }
  
  /**
   * Disconnects from Kafka
   * @returns {Promise<void>}
   */
  async disconnect() {
    if (!this.isConnected) {
      return;
    }
    
    try {
      await this.producer.disconnect();
      this.isConnected = false;
      logger.info('Disconnected from Vantiq Kafka broker');
    } catch (error) {
      logger.error('Failed to disconnect from Vantiq Kafka broker', error);
      throw error;
    }
  }
  
  /**
   * Publishes a message to the appropriate Kafka topic
   * @param {Object} message - AstroShield message
   * @returns {Promise<void>}
   */
  async publishMessage(message) {
    if (!this.isConnected) {
      throw new Error('Not connected to Kafka broker');
    }
    
    try {
      const messageType = message.header.messageType;
      const topic = this.topicMap[messageType];
      
      if (!topic) {
        throw new Error(`No topic configured for message type: ${messageType}`);
      }
      
      const transformedMessage = transformMessage(message);
      
      await this.producer.send({
        topic,
        messages: [{
          value: JSON.stringify(transformedMessage)
        }]
      });
      
      logger.info(`Message published to topic ${topic}`, { messageId: message.header.messageId });
    } catch (error) {
      logger.error('Failed to publish message', error);
      throw error;
    }
  }
}

module.exports = VantiqClient; 