/**
 * AstroShield Vantiq Integration module
 * Connects to AstroShield message bus and forwards events to Vantiq
 */
const VantiqClient = require('./vantiqClient');
const { EventConsumer } = require('../../common/messaging');
const logger = require('../../common/logger');
const { 
  ConfigError,
  ConnectionError 
} = require('../../errors');

class VantiqIntegration {
  /**
   * Creates a new Vantiq integration
   * @param {Object} config - Configuration object
   */
  constructor(config) {
    this.config = this.validateConfig(config);
    this.vantiqClient = new VantiqClient(this.config.vantiq);
    this.eventConsumer = new EventConsumer({
      topics: this.config.subscriptions,
      groupId: this.config.consumer.groupId,
      brokers: this.config.consumer.brokers
    });
    
    this.isRunning = false;
  }
  
  /**
   * Validates configuration
   * @param {Object} config - Config to validate
   * @returns {Object} Validated config
   * @throws {ConfigError} If config is invalid
   */
  validateConfig(config) {
    if (!config) {
      throw new ConfigError('Missing configuration');
    }
    
    const requiredConfigs = [
      'vantiq.brokers',
      'vantiq.clientId',
      'vantiq.maneuverTopic',
      'vantiq.objectDetailsTopic',
      'vantiq.observationTopic', 
      'consumer.brokers',
      'consumer.groupId',
      'subscriptions'
    ];
    
    for (const path of requiredConfigs) {
      const parts = path.split('.');
      let current = config;
      
      for (const part of parts) {
        if (!current || !current[part]) {
          throw new ConfigError(`Missing required config: ${path}`);
        }
        current = current[part];
      }
    }
    
    if (!Array.isArray(config.subscriptions) || config.subscriptions.length === 0) {
      throw new ConfigError('subscriptions must be a non-empty array');
    }
    
    return config;
  }
  
  /**
   * Initializes the integration
   * @returns {Promise<void>}
   */
  async initialize() {
    try {
      logger.info('Initializing Vantiq integration');
      await this.vantiqClient.connect();
      
      this.eventConsumer.on('message', this.handleMessage.bind(this));
      this.eventConsumer.on('error', this.handleError.bind(this));
      
      await this.eventConsumer.connect();
      logger.info('Vantiq integration initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize Vantiq integration', error);
      throw new ConnectionError('Failed to initialize Vantiq integration', error);
    }
  }
  
  /**
   * Starts the integration
   * @returns {Promise<void>}
   */
  async start() {
    if (this.isRunning) {
      logger.warn('Vantiq integration is already running');
      return;
    }
    
    try {
      logger.info('Starting Vantiq integration');
      if (!this.eventConsumer.isConnected) {
        await this.initialize();
      }
      
      await this.eventConsumer.subscribe();
      this.isRunning = true;
      logger.info('Vantiq integration started successfully');
    } catch (error) {
      logger.error('Failed to start Vantiq integration', error);
      throw error;
    }
  }
  
  /**
   * Stops the integration
   * @returns {Promise<void>}
   */
  async stop() {
    if (!this.isRunning) {
      logger.warn('Vantiq integration is not running');
      return;
    }
    
    try {
      logger.info('Stopping Vantiq integration');
      await this.eventConsumer.disconnect();
      await this.vantiqClient.disconnect();
      this.isRunning = false;
      logger.info('Vantiq integration stopped successfully');
    } catch (error) {
      logger.error('Failed to stop Vantiq integration', error);
      throw error;
    }
  }
  
  /**
   * Handles incoming messages
   * @param {Object} message - Message from Kafka
   * @returns {Promise<void>}
   */
  async handleMessage(message) {
    try {
      const parsedMessage = JSON.parse(message.value.toString());
      
      // Check if it's a supported message type
      const messageType = parsedMessage.header?.messageType;
      if (!messageType || !this.vantiqClient.topicMap[messageType]) {
        logger.debug('Ignoring unsupported message type', { messageType });
        return;
      }
      
      logger.debug('Received message', { 
        messageId: parsedMessage.header.messageId,
        messageType
      });
      
      await this.vantiqClient.publishMessage(parsedMessage);
    } catch (error) {
      logger.error('Error handling message', error);
      // Continue processing messages even if one fails
    }
  }
  
  /**
   * Handles errors from the event consumer
   * @param {Error} error - Error from consumer
   */
  handleError(error) {
    logger.error('Event consumer error', error);
    // Implement error handling/retry logic as needed
  }
}

module.exports = VantiqIntegration; 