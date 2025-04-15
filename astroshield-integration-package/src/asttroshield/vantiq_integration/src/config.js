/**
 * Configuration module for the Vantiq integration
 */
const { ConfigError } = require('../../errors');

// Default configuration values
const DEFAULT_CONFIG = {
  vantiq: {
    brokers: ['localhost:9092'],
    clientId: 'asttroshield-vantiq-producer',
    maneuverTopic: 'vantiq.maneuvers',
    objectDetailsTopic: 'vantiq.objects',
    observationTopic: 'vantiq.observations'
  },
  consumer: {
    brokers: ['localhost:9092'],
    groupId: 'asttroshield-vantiq-integration'
  },
  subscriptions: [
    'asttroshield.maneuvers',
    'asttroshield.object_details',
    'asttroshield.observations'
  ]
};

/**
 * Loads configuration from environment variables and merges with defaults
 * @returns {Object} Configuration object
 */
function loadConfig() {
  const config = JSON.parse(JSON.stringify(DEFAULT_CONFIG)); // Deep clone
  
  // Load Vantiq Kafka configuration
  if (process.env.VANTIQ_KAFKA_BROKERS) {
    config.vantiq.brokers = process.env.VANTIQ_KAFKA_BROKERS.split(',');
  }
  
  if (process.env.VANTIQ_KAFKA_CLIENT_ID) {
    config.vantiq.clientId = process.env.VANTIQ_KAFKA_CLIENT_ID;
  }
  
  if (process.env.VANTIQ_MANEUVER_TOPIC) {
    config.vantiq.maneuverTopic = process.env.VANTIQ_MANEUVER_TOPIC;
  }
  
  if (process.env.VANTIQ_OBJECT_TOPIC) {
    config.vantiq.objectDetailsTopic = process.env.VANTIQ_OBJECT_TOPIC;
  }
  
  if (process.env.VANTIQ_OBSERVATION_TOPIC) {
    config.vantiq.observationTopic = process.env.VANTIQ_OBSERVATION_TOPIC;
  }
  
  // Load AstroShield Kafka consumer configuration
  if (process.env.KAFKA_BROKERS) {
    config.consumer.brokers = process.env.KAFKA_BROKERS.split(',');
  }
  
  if (process.env.KAFKA_GROUP_ID) {
    config.consumer.groupId = process.env.KAFKA_GROUP_ID;
  }
  
  // Load subscription configuration
  if (process.env.SUBSCRIBED_TOPICS) {
    try {
      config.subscriptions = process.env.SUBSCRIBED_TOPICS.split(',');
    } catch (error) {
      throw new ConfigError('Invalid SUBSCRIBED_TOPICS format. Expected comma-separated list.', error);
    }
  }
  
  return config;
}

module.exports = {
  DEFAULT_CONFIG,
  loadConfig
}; 