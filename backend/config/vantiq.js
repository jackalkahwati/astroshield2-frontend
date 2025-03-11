/**
 * Vantiq Configuration
 * 
 * This module contains configuration settings for Vantiq integration.
 */

require('dotenv').config();

const vantiqConfig = {
  vantiq: {
    apiUrl: process.env.VANTIQ_API_URL || 'https://dev.vantiq.com/api/v1',
    apiToken: process.env.VANTIQ_API_TOKEN || '',
    namespace: process.env.VANTIQ_NAMESPACE || 'asttroshield',
    sharedSecret: process.env.VANTIQ_WEBHOOK_SECRET || 'default-secret-change-me',
    
    // Authentication settings
    auth: {
      username: process.env.VANTIQ_USERNAME || '',
      password: process.env.VANTIQ_PASSWORD || '',
      clientId: process.env.VANTIQ_CLIENT_ID || '',
      clientSecret: process.env.VANTIQ_CLIENT_SECRET || '',
    },
    
    // Topic mappings
    topics: {
      trajectoryUpdates: process.env.VANTIQ_TRAJECTORY_TOPIC || 'TRAJECTORY_UPDATES',
      threatDetections: process.env.VANTIQ_THREAT_TOPIC || 'THREAT_DETECTIONS',
      commands: process.env.VANTIQ_COMMAND_TOPIC || 'COMMANDS',
    },
    
    // Webhook configuration
    webhook: {
      enabled: process.env.VANTIQ_WEBHOOK_ENABLED === 'true',
      endpoint: process.env.VANTIQ_WEBHOOK_ENDPOINT || '/api/vantiq/webhook',
    },
    
    // Retry configuration
    retry: {
      maxRetries: parseInt(process.env.VANTIQ_MAX_RETRIES || '3', 10),
      retryDelay: parseInt(process.env.VANTIQ_RETRY_DELAY || '1000', 10),
      timeout: parseInt(process.env.VANTIQ_TIMEOUT || '5000', 10),
    },
  }
};

module.exports = vantiqConfig; 