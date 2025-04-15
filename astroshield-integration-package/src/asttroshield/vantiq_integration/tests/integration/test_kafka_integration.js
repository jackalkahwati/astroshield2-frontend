/**
 * Integration tests for Vantiq-Kafka integration
 * Tests the end-to-end flow of messages through Kafka
 */
const assert = require('assert');
const { Kafka } = require('kafkajs');
const testConfig = require('../test_config');
const { 
  transformManeuverToVantiq,
  transformAstroShieldToVantiq,
  transformVantiqToAstroShield
} = require('../../src/transformers');
const {
  publishToKafka,
  setupConsumer
} = require('../../src/kafka_client');

describe('Kafka Integration Tests', function() {
  // These tests may take longer to run due to Kafka setup time
  this.timeout(10000);
  
  // Create kafka client for testing
  let kafka;
  let producer;
  let consumer;
  let receivedMessages = [];
  
  before(async () => {
    // Skip tests if not in integration test environment
    if (!process.env.RUN_INTEGRATION_TESTS) {
      this.skip();
      return;
    }
    
    // Setup Kafka client
    kafka = new Kafka({
      clientId: testConfig.kafka.clientId,
      brokers: [testConfig.kafka.bootstrapServers]
    });
    
    // Setup producer
    producer = kafka.producer();
    await producer.connect();
    
    // Setup consumer with message handler
    consumer = kafka.consumer({ groupId: testConfig.kafka.groupId });
    await consumer.connect();
    await consumer.subscribe({ topic: testConfig.kafka.topics.vantiqInbound, fromBeginning: false });
    
    await consumer.run({
      eachMessage: async ({ topic, partition, message }) => {
        try {
          const messageValue = JSON.parse(message.value.toString());
          receivedMessages.push(messageValue);
        } catch (err) {
          console.error('Error processing message:', err);
        }
      }
    });
  });
  
  beforeEach(() => {
    // Clear received messages before each test
    receivedMessages = [];
  });
  
  after(async () => {
    // Clean up Kafka resources
    if (consumer) await consumer.disconnect();
    if (producer) await producer.disconnect();
  });
  
  describe('Message Flow Tests', () => {
    it('should publish and receive a transformed maneuver message', async () => {
      // Create a test maneuver message using the test helper
      const maneuverMessage = testConfig.helpers.createTestMessage('maneuver');
      
      // Transform and publish the message
      const transformedMessage = transformManeuverToVantiq(maneuverMessage);
      
      await producer.send({
        topic: testConfig.kafka.topics.vantiqInbound,
        messages: [
          { 
            value: JSON.stringify(transformedMessage),
            key: transformedMessage.id
          }
        ]
      });
      
      // Wait for the message to be received
      await testConfig.helpers.waitForCondition(() => receivedMessages.length >= 1);
      
      // Verify the received message
      assert.strictEqual(receivedMessages.length, 1);
      assert.strictEqual(receivedMessages[0].id, maneuverMessage.metadata.id);
      assert.strictEqual(receivedMessages[0].spacecraft_id, maneuverMessage.data.spacecraftId);
      assert.strictEqual(receivedMessages[0].maneuver_type, maneuverMessage.data.maneuverType);
      assert.deepStrictEqual(receivedMessages[0].origin.system, 'AstroShield');
    });
    
    it('should handle multiple messages in the correct order', async () => {
      // Create multiple test messages
      const messages = [];
      for (let i = 0; i < 3; i++) {
        messages.push(testConfig.helpers.createTestMessage('maneuver', {
          data: {
            parameters: { sequence: i }
          }
        }));
      }
      
      // Transform and publish all messages
      const kafkaMessages = messages.map(msg => {
        const transformed = transformManeuverToVantiq(msg);
        return {
          value: JSON.stringify(transformed),
          key: transformed.id
        };
      });
      
      await producer.send({
        topic: testConfig.kafka.topics.vantiqInbound,
        messages: kafkaMessages
      });
      
      // Wait for all messages to be received
      await testConfig.helpers.waitForCondition(() => receivedMessages.length >= 3);
      
      // Verify the received messages
      assert.strictEqual(receivedMessages.length, 3);
      
      // Check that the sequence was preserved
      for (let i = 0; i < 3; i++) {
        const original = messages.find(m => m.metadata.id === receivedMessages[i].id);
        assert.strictEqual(receivedMessages[i].spacecraft_id, original.data.spacecraftId);
        assert.deepStrictEqual(receivedMessages[i].parameters.sequence, original.data.parameters.sequence);
      }
    });
  });
  
  describe('Error Handling Tests', () => {
    it('should gracefully handle malformed messages', async () => {
      // Send a valid message first
      const validMessage = testConfig.helpers.createTestMessage('maneuver');
      
      const transformedValid = transformManeuverToVantiq(validMessage);
      
      // Then send an invalid JSON message
      await producer.send({
        topic: testConfig.kafka.topics.vantiqInbound,
        messages: [
          { 
            value: JSON.stringify(transformedValid),
            key: 'valid-message'
          },
          {
            value: 'This is not JSON',
            key: 'invalid-message'
          }
        ]
      });
      
      // We should still receive the valid message
      await testConfig.helpers.waitForCondition(() => receivedMessages.length >= 1);
      assert.strictEqual(receivedMessages.length, 1);
      assert.strictEqual(receivedMessages[0].id, validMessage.metadata.id);
    });
  });
}); 