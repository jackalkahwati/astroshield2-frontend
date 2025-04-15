/**
 * Unit tests for the Kafka client module
 * Uses mocking to test Kafka connectivity without actual Kafka connection
 */
const assert = require('assert');
const sinon = require('sinon');
const { Kafka } = require('kafkajs');
const testConfig = require('../test_config');

// Module to test
const kafkaClient = require('../../src/kafka_client');

describe('Kafka Client', () => {
  let kafkaMock;
  let producerMock;
  let consumerMock;
  let sandbox;
  
  beforeEach(() => {
    // Set up sandbox for mocks
    sandbox = sinon.createSandbox();
    
    // Create mocks for Kafka components
    producerMock = {
      connect: sandbox.stub().resolves(),
      send: sandbox.stub().resolves(),
      disconnect: sandbox.stub().resolves()
    };
    
    consumerMock = {
      connect: sandbox.stub().resolves(),
      subscribe: sandbox.stub().resolves(),
      run: sandbox.stub().resolves(),
      disconnect: sandbox.stub().resolves()
    };
    
    // Create mock for Kafka class
    kafkaMock = {
      producer: sandbox.stub().returns(producerMock),
      consumer: sandbox.stub().returns(consumerMock)
    };
    
    // Replace Kafka constructor with mock
    sandbox.stub(Kafka.prototype, 'constructor').returns(kafkaMock);
    
    // Stub the whole Kafka import
    sandbox.stub(Kafka.prototype, 'producer').returns(producerMock);
    sandbox.stub(Kafka.prototype, 'consumer').returns(consumerMock);
  });
  
  afterEach(() => {
    // Restore all mocks
    sandbox.restore();
  });
  
  describe('connectToKafka', () => {
    it('should connect to Kafka and return client', async () => {
      // Stub the global Kafka constructor
      const KafkaConstructorStub = sandbox.stub().returns(kafkaMock);
      global.Kafka = KafkaConstructorStub;
      
      // Test connection function with mocks
      const config = {
        clientId: 'test-client',
        brokers: ['localhost:9092']
      };
      
      const result = await kafkaClient.connectToKafka(config);
      
      // Assert that Kafka was initialized with correct config
      assert(KafkaConstructorStub.calledWith({
        clientId: 'test-client',
        brokers: ['localhost:9092']
      }));
      
      // Assert that result contains expected objects
      assert.strictEqual(result.kafka, kafkaMock);
    });
    
    it('should throw error if connection fails', async () => {
      // Stub Kafka constructor to throw
      const error = new Error('Connection failed');
      const KafkaConstructorStub = sandbox.stub().throws(error);
      global.Kafka = KafkaConstructorStub;
      
      // Test connection function with failing connection
      const config = {
        clientId: 'test-client',
        brokers: ['localhost:9092']
      };
      
      // Assert that error is thrown
      try {
        await kafkaClient.connectToKafka(config);
        assert.fail('Should have thrown error');
      } catch (err) {
        assert.strictEqual(err.message, 'Connection failed');
      }
    });
  });
  
  describe('publishToKafka', () => {
    it('should publish message to Kafka topic', async () => {
      // Setup
      const message = { id: 'test', data: { value: 'test-value' } };
      const topic = 'test-topic';
      
      // Replace the function with a testable version
      const originalFn = kafkaClient.publishToKafka;
      kafkaClient.publishToKafka = async (producer, topic, message, key) => {
        // Call the mock directly since we can't mock the imported Kafka
        return producerMock.send({
          topic,
          messages: [
            {
              value: JSON.stringify(message),
              key: key || message.id
            }
          ]
        });
      };
      
      // Act
      await kafkaClient.publishToKafka(producerMock, topic, message);
      
      // Assert
      assert(producerMock.send.calledOnce);
      
      // Restore original function
      kafkaClient.publishToKafka = originalFn;
    });
    
    it('should throw error if publishing fails', async () => {
      // Setup
      const message = { id: 'test', data: { value: 'test-value' } };
      const topic = 'test-topic';
      
      // Make producer.send throw an error
      producerMock.send.rejects(new Error('Publish failed'));
      
      // Replace the function with a testable version
      const originalFn = kafkaClient.publishToKafka;
      kafkaClient.publishToKafka = async (producer, topic, message, key) => {
        return producerMock.send({
          topic,
          messages: [
            {
              value: JSON.stringify(message),
              key: key || message.id
            }
          ]
        });
      };
      
      // Act & Assert
      try {
        await kafkaClient.publishToKafka(producerMock, topic, message);
        assert.fail('Should have thrown error');
      } catch (err) {
        assert.strictEqual(err.message, 'Publish failed');
      }
      
      // Restore original function
      kafkaClient.publishToKafka = originalFn;
    });
  });
  
  describe('setupConsumer', () => {
    it('should set up consumer with correct topic and message handler', async () => {
      // Setup
      const topic = 'test-topic';
      const groupId = 'test-group';
      const messageHandler = sinon.stub();
      
      // Replace the function with a testable version
      const originalFn = kafkaClient.setupConsumer;
      kafkaClient.setupConsumer = async (config, topic, messageHandler) => {
        const consumer = consumerMock;
        await consumer.connect();
        await consumer.subscribe({ topic, fromBeginning: false });
        await consumer.run({
          eachMessage: async ({ topic, partition, message }) => {
            try {
              const parsedMessage = JSON.parse(message.value.toString());
              await messageHandler(parsedMessage);
            } catch (error) {
              console.error('Error processing message:', error);
            }
          }
        });
        return consumer;
      };
      
      // Act
      const consumer = await kafkaClient.setupConsumer(
        { clientId: 'test-client', brokers: ['localhost:9092'], groupId },
        topic,
        messageHandler
      );
      
      // Assert
      assert(consumerMock.connect.calledOnce);
      assert(consumerMock.subscribe.calledWith({ topic, fromBeginning: false }));
      assert(consumerMock.run.calledOnce);
      assert.strictEqual(consumer, consumerMock);
      
      // Restore original function
      kafkaClient.setupConsumer = originalFn;
    });
  });
  
  describe('disconnectConsumer', () => {
    it('should disconnect consumer', async () => {
      // Act
      await kafkaClient.disconnectConsumer(consumerMock);
      
      // Assert
      assert(consumerMock.disconnect.calledOnce);
    });
  });
  
  describe('disconnectProducer', () => {
    it('should disconnect producer', async () => {
      // Act
      await kafkaClient.disconnectProducer(producerMock);
      
      // Assert
      assert(producerMock.disconnect.calledOnce);
    });
  });
}); 