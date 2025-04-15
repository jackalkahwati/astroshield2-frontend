/**
 * Unit tests for the Astroshield REST client
 */
const assert = require('assert');
const axios = require('axios');
const AstroshieldClient = require('../src/restClient');

// Mock axios for testing
jest.mock('axios');

describe('AstroshieldClient', () => {
  let client;
  
  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();
    
    // Create a new client instance for each test
    client = new AstroshieldClient('https://api.astroshield.com', 'test-api-key');
    
    // Set up the axios mock
    axios.create.mockReturnValue({
      get: jest.fn().mockImplementation(() => Promise.resolve({ data: {} }))
    });
  });
  
  describe('constructor', () => {
    it('should create a client instance with correct configuration', () => {
      expect(axios.create).toHaveBeenCalledWith({
        baseURL: 'https://api.astroshield.com',
        headers: {
          'Authorization': 'Bearer test-api-key',
          'Content-Type': 'application/json'
        }
      });
    });
  });
  
  describe('getObjectManeuverHistory', () => {
    it('should make a GET request to the maneuvers endpoint with correct parameters', async () => {
      // Mock API response
      const mockResponse = {
        data: [
          {
            catalogId: 'SATCAT-12345',
            deltaV: 0.25,
            timestamp: '2023-09-01T12:00:00Z',
            type: 'MAJOR_MANEUVER'
          }
        ]
      };
      
      // Set up the mock
      client.axiosInstance.get.mockResolvedValueOnce(mockResponse);
      
      // Call the method
      const result = await client.getObjectManeuverHistory(
        'SATCAT-12345', 
        '2023-09-01T00:00:00Z', 
        '2023-09-02T00:00:00Z'
      );
      
      // Assert the result
      expect(result).toEqual(mockResponse.data);
      
      // Assert the request was made correctly
      expect(client.axiosInstance.get).toHaveBeenCalledWith(
        '/api/v1/maneuvers',
        {
          params: {
            catalogId: 'SATCAT-12345',
            startDate: '2023-09-01T00:00:00Z',
            endDate: '2023-09-02T00:00:00Z'
          }
        }
      );
    });
    
    it('should handle API errors gracefully', async () => {
      // Set up the mock to reject
      const error = new Error('API Error');
      error.response = { status: 500, data: { message: 'Internal Server Error' } };
      client.axiosInstance.get.mockRejectedValueOnce(error);
      
      // Expect the method to throw
      await expect(
        client.getObjectManeuverHistory('SATCAT-12345', '2023-09-01', '2023-09-02')
      ).rejects.toThrow('API Error');
    });
  });
  
  describe('getObjectDetails', () => {
    it('should make a GET request to the objects endpoint with correct parameters', async () => {
      // Mock API response
      const mockResponse = {
        data: {
          catalogId: 'SATCAT-12345',
          name: 'Test Satellite',
          type: 'ACTIVE_PAYLOAD',
          orbit: {
            type: 'LEO',
            altitude: 550,
            inclination: 51.6
          }
        }
      };
      
      // Set up the mock
      client.axiosInstance.get.mockResolvedValueOnce(mockResponse);
      
      // Call the method
      const result = await client.getObjectDetails('SATCAT-12345');
      
      // Assert the result
      expect(result).toEqual(mockResponse.data);
      
      // Assert the request was made correctly
      expect(client.axiosInstance.get).toHaveBeenCalledWith('/api/v1/objects/SATCAT-12345');
    });
  });
  
  describe('getFutureObservationWindows', () => {
    it('should make a GET request to the forecast endpoint with correct parameters', async () => {
      // Mock API response
      const mockResponse = {
        data: [
          {
            location: {
              latitude: 32.9,
              longitude: -117.2,
              locationName: 'San Diego Observatory'
            },
            qualityScore: 0.85,
            observationWindow: {
              startTime: '2023-09-02T22:00:00Z',
              endTime: '2023-09-03T01:00:00Z',
              durationMinutes: 180
            }
          }
        ]
      };
      
      // Set up the mock
      client.axiosInstance.get.mockResolvedValueOnce(mockResponse);
      
      // Call the method
      const result = await client.getFutureObservationWindows('SAN_DIEGO_OBS', 48);
      
      // Assert the result
      expect(result).toEqual(mockResponse.data);
      
      // Assert the request was made correctly
      expect(client.axiosInstance.get).toHaveBeenCalledWith(
        '/api/v1/observations/forecast',
        {
          params: {
            locationId: 'SAN_DIEGO_OBS',
            hours: 48
          }
        }
      );
    });
  });
}); 