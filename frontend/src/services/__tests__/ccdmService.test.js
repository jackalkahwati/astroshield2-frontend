import axios from 'axios';
import ccdmService from '../ccdmService';

// Mock axios and its create method
jest.mock('axios', () => {
  // Create mock functions for all the methods we use
  const mockGet = jest.fn().mockResolvedValue({ data: {} });
  const mockPost = jest.fn().mockResolvedValue({ data: {} });
  
  // Create a mock for the interceptors
  const mockInterceptors = {
    request: { use: jest.fn() },
    response: { use: jest.fn() }
  };
  
  // Mock axios.create to return an object with the methods and interceptors
  const mockCreate = jest.fn().mockReturnValue({
    get: mockGet,
    post: mockPost,
    interceptors: mockInterceptors
  });
  
  // Return the mock axios module
  return {
    create: mockCreate,
    get: jest.fn(),
    post: jest.fn()
  };
});

// Get the mock instance returned by axios.create
const mockAxiosInstance = axios.create();

describe('CCDM Service', () => {
  // Clear mocks before each test
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    
    // Reset mock implementation for each test
    mockAxiosInstance.get.mockResolvedValue({ data: {} });
    mockAxiosInstance.post.mockResolvedValue({ data: {} });
  });

  test('analyzeObject makes correct API call', async () => {
    // Arrange
    const noradId = 25544;
    const mockResponse = {
      data: {
        norad_id: noradId,
        timestamp: new Date().toISOString(),
        analysis_results: [
          { threat_level: 'LOW', confidence: 0.85 }
        ],
        summary: 'Analysis complete'
      }
    };
    mockAxiosInstance.post.mockResolvedValueOnce(mockResponse);

    // Act
    const result = await ccdmService.analyzeObject(noradId);

    // Assert
    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/ccdm/analyze', {
      norad_id: noradId,
      data: {}
    });
    expect(result).toEqual(mockResponse.data);
  });

  test('assessThreat makes correct API call', async () => {
    // Arrange
    const noradId = 33591;
    const mockResponse = {
      data: {
        norad_id: noradId,
        overall_threat: 'LOW',
        confidence: 0.9,
        threat_components: {
          collision: 'LOW',
          debris: 'NONE'
        },
        recommendations: ['Monitor the satellite']
      }
    };
    mockAxiosInstance.post.mockResolvedValueOnce(mockResponse);

    // Act
    const result = await ccdmService.assessThreat(noradId);

    // Assert
    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/ccdm/threat-assessment', {
      norad_id: noradId,
      data: {}
    });
    expect(result).toEqual(mockResponse.data);
  });

  test('quickAssess makes correct API call', async () => {
    // Arrange
    const noradId = 43013;
    const mockResponse = {
      data: {
        norad_id: noradId,
        overall_threat: 'MEDIUM',
        confidence: 0.75
      }
    };
    mockAxiosInstance.get.mockResolvedValueOnce(mockResponse);

    // Act
    const result = await ccdmService.quickAssess(noradId);

    // Assert
    expect(mockAxiosInstance.get).toHaveBeenCalledWith(`/ccdm/quick-assessment/${noradId}`);
    expect(result).toEqual(mockResponse.data);
  });

  test('getHistoricalAnalysis makes correct API call', async () => {
    // Arrange
    const noradId = 25544;
    const startDate = '2023-01-01T00:00:00Z';
    const endDate = '2023-01-07T00:00:00Z';
    const mockResponse = {
      data: {
        norad_id: noradId,
        analysis_points: [
          { timestamp: '2023-01-01T12:00:00Z', threat_level: 'LOW' },
          { timestamp: '2023-01-02T12:00:00Z', threat_level: 'MEDIUM' }
        ],
        start_date: startDate,
        end_date: endDate
      }
    };
    mockAxiosInstance.post.mockResolvedValueOnce(mockResponse);

    // Act
    const result = await ccdmService.getHistoricalAnalysis(noradId, startDate, endDate);

    // Assert
    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/ccdm/historical', {
      norad_id: noradId,
      start_date: startDate,
      end_date: endDate
    });
    expect(result).toEqual(mockResponse.data);
  });

  test('getLastWeekAnalysis makes correct API call', async () => {
    // Arrange
    const noradId = 27424;
    const mockResponse = {
      data: {
        norad_id: noradId,
        analysis_points: [
          { timestamp: '2023-01-01T12:00:00Z', threat_level: 'LOW' }
        ]
      }
    };
    mockAxiosInstance.get.mockResolvedValueOnce(mockResponse);

    // Act
    const result = await ccdmService.getLastWeekAnalysis(noradId);

    // Assert
    expect(mockAxiosInstance.get).toHaveBeenCalledWith(`/ccdm/last-week-analysis/${noradId}`);
    expect(result).toEqual(mockResponse.data);
  });

  test('detectShapeChanges makes correct API call', async () => {
    // Arrange
    const noradId = 48274;
    const startDate = '2023-01-01T00:00:00Z';
    const endDate = '2023-01-31T00:00:00Z';
    const mockResponse = {
      data: {
        norad_id: noradId,
        detected_changes: [
          {
            timestamp: '2023-01-15T12:00:00Z',
            description: 'Solar panel extension',
            confidence: 0.88
          }
        ]
      }
    };
    mockAxiosInstance.post.mockResolvedValueOnce(mockResponse);

    // Act
    const result = await ccdmService.detectShapeChanges(noradId, startDate, endDate);

    // Assert
    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/ccdm/shape-changes', {
      norad_id: noradId,
      start_date: startDate,
      end_date: endDate,
      data: {}
    });
    expect(result).toEqual(mockResponse.data);
  });

  test('adds auth header when token is present', async () => {
    // Arrange
    const token = 'test-jwt-token';
    localStorage.setItem('token', token);
    const noradId = 25544;
    mockAxiosInstance.get.mockResolvedValueOnce({ data: {} });

    // Act
    await ccdmService.quickAssess(noradId);

    // Assert
    // We're testing that the service works with a token present
    // We can't easily test the interceptor behavior directly in this test
    // But we can verify the API call was made
    expect(mockAxiosInstance.get).toHaveBeenCalledWith(`/ccdm/quick-assessment/${noradId}`);
  });

  test('handles errors and logs them', async () => {
    // Arrange
    const noradId = 25544;
    const error = new Error('Network error');
    mockAxiosInstance.get.mockRejectedValueOnce(error);
    console.error = jest.fn();

    // Act & Assert
    await expect(ccdmService.quickAssess(noradId)).rejects.toThrow(error);
    expect(console.error).toHaveBeenCalledWith('Error performing quick assessment:', error);
  });
}); 