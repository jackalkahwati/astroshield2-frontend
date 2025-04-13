import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import StatusPanel from '../StatusPanel';
import axios from 'axios';

// Mock axios
jest.mock('axios');

describe('StatusPanel Component', () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
  });

  test('renders loading state initially', () => {
    render(<StatusPanel />);
    expect(screen.getByText('Checking system status...')).toBeInTheDocument();
  });

  test('displays error when API call fails', async () => {
    // Mock axios to reject
    axios.get.mockRejectedValueOnce(new Error('Failed to fetch'));

    render(<StatusPanel />);

    // Wait for the component to finish loading
    await waitFor(() => {
      expect(screen.getByText('Failed to connect to the backend server')).toBeInTheDocument();
    });

    // Verify that status indicators show disconnected state
    expect(screen.getByText('database')).toBeInTheDocument();
    expect(screen.getAllByText('Disconnected').length).toBeGreaterThan(0);
  });

  test('displays proper status when API call succeeds', async () => {
    // Mock successful API responses
    axios.get.mockImplementation((url) => {
      if (url === '/api/v1/health') {
        return Promise.resolve({
          data: {
            status: 'healthy',
            services: {
              database: 'connected',
              api: 'online'
            }
          }
        });
      }
      if (url === '/api/v1/system-info') {
        return Promise.resolve({
          data: {
            components: {
              backend: 'online',
              frontend: 'online'
            }
          }
        });
      }
      return Promise.reject(new Error('Unexpected URL'));
    });

    render(<StatusPanel />);

    // Wait for the component to finish loading and call APIs
    await waitFor(() => {
      expect(screen.queryByText('Checking system status...')).not.toBeInTheDocument();
    });

    // Check that database status is shown correctly
    await waitFor(() => {
      expect(screen.getByText('database')).toBeInTheDocument();
      expect(screen.getAllByText('Operational').length).toBeGreaterThan(0);
    });

    // Verify that multiple API calls were made
    expect(axios.get).toHaveBeenCalledWith('/api/v1/health');
    expect(axios.get).toHaveBeenCalledWith('/api/v1/system-info');
  });

  test('displays database connected message when database is connected', async () => {
    // Mock successful API responses with connected database
    axios.get.mockImplementation((url) => {
      if (url === '/api/v1/health') {
        return Promise.resolve({
          data: {
            status: 'healthy',
            services: {
              database: 'connected',
              api: 'online'
            }
          }
        });
      }
      if (url === '/api/v1/system-info') {
        return Promise.resolve({
          data: {
            components: {
              backend: 'online'
            }
          }
        });
      }
      return Promise.reject(new Error('Unexpected URL'));
    });

    render(<StatusPanel />);

    // Wait for the component to finish loading
    await waitFor(() => {
      expect(screen.queryByText('Checking system status...')).not.toBeInTheDocument();
    });

    // Check for the database connection success message
    await waitFor(() => {
      expect(screen.getByText('✓ Database connection successful')).toBeInTheDocument();
    });
  });

  test('displays pending database message when database is pending', async () => {
    // Mock successful API responses with pending database
    axios.get.mockImplementation((url) => {
      if (url === '/api/v1/health') {
        return Promise.resolve({
          data: {
            status: 'healthy',
            services: {
              database: 'pending',
              api: 'online'
            }
          }
        });
      }
      if (url === '/api/v1/system-info') {
        return Promise.resolve({
          data: {
            components: {
              backend: 'online'
            }
          }
        });
      }
      return Promise.reject(new Error('Unexpected URL'));
    });

    render(<StatusPanel />);

    // Wait for the component to finish loading
    await waitFor(() => {
      expect(screen.queryByText('Checking system status...')).not.toBeInTheDocument();
    });

    // Check for the database pending message
    await waitFor(() => {
      expect(screen.getByText('! Database setup in progress')).toBeInTheDocument();
      expect(screen.getByText('Run setup_database.sh to initialize')).toBeInTheDocument();
    });
  });
}); 