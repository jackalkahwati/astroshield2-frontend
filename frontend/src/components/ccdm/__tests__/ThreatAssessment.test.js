import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ThreatAssessment from '../ThreatAssessment';
import ccdmService from '../../../services/ccdmService';

// Mock the ccdmService
jest.mock('../../../services/ccdmService');

describe('ThreatAssessment Component', () => {
  const mockThreatAssessment = {
    norad_id: 25544,
    timestamp: '2023-04-15T12:30:45Z',
    overall_threat: 'MEDIUM',
    confidence: 0.85,
    threat_components: {
      collision: 'LOW',
      debris: 'MEDIUM',
      maneuver: 'HIGH',
      radiation: 'NONE'
    },
    recommendations: [
      'Monitor for orbital changes',
      'Track nearby space debris'
    ],
    metadata: {
      satellite_name: 'ISS',
      orbit_type: 'LEO'
    }
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders loading state initially', () => {
    ccdmService.quickAssess.mockResolvedValueOnce(mockThreatAssessment);
    render(<ThreatAssessment noradId={25544} />);
    expect(screen.getByRole('status')).toBeInTheDocument(); // assuming the loading spinner has role='status'
  });

  test('displays message when no noradId is provided', () => {
    render(<ThreatAssessment noradId={null} />);
    expect(screen.getByText('Please select a satellite to view threat assessment.')).toBeInTheDocument();
  });

  test('fetches and displays threat assessment data', async () => {
    ccdmService.quickAssess.mockResolvedValueOnce(mockThreatAssessment);
    
    render(<ThreatAssessment noradId={25544} />);
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('Threat Assessment')).toBeInTheDocument();
    });
    
    // Check that the service was called with the correct noradId
    expect(ccdmService.quickAssess).toHaveBeenCalledWith(25544);
    
    // Verify that key data is displayed
    expect(screen.getByText('25544')).toBeInTheDocument(); // NORAD ID
    
    // Use a more specific query for MEDIUM since it appears multiple times
    const threatLevelElement = screen.getAllByText('MEDIUM')[0]; // Get the first MEDIUM element (overall threat)
    expect(threatLevelElement).toBeInTheDocument();
    
    // Fix the confidence text to match what's actually rendered
    expect(screen.getByText('Confidence: 85%')).toBeInTheDocument(); // Confidence
    
    // Check that threat components are shown
    expect(screen.getByText('collision')).toBeInTheDocument();
    expect(screen.getByText('debris')).toBeInTheDocument();
    expect(screen.getByText('maneuver')).toBeInTheDocument();
    expect(screen.getByText('radiation')).toBeInTheDocument();
    
    // Check recommendations
    expect(screen.getByText('Monitor for orbital changes')).toBeInTheDocument();
    expect(screen.getByText('Track nearby space debris')).toBeInTheDocument();
    
    // Check metadata
    expect(screen.getByText(/satellite name/i)).toBeInTheDocument();
    expect(screen.getByText('ISS')).toBeInTheDocument();
  });

  test('shows error message when API call fails', async () => {
    // Mock a rejected promise
    ccdmService.quickAssess.mockRejectedValueOnce(new Error('API Error'));
    
    render(<ThreatAssessment noradId={25544} />);
    
    await waitFor(() => {
      expect(screen.getByText(/Failed to fetch threat assessment data/i)).toBeInTheDocument();
    });
  });
  
  test('displays message when no assessment data is available', async () => {
    ccdmService.quickAssess.mockResolvedValueOnce(null);
    
    render(<ThreatAssessment noradId={25544} />);
    
    await waitFor(() => {
      expect(screen.getByText('No threat assessment data available.')).toBeInTheDocument();
    });
  });
}); 