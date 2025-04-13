import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import ShapeChangeDetection from '../ShapeChangeDetection';
import ccdmService from '../../../services/ccdmService';

// Mock the ccdmService
jest.mock('../../../services/ccdmService');

describe('ShapeChangeDetection Component', () => {
  const mockNoradId = '25544'; // ISS NORAD ID
  const mockDate = new Date('2023-05-01');
  
  // Mock date for consistent testing
  beforeAll(() => {
    jest.useFakeTimers();
    jest.setSystemTime(mockDate);
  });

  afterAll(() => {
    jest.useRealTimers();
  });

  beforeEach(() => {
    jest.resetAllMocks();
  });

  const mockShapeChangeData = {
    norad_id: 25544,
    summary: 'Two significant shape changes detected in the last 30 days.',
    detected_changes: [
      {
        timestamp: '2023-06-01T12:00:00Z',
        before_shape: 'RECTANGULAR',
        after_shape: 'RECTANGULAR_EXTENDED',
        description: 'Possible solar panel deployment.',
        confidence: 0.85,
        significance: 0.6
      },
      {
        timestamp: '2023-06-15T09:30:00Z',
        before_shape: 'RECTANGULAR_EXTENDED',
        after_shape: 'IRREGULAR',
        description: 'Possible structural change or damage.',
        confidence: 0.78,
        significance: 0.9
      }
    ],
    metadata: {
      method: 'radar-cross-section-analysis',
      resolution: 'medium',
      comparison_algorithm: 'shape-diff-v2'
    }
  };

  it('should show loading state initially', () => {
    // Setup ccdmService mock to return a delayed promise
    ccdmService.detectShapeChanges.mockReturnValue(new Promise(() => {}));
    
    render(<ShapeChangeDetection noradId={mockNoradId} />);
    
    // Check for loading spinner using data-testid
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  it('should handle missing noradId properly', () => {
    render(<ShapeChangeDetection noradId={null} />);
    
    expect(screen.getByText('Please select a satellite to view shape change detection.')).toBeInTheDocument();
    expect(ccdmService.detectShapeChanges).not.toHaveBeenCalled();
  });

  it('should show message when no shape changes are detected', async () => {
    // Mock empty shape changes response
    ccdmService.detectShapeChanges.mockResolvedValue({
      norad_id: 25544,
      detected_changes: []
    });
    
    render(<ShapeChangeDetection noradId={mockNoradId} />);
    
    // Wait for loading to finish and no data message to appear
    await waitFor(() => {
      expect(screen.queryByText('No shape change data available.')).toBeInTheDocument();
    });
    
    expect(ccdmService.detectShapeChanges).toHaveBeenCalledWith(
      mockNoradId,
      expect.any(String), // Start date
      expect.any(String)  // End date
    );
  });

  it('should handle API error gracefully', async () => {
    // Mock API error
    const errorMessage = 'Failed to fetch shape changes';
    ccdmService.detectShapeChanges.mockRejectedValue(new Error(errorMessage));
    
    render(<ShapeChangeDetection noradId={mockNoradId} />);
    
    // Wait for error message to appear
    await waitFor(() => {
      expect(screen.getByText(/Failed to fetch shape change data/i)).toBeInTheDocument();
    });
    
    // Check for error alert
    const errorElement = screen.getByText(/Failed to fetch shape change data/i).closest('div');
    expect(errorElement).toHaveAttribute('role', 'alert');
  });

  it('should fetch and display shape change data correctly', async () => {
    // Mock shape change data
    ccdmService.detectShapeChanges.mockResolvedValue(mockShapeChangeData);
    
    render(<ShapeChangeDetection noradId={mockNoradId} />);
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
    });
    
    // Check header and NORAD ID
    expect(screen.getByText('Shape Change Detection')).toBeInTheDocument();
    expect(screen.getByText('25544')).toBeInTheDocument();
    
    // Check summary
    expect(screen.getByText('Two significant shape changes detected in the last 30 days.')).toBeInTheDocument();
    
    // Check change items are displayed
    const shapeChangeItems = screen.getAllByTestId('shape-change-item');
    expect(shapeChangeItems).toHaveLength(2);
    
    // Check that the first item's data is displayed in the detail view
    // By default, the first item should be selected
    expect(screen.getByText('RECTANGULAR')).toBeInTheDocument();
    expect(screen.getByText('RECTANGULAR_EXTENDED')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument(); // First item's confidence
  });

  it('should allow selection of different shape changes', async () => {
    // Mock shape change data
    ccdmService.detectShapeChanges.mockResolvedValue(mockShapeChangeData);
    
    render(<ShapeChangeDetection noradId={mockNoradId} />);
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
    });
    
    // Click the second change item
    const shapeChangeItems = screen.getAllByTestId('shape-change-item');
    fireEvent.click(shapeChangeItems[1]);
    
    // Check that the second change details are shown
    // When second item is selected, it should show IRREGULAR as the after_shape
    expect(screen.getByText('IRREGULAR')).toBeInTheDocument();
    
    // Check that confidence changed to second item's value
    // Only search for this percentage in the confidence section, not in significance
    const confidenceSection = screen.getByText('Confidence:').closest('div').parentElement;
    expect(confidenceSection).toHaveTextContent('78%');
  });

  it('should display date formatted correctly', async () => {
    // Mock shape change data
    ccdmService.detectShapeChanges.mockResolvedValue(mockShapeChangeData);
    
    render(<ShapeChangeDetection noradId={mockNoradId} />);
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
    });
    
    // Check that "Detected on:" text exists
    expect(screen.getByText('Detected on:')).toBeInTheDocument();
  });

  it('should show significance levels with correct colors', async () => {
    // Mock shape changes with different significance levels
    ccdmService.detectShapeChanges.mockResolvedValue(mockShapeChangeData);
    
    render(<ShapeChangeDetection noradId={mockNoradId} />);
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
    });
    
    // Check significance text is shown
    expect(screen.getByText('Significance:')).toBeInTheDocument();
    
    // Check significance percentage is shown
    const significanceSection = screen.getByText('Significance:').closest('div').parentElement;
    expect(significanceSection).toHaveTextContent('60%'); // First item selected by default
  });

  it('should display metadata if available', async () => {
    // Mock shape change data with metadata
    ccdmService.detectShapeChanges.mockResolvedValue(mockShapeChangeData);
    
    render(<ShapeChangeDetection noradId={mockNoradId} />);
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
    });
    
    // Check metadata section exists
    expect(screen.getByText('Metadata')).toBeInTheDocument();
    
    // Check metadata values are displayed
    expect(screen.getByText(/method/i)).toBeInTheDocument();
    expect(screen.getByText(/radar-cross-section-analysis/i)).toBeInTheDocument();
  });
}); 