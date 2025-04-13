import React, { useState, useEffect } from 'react';
import ccdmService from '../../services/ccdmService';

const ShapeChangeDetection = ({ noradId }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [shapeChanges, setShapeChanges] = useState(null);
  const [selectedChange, setSelectedChange] = useState(null);

  useEffect(() => {
    // Skip if no NORAD ID is provided
    if (!noradId) {
      setLoading(false);
      return;
    }

    const fetchShapeChanges = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Get dates for last 30 days
        const endDate = new Date();
        const startDate = new Date();
        startDate.setDate(endDate.getDate() - 30);
        
        const data = await ccdmService.detectShapeChanges(
          noradId,
          startDate.toISOString(),
          endDate.toISOString()
        );
        
        setShapeChanges(data);
        
        // Select the first change by default if available
        if (data.detected_changes && data.detected_changes.length > 0) {
          setSelectedChange(data.detected_changes[0]);
        }
      } catch (err) {
        console.error('Error fetching shape changes:', err);
        setError('Failed to fetch shape change data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchShapeChanges();
  }, [noradId]);

  if (loading) {
    return (
      <div className="flex justify-center items-center p-8">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">Error:</strong>
        <span className="block sm:inline"> {error}</span>
      </div>
    );
  }

  if (!noradId) {
    return (
      <div className="bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded">
        Please select a satellite to view shape change detection.
      </div>
    );
  }

  if (!shapeChanges || !shapeChanges.detected_changes || shapeChanges.detected_changes.length === 0) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded">
        No shape change data available.
      </div>
    );
  }

  // Format date for display
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Get significance level color
  const getSignificanceColor = (significance) => {
    if (significance < 0.3) return '#10B981'; // green (low significance)
    if (significance < 0.6) return '#F59E0B'; // amber (medium significance)
    return '#EF4444'; // red (high significance)
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Shape Change Detection</h2>
      
      <div className="mb-4 p-4 bg-gray-50 rounded-md">
        <div className="flex justify-between items-center">
          <span className="text-gray-600">NORAD ID:</span>
          <span className="font-mono font-bold">{shapeChanges.norad_id}</span>
        </div>
      </div>

      <div className="mb-6">
        <h3 className="font-bold text-lg mb-2 text-gray-700">Summary</h3>
        <p className="text-gray-600">{shapeChanges.summary}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="md:col-span-1">
          <h3 className="font-bold text-lg mb-3 text-gray-700">Detected Changes</h3>
          <div className="space-y-2 max-h-80 overflow-y-auto pr-2">
            {shapeChanges.detected_changes.map((change, index) => (
              <div
                key={index}
                className={`p-3 border rounded-md cursor-pointer hover:bg-gray-50 transition ${
                  selectedChange === change ? 'border-blue-500 bg-blue-50' : ''
                }`}
                onClick={() => setSelectedChange(change)}
              >
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">
                    {formatDate(change.timestamp)}
                  </span>
                  <span 
                    className="inline-block w-3 h-3 rounded-full" 
                    style={{ backgroundColor: getSignificanceColor(change.significance) }}
                    title={`Significance: ${Math.round(change.significance * 100)}%`}
                  ></span>
                </div>
                <div className="mt-1 text-xs text-gray-600 truncate">
                  {change.description}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="md:col-span-2">
          {selectedChange && (
            <div className="border rounded-lg p-4">
              <h3 className="font-bold text-lg mb-3 text-gray-700">Change Details</h3>
              
              <div className="mb-4">
                <span className="text-sm text-gray-500">Detected on:</span>
                <span className="block font-medium">{formatDate(selectedChange.timestamp)}</span>
              </div>
              
              <div className="mb-4">
                <span className="text-sm text-gray-500">Description:</span>
                <p className="mt-1">{selectedChange.description}</p>
              </div>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                <div className="border rounded p-3 bg-gray-50">
                  <div className="text-sm font-medium mb-2 text-gray-700">Before Change</div>
                  <div className="p-4 bg-white border rounded flex items-center justify-center h-40">
                    {/* Placeholder for shape visualization - in a real app, this would be a 3D model or image */}
                    <div className="text-center">
                      <div className="inline-block p-8 border-2 border-dashed border-gray-300 rounded-md">
                        <div className="text-gray-400 uppercase text-xs mb-2">Configuration</div>
                        <div className="font-mono text-sm">{selectedChange.before_shape}</div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="border rounded p-3 bg-gray-50">
                  <div className="text-sm font-medium mb-2 text-gray-700">After Change</div>
                  <div className="p-4 bg-white border rounded flex items-center justify-center h-40">
                    {/* Placeholder for shape visualization - in a real app, this would be a 3D model or image */}
                    <div className="text-center">
                      <div className="inline-block p-8 border-2 border-dashed border-gray-300 rounded-md">
                        <div className="text-gray-400 uppercase text-xs mb-2">Configuration</div>
                        <div className="font-mono text-sm">{selectedChange.after_shape}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mb-4">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-gray-500">Confidence:</span>
                  <span className="text-sm font-medium">{Math.round(selectedChange.confidence * 100)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${Math.round(selectedChange.confidence * 100)}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="mb-4">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-gray-500">Significance:</span>
                  <span 
                    className="text-sm font-medium"
                    style={{ color: getSignificanceColor(selectedChange.significance) }}
                  >
                    {Math.round(selectedChange.significance * 100)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="h-2 rounded-full"
                    style={{ 
                      width: `${Math.round(selectedChange.significance * 100)}%`,
                      backgroundColor: getSignificanceColor(selectedChange.significance)
                    }}
                  ></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {shapeChanges.metadata && Object.keys(shapeChanges.metadata).length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <h3 className="font-bold text-sm mb-2 text-gray-700">Metadata</h3>
          <div className="text-xs text-gray-500">
            {Object.entries(shapeChanges.metadata).map(([key, value]) => (
              <div key={key} className="mb-1">
                <span className="font-medium capitalize">{key.replace('_', ' ')}:</span>{' '}
                <span>{typeof value === 'object' ? JSON.stringify(value) : value}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ShapeChangeDetection; 