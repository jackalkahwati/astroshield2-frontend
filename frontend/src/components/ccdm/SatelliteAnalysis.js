import React, { useState, useEffect } from 'react';
import ccdmService from '../../services/ccdmService';

const SatelliteAnalysis = ({ noradId }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [analysis, setAnalysis] = useState(null);

  useEffect(() => {
    // Skip if no NORAD ID is provided
    if (!noradId) {
      setLoading(false);
      return;
    }

    const fetchAnalysis = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const data = await ccdmService.analyzeObject(noradId);
        setAnalysis(data);
      } catch (err) {
        console.error('Error fetching satellite analysis:', err);
        setError('Failed to fetch analysis data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysis();
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
        Please select a satellite to analyze.
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded">
        No analysis data available.
      </div>
    );
  }

  return (
    <div className="bg-white shadow-md rounded-lg p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Satellite Analysis</h2>
      <div className="mb-4 p-4 bg-gray-50 rounded-md">
        <div className="flex justify-between items-center">
          <span className="text-gray-600">NORAD ID:</span>
          <span className="font-mono font-bold">{analysis.norad_id}</span>
        </div>
        <div className="flex justify-between items-center mt-2">
          <span className="text-gray-600">Timestamp:</span>
          <span>{new Date(analysis.timestamp).toLocaleString()}</span>
        </div>
      </div>

      <div className="mb-6">
        <h3 className="font-bold text-lg mb-2 text-gray-700">Analysis Summary</h3>
        <p className="text-gray-600">{analysis.summary}</p>
      </div>

      <div className="mb-6">
        <h3 className="font-bold text-lg mb-2 text-gray-700">Analysis Results</h3>
        <div className="space-y-4">
          {analysis.analysis_results.map((result, index) => (
            <div key={index} className="p-4 border rounded-md">
              <div className="flex justify-between items-center">
                <div className="flex items-center">
                  <span className="text-gray-600 mr-2">Threat Level:</span>
                  <ThreatLevelBadge level={result.threat_level} />
                </div>
                <span className="text-sm text-gray-500">
                  {new Date(result.timestamp).toLocaleString()}
                </span>
              </div>
              <div className="mt-2">
                <span className="text-gray-600">Confidence:</span>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
                  <div
                    className="bg-blue-600 h-2.5 rounded-full"
                    style={{ width: `${Math.round(result.confidence * 100)}%` }}
                  ></div>
                </div>
                <span className="text-xs text-gray-500">
                  {Math.round(result.confidence * 100)}%
                </span>
              </div>
              {result.details && Object.keys(result.details).length > 0 && (
                <div className="mt-3 pt-3 border-t">
                  <h4 className="text-sm font-semibold text-gray-700 mb-1">Details</h4>
                  <div className="text-sm text-gray-600">
                    {Object.entries(result.details).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="capitalize">{key.replace('_', ' ')}:</span>
                        <span className="font-mono">
                          {typeof value === 'object' ? JSON.stringify(value) : value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {analysis.metadata && Object.keys(analysis.metadata).length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <h3 className="font-bold text-sm mb-2 text-gray-700">Metadata</h3>
          <div className="text-xs text-gray-500">
            {Object.entries(analysis.metadata).map(([key, value]) => (
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

// Helper component for threat level badges
const ThreatLevelBadge = ({ level }) => {
  let bgColor, textColor;
  
  switch (level) {
    case 'NONE':
      bgColor = 'bg-green-100';
      textColor = 'text-green-800';
      break;
    case 'LOW':
      bgColor = 'bg-blue-100';
      textColor = 'text-blue-800';
      break;
    case 'MEDIUM':
      bgColor = 'bg-yellow-100';
      textColor = 'text-yellow-800';
      break;
    case 'HIGH':
      bgColor = 'bg-orange-100';
      textColor = 'text-orange-800';
      break;
    case 'CRITICAL':
      bgColor = 'bg-red-100';
      textColor = 'text-red-800';
      break;
    default:
      bgColor = 'bg-gray-100';
      textColor = 'text-gray-800';
  }

  return (
    <span className={`px-2 py-1 rounded text-xs font-semibold ${bgColor} ${textColor}`}>
      {level}
    </span>
  );
};

export default SatelliteAnalysis; 