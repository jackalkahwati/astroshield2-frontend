import React, { useState, useEffect } from 'react';
import ccdmService from '../../services/ccdmService';

const ThreatAssessment = ({ noradId }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [assessment, setAssessment] = useState(null);

  useEffect(() => {
    // Skip if no NORAD ID is provided
    if (!noradId) {
      setLoading(false);
      return;
    }

    const fetchThreatAssessment = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const data = await ccdmService.quickAssess(noradId);
        setAssessment(data);
      } catch (err) {
        console.error('Error fetching threat assessment:', err);
        setError('Failed to fetch threat assessment data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchThreatAssessment();
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
        Please select a satellite to view threat assessment.
      </div>
    );
  }

  if (!assessment) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded">
        No threat assessment data available.
      </div>
    );
  }

  // Get the appropriate color based on threat level
  const getThreatLevelColor = (level) => {
    switch (level) {
      case 'NONE': return 'green';
      case 'LOW': return 'blue';
      case 'MEDIUM': return 'yellow';
      case 'HIGH': return 'orange';
      case 'CRITICAL': return 'red';
      default: return 'gray';
    }
  };

  const threatColor = getThreatLevelColor(assessment.overall_threat);

  return (
    <div className="bg-white shadow-md rounded-lg p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Threat Assessment</h2>
      
      <div className="mb-4 p-4 bg-gray-50 rounded-md">
        <div className="flex justify-between items-center">
          <span className="text-gray-600">NORAD ID:</span>
          <span className="font-mono font-bold">{assessment.norad_id}</span>
        </div>
        <div className="flex justify-between items-center mt-2">
          <span className="text-gray-600">Timestamp:</span>
          <span>{new Date(assessment.timestamp).toLocaleString()}</span>
        </div>
      </div>

      <div className="mb-6 text-center">
        <h3 className="font-bold text-lg mb-3 text-gray-700">Overall Threat Level</h3>
        <div className={`inline-block px-6 py-3 rounded-md bg-${threatColor}-100 border border-${threatColor}-300`}>
          <div className="flex items-center">
            <div className={`w-4 h-4 rounded-full bg-${threatColor}-500 mr-2`}></div>
            <span className={`text-${threatColor}-800 font-bold text-xl`}>
              {assessment.overall_threat}
            </span>
          </div>
          <div className="mt-2 text-sm text-gray-600">
            Confidence: {Math.round(assessment.confidence * 100)}%
          </div>
        </div>
      </div>

      <div className="mb-6">
        <h3 className="font-bold text-lg mb-3 text-gray-700">Threat Components</h3>
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(assessment.threat_components).map(([key, value]) => (
            <div key={key} className="p-3 border rounded-md">
              <div className="text-sm text-gray-600 capitalize mb-1">{key.replace('_', ' ')}</div>
              <div className="flex items-center">
                <div className={`w-3 h-3 rounded-full bg-${getThreatLevelColor(value)}-500 mr-2`}></div>
                <span className={`text-${getThreatLevelColor(value)}-700 font-semibold`}>
                  {value}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {assessment.recommendations && assessment.recommendations.length > 0 && (
        <div className="mb-6">
          <h3 className="font-bold text-lg mb-3 text-gray-700">Recommendations</h3>
          <ul className="list-disc pl-5 space-y-1 text-gray-600">
            {assessment.recommendations.map((recommendation, index) => (
              <li key={index}>{recommendation}</li>
            ))}
          </ul>
        </div>
      )}

      {assessment.metadata && Object.keys(assessment.metadata).length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <h3 className="font-bold text-sm mb-2 text-gray-700">Metadata</h3>
          <div className="text-xs text-gray-500">
            {Object.entries(assessment.metadata).map(([key, value]) => (
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

export default ThreatAssessment; 