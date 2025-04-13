import React, { useState, useEffect } from 'react';
import ccdmService from '../../services/ccdmService';

const HistoricalAnalysis = ({ noradId, customDateRange }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [historical, setHistorical] = useState(null);
  const [activeTab, setActiveTab] = useState('chart');

  useEffect(() => {
    // Skip if no NORAD ID is provided
    if (!noradId) {
      setLoading(false);
      return;
    }

    const fetchHistoricalData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        let data;
        if (customDateRange) {
          // If custom date range provided
          data = await ccdmService.getHistoricalAnalysis(
            noradId, 
            customDateRange.startDate, 
            customDateRange.endDate
          );
        } else {
          // Default to last week
          data = await ccdmService.getLastWeekAnalysis(noradId);
        }
        
        setHistorical(data);
      } catch (err) {
        console.error('Error fetching historical analysis:', err);
        setError('Failed to fetch historical data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchHistoricalData();
  }, [noradId, customDateRange]);

  if (loading) {
    return (
      <div className="flex justify-center items-center p-8">
        <div data-testid="loading-spinner" className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
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
        Please select a satellite to view historical analysis.
      </div>
    );
  }

  if (!historical || !historical.analysis_points || historical.analysis_points.length === 0) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded">
        No historical analysis data available.
      </div>
    );
  }

  // Sort analysis points by timestamp
  const sortedPoints = [...historical.analysis_points].sort(
    (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
  );

  // Get threat level color
  const getThreatColor = (level) => {
    switch (level) {
      case 'NONE': return '#10B981'; // green-500
      case 'LOW': return '#3B82F6'; // blue-500
      case 'MEDIUM': return '#F59E0B'; // amber-500
      case 'HIGH': return '#F97316'; // orange-500
      case 'CRITICAL': return '#EF4444'; // red-500
      default: return '#6B7280'; // gray-500
    }
  };

  // Format date for display
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  // Format time for display
  const formatTime = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Historical Analysis</h2>
      
      <div className="mb-4 p-4 bg-gray-50 rounded-md">
        <div className="flex justify-between items-center">
          <span className="text-gray-600">NORAD ID:</span>
          <span className="font-mono font-bold">{historical.norad_id}</span>
        </div>
        <div className="flex justify-between items-center mt-2">
          <span className="text-gray-600">Period:</span>
          <span>{formatDate(historical.start_date)} - {formatDate(historical.end_date)}</span>
        </div>
      </div>

      <div className="mb-4">
        <h3 className="font-bold text-lg mb-2 text-gray-700">Trend Summary</h3>
        <p className="text-gray-600">{historical.trend_summary}</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-4">
        <nav className="flex space-x-8">
          <button
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'chart'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
            onClick={() => setActiveTab('chart')}
          >
            Timeline Chart
          </button>
          <button
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'table'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
            onClick={() => setActiveTab('table')}
          >
            Data Table
          </button>
        </nav>
      </div>

      {/* Chart View */}
      {activeTab === 'chart' && (
        <div className="mb-6">
          <div className="h-64 relative">
            {/* Simple timeline visualization */}
            <div className="absolute left-0 right-0 top-0 bottom-0 flex items-end">
              {sortedPoints.map((point, index) => {
                // Calculate height based on threat level
                const threatLevels = ['NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
                const threatIndex = threatLevels.indexOf(point.threat_level);
                const height = 20 + (threatIndex * 20); // 20% height for NONE, up to 100% for CRITICAL
                
                // Calculate width of each bar
                const barWidth = `calc(${100 / sortedPoints.length}% - 8px)`;
                
                return (
                  <div 
                    key={index}
                    className="mx-1 rounded-t-sm cursor-pointer relative group"
                    style={{ 
                      height: `${height}%`, 
                      width: barWidth,
                      backgroundColor: getThreatColor(point.threat_level)
                    }}
                    title={`${point.threat_level} - ${formatDate(point.timestamp)}`}
                  >
                    {/* Tooltip */}
                    <div className="hidden group-hover:block absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white text-xs rounded py-1 px-2 whitespace-nowrap z-10">
                      <div>{formatDate(point.timestamp)} {formatTime(point.timestamp)}</div>
                      <div className="font-bold">{point.threat_level}</div>
                      <div>Confidence: {Math.round(point.confidence * 100)}%</div>
                    </div>
                  </div>
                );
              })}
            </div>
            
            {/* X-Axis: Dates */}
            <div className="absolute left-0 right-0 bottom-0 flex justify-between text-xs text-gray-500 pt-2 border-t">
              <span>{formatDate(historical.start_date)}</span>
              <span>{formatDate(historical.end_date)}</span>
            </div>
            
            {/* Y-Axis: Threat Levels */}
            <div className="absolute left-0 top-0 bottom-8 flex flex-col justify-between text-xs text-gray-500 pr-2">
              <span>CRITICAL</span>
              <span>HIGH</span>
              <span>MEDIUM</span>
              <span>LOW</span>
              <span>NONE</span>
            </div>
          </div>
          
          {/* Legend */}
          <div className="mt-8 flex flex-wrap justify-center gap-4">
            {['NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].map(level => (
              <div key={level} className="flex items-center">
                <div 
                  className="w-4 h-4 mr-1 rounded" 
                  style={{ backgroundColor: getThreatColor(level) }}
                ></div>
                <span className="text-xs text-gray-600">{level}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Table View */}
      {activeTab === 'table' && (
        <div className="overflow-x-auto mb-6">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date & Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Threat Level
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Details
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {sortedPoints.map((point, index) => (
                <tr key={index} className={index % 2 ? 'bg-gray-50' : ''}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatDate(point.timestamp)} {formatTime(point.timestamp)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span 
                      className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full" 
                      style={{ 
                        backgroundColor: `${getThreatColor(point.threat_level)}20`, // 20% opacity
                        color: getThreatColor(point.threat_level) 
                      }}
                    >
                      {point.threat_level}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {Math.round(point.confidence * 100)}%
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500 max-w-xs truncate">
                    {point.details && Object.keys(point.details).length > 0 ? (
                      <details className="cursor-pointer">
                        <summary>View Details</summary>
                        <div className="mt-2 text-xs space-y-1 pl-2 border-l-2 border-gray-200">
                          {Object.entries(point.details).map(([key, value]) => (
                            <div key={key}>
                              <span className="font-medium capitalize">{key.replace('_', ' ')}:</span>{' '}
                              <span>{typeof value === 'object' ? JSON.stringify(value) : value}</span>
                            </div>
                          ))}
                        </div>
                      </details>
                    ) : (
                      <span className="text-gray-400">No details</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {historical.metadata && Object.keys(historical.metadata).length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <h3 className="font-bold text-sm mb-2 text-gray-700">Metadata</h3>
          <div className="text-xs text-gray-500">
            {Object.entries(historical.metadata).map(([key, value]) => (
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

export default HistoricalAnalysis; 