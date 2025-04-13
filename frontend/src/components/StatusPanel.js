import React, { useEffect, useState } from 'react';
import axios from 'axios';

const StatusPanel = () => {
  const [status, setStatus] = useState({
    database: 'checking',
    frontend: 'online',
    backend: 'checking',
    api: 'checking',
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        setLoading(true);
        
        // Call the backend health endpoint
        const response = await axios.get('/api/v1/health');
        
        // Call the system info endpoint
        const systemInfo = await axios.get('/api/v1/system-info');
        
        // Update status based on responses
        setStatus({
          database: response.data.services.database,
          api: response.data.services.api,
          frontend: 'online',
          backend: systemInfo.data.components.backend,
        });
        
        setError(null);
      } catch (err) {
        console.error('Error checking status:', err);
        setError('Failed to connect to the backend server');
        setStatus({
          database: 'disconnected',
          frontend: 'online',
          backend: 'disconnected',
          api: 'disconnected'
        });
      } finally {
        setLoading(false);
      }
    };

    // Check status immediately and then every 30 seconds
    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    
    // Clean up
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'online':
      case 'connected':
      case 'active':
        return 'bg-green-500';
      case 'pending':
      case 'connecting':
      case 'checking':
        return 'bg-yellow-500';
      case 'disconnected':
      case 'offline':
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusLabel = (status) => {
    switch (status) {
      case 'online':
      case 'connected':
      case 'active':
        return 'Operational';
      case 'pending':
      case 'connecting':
      case 'checking':
        return 'Connecting...';
      case 'disconnected':
      case 'offline':
      case 'error':
        return 'Disconnected';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="bg-gray-800 text-white p-4 rounded-lg shadow-lg">
      <h2 className="text-xl font-bold mb-4">System Status</h2>
      
      {loading && <p className="text-gray-400 mb-2">Checking system status...</p>}
      
      {error && (
        <div className="bg-red-900 p-3 rounded mb-4">
          <p className="text-white">{error}</p>
        </div>
      )}
      
      <div className="space-y-2">
        {Object.entries(status).map(([key, value]) => (
          <div key={key} className="flex items-center justify-between">
            <span className="capitalize">{key}</span>
            <div className="flex items-center">
              <span className="mr-2">{getStatusLabel(value)}</span>
              <span className={`w-3 h-3 rounded-full ${getStatusColor(value)}`}></span>
            </div>
          </div>
        ))}
      </div>
      
      {status.database === 'connected' && (
        <div className="mt-4 text-green-400">
          <p>✓ Database connection successful</p>
        </div>
      )}
      
      {status.database === 'pending' && (
        <div className="mt-4 text-yellow-400">
          <p>! Database setup in progress</p>
          <p className="text-xs mt-1">Run setup_database.sh to initialize</p>
        </div>
      )}

      <div className="mt-6 text-xs text-gray-400">
        Last checked: {new Date().toLocaleTimeString()}
      </div>
    </div>
  );
};

export default StatusPanel; 