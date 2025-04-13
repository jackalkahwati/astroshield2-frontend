/**
 * Health check endpoint for the frontend service
 * Used by container orchestration and monitoring systems
 */

// Get package.json version
import pkg from '../../../package.json';

export default function handler(req, res) {
  const startTime = process.hrtime();
  
  // Basic health information
  const healthInfo = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'frontend',
    version: pkg.version,
    environment: process.env.NODE_ENV || 'development',
    uptime: process.uptime(),
    memoryUsage: formatMemoryUsage(process.memoryUsage()),
  };
  
  // Check backend connection if needed
  try {
    // This is a placeholder - in a real app, you might want to do a
    // lightweight check to the backend API
    healthInfo.backendConnection = {
      status: 'unknown', // We're not actually checking in this implementation
    };
  } catch (error) {
    healthInfo.backendConnection = {
      status: 'error',
      message: error.message,
    };
    healthInfo.status = 'degraded';
  }
  
  // Calculate response time
  const hrTime = process.hrtime(startTime);
  healthInfo.responseTimeMs = hrTime[0] * 1000 + hrTime[1] / 1000000;
  
  // Return health information
  res.status(200).json(healthInfo);
}

/**
 * Format memory usage values to be more readable
 */
function formatMemoryUsage(memoryData) {
  const MB = 1024 * 1024;
  return {
    rss: `${Math.round(memoryData.rss / MB)} MB`,
    heapTotal: `${Math.round(memoryData.heapTotal / MB)} MB`,
    heapUsed: `${Math.round(memoryData.heapUsed / MB)} MB`,
    external: `${Math.round(memoryData.external / MB)} MB`,
  };
} 