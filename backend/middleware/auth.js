const jwt = require('jsonwebtoken');
const config = require('../config/vantiq');

// Middleware to authenticate Vantiq requests
const authenticateVantiq = (req, res, next) => {
  // Get token from header
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized - No token provided' });
  }
  
  // Extract token
  const token = authHeader.split(' ')[1];
  
  try {
    // Verify token
    const decoded = jwt.verify(token, config.vantiq.sharedSecret);
    
    // Check if the token is from Vantiq
    if (decoded.issuer !== 'vantiq') {
      return res.status(401).json({ error: 'Unauthorized - Invalid token issuer' });
    }
    
    // Add Vantiq identity to request
    req.vantiq = {
      id: decoded.id,
      namespace: decoded.namespace,
      permissions: decoded.permissions || []
    };
    
    next();
  } catch (error) {
    console.error('Token verification failed:', error.message);
    return res.status(401).json({ error: 'Unauthorized - Invalid token' });
  }
};

module.exports = { authenticateVantiq }; 