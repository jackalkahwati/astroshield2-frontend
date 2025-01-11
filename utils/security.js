const crypto = require('crypto');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const rateLimit = require('express-rate-limit');
const csrf = require('csurf');

const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY || crypto.randomBytes(32);
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';
const SALT_ROUNDS = 10;

class Security {
  async hashPassword(password) {
    return bcrypt.hash(password, SALT_ROUNDS);
  }

  async verifyPassword(password, hash) {
    return bcrypt.compare(password, hash);
  }

  generateJWT(userData) {
    return jwt.sign(userData, JWT_SECRET, {
      expiresIn: '24h'
    });
  }

  verifyJWT(token) {
    return jwt.verify(token, JWT_SECRET);
  }

  encryptPII(data) {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-gcm', ENCRYPTION_KEY, iv);
    
    const encrypted = Buffer.concat([
      cipher.update(JSON.stringify(data), 'utf8'),
      cipher.final()
    ]);

    const tag = cipher.getAuthTag();

    return {
      iv: iv.toString('hex'),
      encrypted: encrypted.toString('hex'),
      tag: tag.toString('hex')
    };
  }

  decryptPII(encryptedData) {
    const decipher = crypto.createDecipheriv(
      'aes-256-gcm',
      ENCRYPTION_KEY,
      Buffer.from(encryptedData.iv, 'hex')
    );

    decipher.setAuthTag(Buffer.from(encryptedData.tag, 'hex'));

    const decrypted = Buffer.concat([
      decipher.update(Buffer.from(encryptedData.encrypted, 'hex')),
      decipher.final()
    ]);

    return JSON.parse(decrypted.toString());
  }

  csrfProtection() {
    return csrf({ cookie: true });
  }

  rateLimiter() {
    return rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // limit each IP to 100 requests per windowMs
      message: {
        status: 'error',
        message: 'Too many requests, please try again later.'
      }
    });
  }

  async enforceRetentionPolicy() {
    const retentionPeriod = 365 * 24 * 60 * 60 * 1000; // 1 year in milliseconds
    const cutoffDate = new Date(Date.now() - retentionPeriod);

    const db = require('../test-utils/db').db;
    await db.raw(
      'DELETE FROM test.user_data WHERE created_at < $1',
      [cutoffDate]
    );
  }

  async performSensitiveOperation(userId, action) {
    // Log the operation
    const auditLogger = require('./audit-logger');
    await auditLogger.log({
      user: userId,
      action: action,
      timestamp: new Date().toISOString()
    });

    // Perform the operation
    // ... implement actual operation logic
  }
}

module.exports = new Security(); 