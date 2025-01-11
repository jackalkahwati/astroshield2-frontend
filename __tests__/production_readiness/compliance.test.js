const axios = require('axios');
const crypto = require('crypto');
const { validateSchema } = require('../../utils/schema');
const security = require('../../utils/security');

describe('Compliance & Security', () => {
  describe('Data Privacy', () => {
    test('PII is properly encrypted', () => {
      const pii = {
        name: 'John Doe',
        email: 'john@example.com',
        ssn: '123-45-6789'
      };
      
      const encrypted = security.encryptPII(pii);
      expect(encrypted).not.toContain(pii.name);
      expect(encrypted).not.toContain(pii.email);
      expect(encrypted).not.toContain(pii.ssn);
      
      // Verify decryption works
      const decrypted = security.decryptPII(encrypted);
      expect(decrypted).toEqual(pii);
    });

    test('data retention policies are enforced', async () => {
      const db = require('../test-utils/db');
      const oldDate = new Date();
      oldDate.setFullYear(oldDate.getFullYear() - 2);
      
      // Add old data
      await db.userData.insert({
        id: 'test-user',
        data: 'sensitive info',
        created_at: oldDate
      });
      
      // Run retention policy
      await security.enforceRetentionPolicy();
      
      // Verify old data is removed
      const record = await db.userData.findById('test-user');
      expect(record).toBeNull();
    });
  });

  describe('Authentication & Authorization', () => {
    test('password hashing meets security standards', async () => {
      const password = 'TestPassword123!';
      const hashedPassword = await security.hashPassword(password);
      
      // Check hash strength
      expect(hashedPassword).toMatch(/^\$2[ayb]\$.{56}$/); // bcrypt format
      
      // Verify hashing is consistent
      const isValid = await security.verifyPassword(password, hashedPassword);
      expect(isValid).toBe(true);
    });

    test('JWT tokens are properly secured', () => {
      const userData = { id: 'user-1', role: 'admin' };
      const token = security.generateJWT(userData);
      
      // Verify token structure
      expect(token.split('.')).toHaveLength(3);
      
      // Verify signature
      const decoded = security.verifyJWT(token);
      expect(decoded).toMatchObject(userData);
    });

    test('role-based access control works', async () => {
      const adminToken = security.generateJWT({ role: 'admin' });
      const userToken = security.generateJWT({ role: 'user' });
      
      // Admin access
      const adminResponse = await axios.get('/admin/settings', {
        headers: { Authorization: `Bearer ${adminToken}` }
      });
      expect(adminResponse.status).toBe(200);
      
      // User access (should fail)
      try {
        await axios.get('/admin/settings', {
          headers: { Authorization: `Bearer ${userToken}` }
        });
        fail('Should not allow user access');
      } catch (error) {
        expect(error.response.status).toBe(403);
      }
    });
  });

  describe('API Security', () => {
    test('input validation prevents injection', async () => {
      const maliciousInput = {
        query: "'; DROP TABLE users; --"
      };
      
      try {
        await axios.post('/api/search', maliciousInput);
        fail('Should reject malicious input');
      } catch (error) {
        expect(error.response.status).toBe(400);
      }
    });

    test('rate limiting prevents abuse', async () => {
      const requests = Array(100).fill().map(() => 
        axios.get('/api/test')
      );
      
      const responses = await Promise.allSettled(requests);
      const tooManyRequests = responses.filter(r => 
        r.status === 'rejected' && r.reason.response.status === 429
      );
      
      expect(tooManyRequests.length).toBeGreaterThan(0);
    });

    test('prevents CSRF attacks', async () => {
      const response = await axios.get('/api/csrf-token');
      const csrfToken = response.data.token;
      
      // Valid request with token
      const validResponse = await axios.post('/api/protected', {}, {
        headers: { 'X-CSRF-Token': csrfToken }
      });
      expect(validResponse.status).toBe(200);
      
      // Invalid request without token
      try {
        await axios.post('/api/protected', {});
        fail('Should require CSRF token');
      } catch (error) {
        expect(error.response.status).toBe(403);
      }
    });
  });

  describe('Audit Logging', () => {
    test('sensitive operations are logged', async () => {
      const auditLogger = require('../../utils/audit-logger');
      
      await security.performSensitiveOperation('test-user', 'data-export');
      
      const logs = await auditLogger.getLogs('test-user');
      expect(logs).toContainEqual(
        expect.objectContaining({
          action: 'data-export',
          user: 'test-user',
          timestamp: expect.any(String)
        })
      );
    });

    test('log integrity is maintained', async () => {
      const auditLogger = require('../../utils/audit-logger');
      
      // Create log entry
      const logEntry = await auditLogger.log({
        action: 'test-action',
        user: 'test-user'
      });
      
      // Verify hash chain
      const isValid = await auditLogger.verifyIntegrity(logEntry.id);
      expect(isValid).toBe(true);
    });
  });

  describe('Data Validation', () => {
    test('validates spacecraft command schema', () => {
      const validCommand = {
        type: 'MANEUVER',
        parameters: {
          delta_v: [1.0, 0.0, 0.0],
          timestamp: '2024-01-01T00:00:00Z'
        }
      };
      
      const result = validateSchema('spacecraft-command', validCommand);
      expect(result.valid).toBe(true);
    });

    test('validates telemetry data format', () => {
      const telemetry = {
        spacecraft_id: 'test-1',
        timestamp: '2024-01-01T00:00:00Z',
        measurements: {
          temperature: 293.15,
          pressure: 101.325,
          radiation: 0.1
        }
      };
      
      const result = validateSchema('telemetry', telemetry);
      expect(result.valid).toBe(true);
    });
  });
}); 