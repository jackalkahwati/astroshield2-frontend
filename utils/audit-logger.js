const crypto = require('crypto');
const { createLogger } = require('./logger');

class AuditLogger {
  constructor() {
    this.logger = createLogger();
    this.logChain = [];
  }

  async log(entry) {
    // Add timestamp if not present
    if (!entry.timestamp) {
      entry.timestamp = new Date().toISOString();
    }

    // Add hash of previous entry to maintain chain
    const previousHash = this.logChain.length > 0 
      ? this.logChain[this.logChain.length - 1].hash 
      : '0';

    // Create hash of current entry
    const hash = this.createHash({
      ...entry,
      previousHash
    });

    const logEntry = {
      ...entry,
      hash,
      previousHash
    };

    // Add to chain
    this.logChain.push(logEntry);

    // Log to persistent storage
    this.logger.info('Audit Log Entry', {
      ...logEntry,
      type: 'AUDIT'
    });

    return logEntry;
  }

  async getLogs(userId) {
    return this.logChain.filter(entry => entry.user === userId);
  }

  async verifyIntegrity(entryId) {
    const index = this.logChain.findIndex(entry => entry.id === entryId);
    if (index === -1) return false;

    // Verify current entry
    const entry = this.logChain[index];
    const calculatedHash = this.createHash({
      ...entry,
      hash: undefined,
      previousHash: entry.previousHash
    });

    if (calculatedHash !== entry.hash) return false;

    // Verify chain up to this entry
    for (let i = index; i > 0; i--) {
      const currentEntry = this.logChain[i];
      const previousEntry = this.logChain[i - 1];

      if (currentEntry.previousHash !== previousEntry.hash) {
        return false;
      }
    }

    return true;
  }

  createHash(data) {
    return crypto
      .createHash('sha256')
      .update(JSON.stringify(data))
      .digest('hex');
  }
}

module.exports = new AuditLogger(); 