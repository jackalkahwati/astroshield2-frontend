const knex = require('knex');
const config = require('../../knexfile');

const db = knex(config.test);

async function setupTestDatabase() {
  try {
    // Run migrations
    await db.migrate.latest();

    // Clean existing test data
    await teardownTestDatabase();

    // Seed test data
    await db('users').insert({
      id: 'test-user',
      username: 'testuser',
      password_hash: '$2b$10$abcdefghijklmnopqrstuvwxyz123456',
      role: 'admin'
    });

    await db('spacecraft').insert({
      id: 'test-1',
      name: 'Test Spacecraft 1',
      status: 'operational'
    });
  } catch (error) {
    console.error('Error setting up test database:', error);
    throw error;
  }
}

async function teardownTestDatabase() {
  try {
    // Clean up test data - use knex's delete instead of raw queries
    await db('audit_logs').delete();
    await db('commands').delete();
    await db('telemetry').delete();
    await db('spacecraft').delete();
    await db('users').delete();
  } catch (error) {
    console.error('Error tearing down test database:', error);
    throw error;
  }
}

module.exports = {
  db,
  setupTestDatabase,
  teardownTestDatabase
}; 