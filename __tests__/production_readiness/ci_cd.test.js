const axios = require('axios');
const fs = require('fs').promises;
const yaml = require('js-yaml');

describe('CI/CD Pipeline Tests', () => {
  describe('Deployment Configuration', () => {
    test('validates deployment configuration files', async () => {
      const config = await fs.readFile('./deployment/config.yml', 'utf8');
      const parsedConfig = yaml.load(config);
      
      expect(parsedConfig.version).toBeDefined();
      expect(parsedConfig.services).toBeDefined();
      expect(parsedConfig.environment).toBeDefined();
    });

    test('environment variables are properly configured', () => {
      const requiredEnvVars = [
        'DATABASE_URL',
        'API_KEY',
        'NODE_ENV',
        'LOG_LEVEL',
        'REDIS_URL'
      ];

      requiredEnvVars.forEach(envVar => {
        expect(process.env[envVar]).toBeDefined();
      });
    });

    test('database migration scripts are valid', async () => {
      const migrations = await fs.readdir('./migrations');
      expect(migrations.length).toBeGreaterThan(0);
      
      migrations.forEach(migration => {
        expect(migration).toMatch(/^\d{14}_\w+\.js$/);
      });
    });
  });

  describe('Build Process', () => {
    test('npm build succeeds', async () => {
      const { execSync } = require('child_process');
      expect(() => {
        execSync('npm run build', { stdio: 'pipe' });
      }).not.toThrow();
    });

    test('build artifacts are generated correctly', async () => {
      const buildDir = await fs.readdir('./dist');
      expect(buildDir).toContain('main.js');
      expect(buildDir).toContain('assets');
    });
  });

  describe('Rollback Procedures', () => {
    test('rollback script exists and is executable', async () => {
      const stats = await fs.stat('./scripts/rollback.sh');
      expect(stats.mode & 0o111).toBeTruthy(); // Check executable permission
    });

    test('database rollback works', async () => {
      const db = require('../test-utils/db');
      await db.migrate.up();
      await db.migrate.down();
      
      const tables = await db.raw('SELECT tablename FROM pg_tables');
      expect(tables).toEqual(expect.arrayContaining(['schema_migrations']));
    });
  });

  describe('Health Checks', () => {
    test('readiness probe returns correct status', async () => {
      const response = await axios.get('/health/ready');
      expect(response.status).toBe(200);
      expect(response.data.status).toBe('ready');
    });

    test('liveness probe detects system health', async () => {
      const response = await axios.get('/health/live');
      expect(response.status).toBe(200);
      expect(response.data.status).toBe('alive');
    });
  });

  describe('Configuration Validation', () => {
    test('validates kubernetes manifests', async () => {
      const k8sFiles = await fs.readdir('./k8s');
      
      for (const file of k8sFiles) {
        const content = await fs.readFile(`./k8s/${file}`, 'utf8');
        const manifest = yaml.load(content);
        
        expect(manifest.apiVersion).toBeDefined();
        expect(manifest.kind).toBeDefined();
        expect(manifest.metadata).toBeDefined();
      }
    });

    test('checks resource limits are set', async () => {
      const deploymentFile = await fs.readFile('./k8s/deployment.yml', 'utf8');
      const deployment = yaml.load(deploymentFile);
      
      const containers = deployment.spec.template.spec.containers;
      containers.forEach(container => {
        expect(container.resources.limits).toBeDefined();
        expect(container.resources.requests).toBeDefined();
      });
    });
  });
}); 