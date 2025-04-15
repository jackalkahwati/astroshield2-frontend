/**
 * Deploy Astroshield components to Vantiq
 */
const fs = require('fs');
const path = require('path');
const axios = require('axios');

// Configuration
const config = {
    vantiqUrl: process.env.VANTIQ_URL || 'https://dev.vantiq.com',
    authToken: process.env.VANTIQ_TOKEN,
    namespace: process.env.VANTIQ_NAMESPACE || 'system',
    resources: [
        { type: 'type', path: './types/ManeuverDetection.json' },
        { type: 'type', path: './types/ObservationWindow.json' },
        { type: 'source', path: './config/vantiq-source-config.json' },
        { type: 'procedure', path: './src/messageTransformer.js' }
    ]
};

// Create Vantiq API client
const vantiqClient = axios.create({
    baseURL: config.vantiqUrl,
    headers: {
        'Authorization': `Bearer ${config.authToken}`,
        'Content-Type': 'application/json'
    }
});

// Deploy resources
async function deployResources() {
    for (const resource of config.resources) {
        try {
            console.log(`Deploying ${resource.type}: ${resource.path}`);
            
            // Read resource file
            const content = fs.readFileSync(path.resolve(__dirname, resource.path), 'utf8');
            const resourceData = JSON.parse(content);
            
            // Create resource in Vantiq
            await vantiqClient.post(`/api/v1/${config.namespace}/${resource.type}s`, resourceData);
            
            console.log(`Successfully deployed ${resource.type}: ${resourceData.name}`);
        } catch (error) {
            console.error(`Error deploying ${resource.path}:`, error.message);
        }
    }
}

// Run deployment
deployResources()
    .then(() => console.log('Deployment completed'))
    .catch(err => console.error('Deployment failed:', err)); 