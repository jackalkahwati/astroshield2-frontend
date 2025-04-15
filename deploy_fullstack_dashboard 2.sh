#!/bin/bash
set -e

echo "=== Deploying Full AstroShield Dashboard Application ==="
echo "This script will deploy the complete stack with Next.js frontend, API backend, Redis, and databases"

# Create the deployment script to run on the EC2 instance
cat > ec2_dashboard_deploy.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Deploying Complete AstroShield Dashboard ==="
cd /home/stardrive

# Create the dashboard directory
echo "Creating astroshield-dashboard directory..."
mkdir -p astroshield-dashboard
cd astroshield-dashboard

# Create proper docker-compose.yml for full stack
echo "Creating full stack docker-compose configuration..."

mkdir -p config

cat > docker-compose.yml << 'EOT'
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    restart: always
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=/api/v1
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
      - /app/.next
    depends_on:
      - backend
    networks:
      - astroshield-net

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    restart: always
    environment:
      - PORT=5000
      - NODE_ENV=production
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/astroshield
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=${JWT_SECRET}
    ports:
      - "5000:5000"
    depends_on:
      - postgres
      - redis
    networks:
      - astroshield-net

  postgres:
    image: postgres:14-alpine
    restart: always
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=astroshield
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    networks:
      - astroshield-net

  redis:
    image: redis:alpine
    restart: always
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - astroshield-net

  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
      - ./config/ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend
    networks:
      - astroshield-net

networks:
  astroshield-net:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
EOT

# Create .env file with secrets
cat > .env << 'EOT'
DB_PASSWORD=astroshield_secure_password
JWT_SECRET=astroshield_jwt_secret_key_very_secure_and_random
EOT

# Configure Nginx
mkdir -p config/ssl
cat > config/nginx.conf << 'EOT'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log;
pid /run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    sendfile            on;
    tcp_nopush          on;
    tcp_nodelay         on;
    keepalive_timeout   65;
    types_hash_max_size 2048;

    include             /etc/nginx/mime.types;
    default_type        application/octet-stream;

    server {
        listen 80;
        server_name _;
        
        location / {
            proxy_pass http://frontend:3000;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
        }

        location /api/v1/ {
            proxy_pass http://backend:5000/api/v1/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
        }
    }
}
EOT

# Create frontend directory with Next.js
mkdir -p frontend
cat > frontend/Dockerfile << 'EOT'
FROM node:16-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
EOT

cat > frontend/package.json << 'EOT'
{
  "name": "astroshield-dashboard",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "@emotion/react": "^11.10.6",
    "@emotion/styled": "^11.10.6",
    "@mui/icons-material": "^5.11.11",
    "@mui/material": "^5.11.11",
    "@mui/x-data-grid": "^6.0.0",
    "axios": "^1.3.4",
    "chart.js": "^4.2.1",
    "next": "13.2.3",
    "react": "18.2.0",
    "react-chartjs-2": "^5.2.0",
    "react-dom": "18.2.0",
    "swr": "^2.1.0"
  },
  "devDependencies": {
    "@types/node": "18.14.6",
    "@types/react": "18.0.28",
    "@types/react-dom": "18.0.11",
    "eslint": "8.35.0",
    "eslint-config-next": "13.2.3",
    "typescript": "4.9.5"
  }
}
EOT

mkdir -p frontend/pages
cat > frontend/pages/index.tsx << 'EOT'
import { useState, useEffect } from 'react';
import Head from 'next/head';
import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  Paper,
  AppBar,
  Toolbar,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  CircularProgress
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import DashboardIcon from '@mui/icons-material/Dashboard';
import SatelliteIcon from '@mui/icons-material/Satellite';
import WarningIcon from '@mui/icons-material/Warning';
import SettingsIcon from '@mui/icons-material/Settings';
import Person from '@mui/icons-material/Person';
import { Bar, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Define the Event type
interface Event {
  id: string;
  type: string;
  severity: 'High' | 'Medium' | 'Low';
  timestamp: string;
  description: string;
}

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export default function Home() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [satelliteData, setSatelliteData] = useState([]);
  const [eventsData, setEventsData] = useState<Event[]>([]); // Use the Event type here
  const [apiStatus, setApiStatus] = useState('loading');
  const [dbStatus, setDbStatus] = useState('loading');

  useEffect(() => {
    // Check API health
    fetch('/api/v1/health')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'healthy') {
          setApiStatus('online');
        } else {
          setApiStatus('error');
        }
      })
      .catch(() => setApiStatus('error'));
    
    // Fetch satellites data
    fetch('/api/v1/satellites')
      .then(res => res.json())
      .then(data => {
        setSatelliteData(data);
        setDbStatus('online');
      })
      .catch(() => setDbStatus('error'));

    // Fetch events data
    fetch('/api/v1/events')
      .then(res => res.json())
      .then((data: Event[]) => { // Type assertion for fetched data
        setEventsData(data);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  // Chart data for satellite orbit distribution
  const orbitDistributionData = {
    labels: ['LEO', 'MEO', 'GEO', 'HEO'],
    datasets: [
      {
        label: 'Satellites by Orbit',
        data: [65, 22, 18, 5],
        backgroundColor: [
          'rgba(54, 162, 235, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(153, 102, 255, 0.6)',
          'rgba(255, 159, 64, 0.6)'
        ],
        borderColor: [
          'rgba(54, 162, 235, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
          'rgba(255, 159, 64, 1)'
        ],
        borderWidth: 1,
      },
    ],
  };

  // Chart data for monthly events
  const eventTrendData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        label: 'Proximity Events',
        data: [12, 19, 15, 22, 28, 25],
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
      },
      {
        label: 'Maneuver Events',
        data: [8, 15, 12, 17, 10, 14],
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
      },
    ],
  };

  return (
    <>
      <Head>
        <title>AstroShield Dashboard</title>
        <meta name="description" content="Space Situational Awareness & Satellite Protection System" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Box sx={{ display: 'flex' }}>
        {/* App Bar */}
        <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={() => setDrawerOpen(!drawerOpen)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
            <SatelliteIcon sx={{ mr: 1 }} />
            <Typography variant="h6" noWrap component="div">
              AstroShield Dashboard
            </Typography>
          </Toolbar>
        </AppBar>

        {/* Drawer */}
        <Drawer
          variant="temporary"
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          sx={{
            width: 240,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: 240,
              boxSizing: 'border-box',
            },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto' }}>
            <List>
              <ListItem button selected>
                <ListItemIcon>
                  <DashboardIcon />
                </ListItemIcon>
                <ListItemText primary="Dashboard" />
              </ListItem>
              <ListItem button>
                <ListItemIcon>
                  <SatelliteIcon />
                </ListItemIcon>
                <ListItemText primary="Satellites" />
              </ListItem>
              <ListItem button>
                <ListItemIcon>
                  <WarningIcon />
                </ListItemIcon>
                <ListItemText primary="Alerts" />
              </ListItem>
            </List>
            <Divider />
            <List>
              <ListItem button>
                <ListItemIcon>
                  <Person />
                </ListItemIcon>
                <ListItemText primary="Account" />
              </ListItem>
              <ListItem button>
                <ListItemIcon>
                  <SettingsIcon />
                </ListItemIcon>
                <ListItemText primary="Settings" />
              </ListItem>
            </List>
          </Box>
        </Drawer>

        {/* Main content */}
        <Box
          component="main"
          sx={{ flexGrow: 1, p: 3, pt: 10 }}
        >
          <Container maxWidth="lg">
            <Typography variant="h4" gutterBottom component="div" sx={{ fontWeight: 'bold', mb: 4 }}>
              Space Situational Awareness Dashboard
            </Typography>

            {/* Status Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      System Status
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Chip 
                        label="Online" 
                        color="success" 
                        size="small" 
                        sx={{ mr: 1 }} 
                      />
                      <Typography variant="h6">
                        Operational
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      API Status
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {apiStatus === 'loading' ? (
                        <CircularProgress size={20} sx={{ mr: 1 }} />
                      ) : (
                        <Chip 
                          label={apiStatus === 'online' ? 'Online' : 'Error'} 
                          color={apiStatus === 'online' ? 'success' : 'error'} 
                          size="small" 
                          sx={{ mr: 1 }} 
                        />
                      )}
                      <Typography variant="h6">
                        {apiStatus === 'loading' ? 'Checking...' : 
                         apiStatus === 'online' ? 'Healthy' : 'Degraded'}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      Active Satellites
                    </Typography>
                    <Typography variant="h4">
                      {loading ? <CircularProgress size={20} /> : satelliteData.length || "3"}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      Recent Events
                    </Typography>
                    <Typography variant="h4">
                      {loading ? <CircularProgress size={20} /> : eventsData.length || "2"}
                    </Typography>
                    <Chip 
                      label="1 High Priority" 
                      color="error" 
                      size="small" 
                      sx={{ mt: 1 }} 
                    />
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Charts Section */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Satellite Orbit Distribution
                  </Typography>
                  <Box sx={{ height: 300 }}>
                    <Bar data={orbitDistributionData} options={{ maintainAspectRatio: false }} />
                  </Box>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Monthly Event Trends
                  </Typography>
                  <Box sx={{ height: 300 }}>
                    <Line data={eventTrendData} options={{ maintainAspectRatio: false }} />
                  </Box>
                </Paper>
              </Grid>
            </Grid>

            {/* Recent Events Table */}
            <Paper sx={{ p: 2, mb: 4 }}>
              <Typography variant="h6" gutterBottom>
                Recent Events
              </Typography>
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : (
                <Box>
                  {eventsData && eventsData.length > 0 ? (
                    eventsData.map((event, index) => (
                      <Card key={index} sx={{ mb: 2 }}>
                        <CardContent>
                          <Grid container spacing={2} alignItems="center">
                            <Grid item xs={1}>
                              <WarningIcon color={event.severity === 'High' ? 'error' : 'warning'} />
                            </Grid>
                            <Grid item xs={7}>
                              <Typography variant="subtitle1">{event.description}</Typography>
                              <Typography variant="body2" color="textSecondary">
                                Event ID: {event.id}
                              </Typography>
                            </Grid>
                            <Grid item xs={2}>
                              <Chip 
                                label={event.severity} 
                                color={event.severity === 'High' ? 'error' : 'warning'} 
                                size="small" 
                              />
                            </Grid>
                            <Grid item xs={2}>
                              <Button size="small" variant="outlined">Details</Button>
                            </Grid>
                          </Grid>
                        </CardContent>
                      </Card>
                    ))
                  ) : (
                    <Typography>No events found</Typography>
                  )}
                </Box>
              )}
            </Paper>

            {/* API Testing */}
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                API Testing
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <Button variant="contained" color="primary">
                  Test Health Endpoint
                </Button>
                <Button variant="contained" color="primary">
                  Get Satellites
                </Button>
                <Button variant="contained" color="primary">
                  Get Events
                </Button>
              </Box>
              <Paper sx={{ p: 2, backgroundColor: '#f5f5f5' }}>
                <Typography variant="body2" component="pre" sx={{ fontFamily: 'monospace' }}>
                  API response will appear here...
                </Typography>
              </Paper>
            </Paper>
          </Container>
        </Box>
      </Box>
    </>
  );
}
EOT

# Create Next.js config
cat > frontend/next.config.js << 'EOT'
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://backend:5000/api/:path*',
      },
    ];
  },
}

module.exports = nextConfig
EOT

# Create frontend tsconfig
cat > frontend/tsconfig.json << 'EOT'
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
EOT

# Create backend directory and files
mkdir -p backend
cat > backend/Dockerfile << 'EOT'
FROM node:16-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5000
CMD ["npm", "start"]
EOT

cat > backend/package.json << 'EOT'
{
  "name": "astroshield-api",
  "version": "1.0.0",
  "description": "AstroShield API Backend",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "cors": "^2.8.5",
    "dotenv": "^16.0.3",
    "express": "^4.18.2",
    "jsonwebtoken": "^9.0.0",
    "pg": "^8.10.0",
    "redis": "^4.6.5"
  },
  "devDependencies": {
    "nodemon": "^2.0.21"
  }
}
EOT

cat > backend/server.js << 'EOT'
const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');
const redis = require('redis');
const jwt = require('jsonwebtoken');

require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

// Redis client
let redisClient;
(async () => {
  redisClient = redis.createClient({
    url: process.env.REDIS_URL
  });
  
  redisClient.on('error', (err) => {
    console.log('Redis Client Error', err);
  });
  
  await redisClient.connect();
})();

// Health check endpoint
app.get('/api/v1/health', async (req, res) => {
  try {
    // Check database connection
    await pool.query('SELECT NOW()');
    
    // Check Redis connection
    await redisClient.ping();
    
    res.json({ 
      status: 'healthy', 
      version: '1.0.0',
      services: {
        database: 'connected',
        redis: 'connected'
      }
    });
  } catch (error) {
    console.error('Health check failed', error);
    res.status(500).json({ 
      status: 'unhealthy',
      error: error.message
    });
  }
});

// Satellites endpoint
app.get('/api/v1/satellites', async (req, res) => {
  try {
    // Try to get from cache first
    const cachedData = await redisClient.get('satellites');
    
    if (cachedData) {
      return res.json(JSON.parse(cachedData));
    }
    
    // Sample satellite data (in a real app, this would come from the database)
    const satellites = [
      {
        id: 'sat-001',
        name: 'Starlink-1234',
        type: 'Communication',
        orbit: 'LEO',
        status: 'Active',
        lastUpdated: new Date().toISOString()
      },
      {
        id: 'sat-002',
        name: 'ISS',
        type: 'Space Station',
        orbit: 'LEO',
        status: 'Active',
        lastUpdated: new Date().toISOString()
      },
      {
        id: 'sat-003',
        name: 'GPS-IIF-10',
        type: 'Navigation',
        orbit: 'MEO',
        status: 'Active',
        lastUpdated: new Date().toISOString()
      }
    ];
    
    // Cache the data
    await redisClient.set('satellites', JSON.stringify(satellites), {
      EX: 3600 // 1 hour expiration
    });
    
    res.json(satellites);
  } catch (error) {
    console.error('Error fetching satellites', error);
    res.status(500).json({ error: error.message });
  }
});

// Events endpoint
app.get('/api/v1/events', async (req, res) => {
  try {
    // Sample event data
    const events = [
      {
        id: 'evt-001',
        type: 'Proximity',
        severity: 'High',
        timestamp: new Date().toISOString(),
        description: 'Close approach detected between Starlink-1234 and debris object'
      },
      {
        id: 'evt-002',
        type: 'Maneuver',
        severity: 'Medium',
        timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        description: 'Orbital adjustment completed for GPS-IIF-10'
      }
    ];
    
    res.json(events);
  } catch (error) {
    console.error('Error fetching events', error);
    res.status(500).json({ error: error.message });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
EOT

# Create init-scripts directory for PostgreSQL initialization
mkdir -p init-scripts
cat > init-scripts/01-init.sql << 'EOT'
-- Create tables for the AstroShield database

-- Users table
CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(50) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  email VARCHAR(100) UNIQUE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Satellites table
CREATE TABLE IF NOT EXISTS satellites (
  id SERIAL PRIMARY KEY,
  satellite_id VARCHAR(50) UNIQUE NOT NULL,
  name VARCHAR(100) NOT NULL,
  type VARCHAR(50) NOT NULL,
  orbit VARCHAR(20) NOT NULL,
  status VARCHAR(20) NOT NULL,
  last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Events table
CREATE TABLE IF NOT EXISTS events (
  id SERIAL PRIMARY KEY,
  event_id VARCHAR(50) UNIQUE NOT NULL,
  type VARCHAR(50) NOT NULL,
  severity VARCHAR(20) NOT NULL,
  description TEXT,
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO satellites (satellite_id, name, type, orbit, status)
VALUES 
  ('sat-001', 'Starlink-1234', 'Communication', 'LEO', 'Active'),
  ('sat-002', 'ISS', 'Space Station', 'LEO', 'Active'),
  ('sat-003', 'GPS-IIF-10', 'Navigation', 'MEO', 'Active');

INSERT INTO events (event_id, type, severity, description, timestamp)
VALUES 
  ('evt-001', 'Proximity', 'High', 'Close approach detected between Starlink-1234 and debris object', NOW()),
  ('evt-002', 'Maneuver', 'Medium', 'Orbital adjustment completed for GPS-IIF-10', NOW() - INTERVAL '1 day');
EOT

# Generate self-signed SSL certificates for Nginx
mkdir -p config/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout config/ssl/server.key \
  -out config/ssl/server.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=astroshield.local"

# Start the deployment
echo "Starting the full dashboard deployment..."
docker-compose down
docker-compose build
docker-compose up -d

# Check the status
echo "Checking service status..."
docker ps

echo "=== Full dashboard deployment completed! ==="
echo "The application can now be accessed through SSH tunneling."
echo "Use: ssh -L 8080:localhost:80 <user>@<host>"
echo "Then open http://127.0.0.1:8080 in your browser."
EOF

# Transfer the script to EC2
echo "Transferring deployment script to EC2..."
chmod +x ec2_dashboard_deploy.sh
scp ec2_dashboard_deploy.sh astroshield:~/

# Run the script on EC2
echo "Running full dashboard deployment on EC2..."
ssh astroshield "chmod +x ~/ec2_dashboard_deploy.sh && sudo ~/ec2_dashboard_deploy.sh"

echo "Full dashboard application deployment completed!"
echo "You can now access the complete dashboard with:"
echo "ssh -L 8080:localhost:80 astroshield"
echo "Then access: http://127.0.0.1:8080" 