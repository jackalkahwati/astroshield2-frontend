#!/bin/bash
set -e

echo "=== Deploying Simple AstroShield Dashboard Application ==="
echo "This script will deploy a complete dashboard with frontend, API and database"

# Create the deployment script to run on the EC2 instance
cat > ec2_simple_dashboard.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Deploying Simple AstroShield Dashboard ==="
cd /home/stardrive

# Create application directory
echo "Creating application directory..."
mkdir -p astroshield-dashboard
cd astroshield-dashboard

# Create docker-compose.yml for the stack
echo "Creating docker-compose configuration..."

mkdir -p config nginx/conf.d

cat > docker-compose.yml << 'EOT'
version: '3'

services:
  frontend:
    image: nginx:alpine
    restart: always
    volumes:
      - ./frontend:/usr/share/nginx/html
      - ./nginx/conf.d:/etc/nginx/conf.d
    ports:
      - "9000:80"
    networks:
      - astroshield-net

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    restart: always
    environment:
      - PORT=9001
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/astroshield
    ports:
      - "9001:9001"
    depends_on:
      - postgres
    networks:
      - astroshield-net

  postgres:
    image: postgres:14-alpine
    restart: always
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=astroshield
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "9002:5432"
    networks:
      - astroshield-net

networks:
  astroshield-net:
    driver: bridge

volumes:
  postgres_data:
EOT

# Configure nginx
cat > nginx/conf.d/default.conf << 'EOT'
server {
    listen 80;
    server_name _;
    
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    
    location /api/ {
        proxy_pass http://backend:9001/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
EOT

# Create a simple frontend with modern design
mkdir -p frontend
cat > frontend/index.html << 'EOT'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AstroShield Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.3/font/bootstrap-icons.css">
    <style>
        :root {
            --primary-color: #0a2342;
            --secondary-color: #2c5282;
            --accent-color: #4fd1c5;
            --light-bg: #f7fafc;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background-color: var(--light-bg);
            color: #333;
            margin: 0;
            padding-top: 56px;
        }
        
        .navbar {
            background-color: var(--primary-color);
        }
        
        .navbar-brand {
            display: flex;
            align-items: center;
            color: white;
            font-weight: bold;
        }
        
        .status-card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 1rem;
        }
        
        .status-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }
        
        .card-header {
            background-color: var(--primary-color);
            color: white;
        }
        
        .status-badge {
            font-size: 0.8rem;
            padding: 0.25rem 0.5rem;
        }
        
        .chart-container {
            height: 300px;
            position: relative;
        }
        
        .sidebar {
            background-color: var(--primary-color);
            color: white;
            height: calc(100vh - 56px);
            position: fixed;
            top: 56px;
            left: 0;
            width: 250px;
            padding-top: 20px;
            transition: all 0.3s;
            z-index: 999;
        }
        
        .sidebar.collapsed {
            margin-left: -250px;
        }
        
        .sidebar .nav-link {
            color: rgba(255, 255, 255, 0.8);
            padding: 10px 20px;
            border-radius: 5px;
            margin: 5px 10px;
        }
        
        .sidebar .nav-link:hover,
        .sidebar .nav-link.active {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
        }
        
        .sidebar .nav-link i {
            margin-right: 10px;
        }
        
        .content {
            margin-left: 250px;
            padding: 20px;
            transition: all 0.3s;
        }
        
        .content.expanded {
            margin-left: 0;
        }
        
        .toggle-sidebar {
            position: fixed;
            bottom: 20px;
            left: 20px;
            z-index: 1000;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        
        .toggle-sidebar:focus {
            outline: none;
        }
        
        .api-response {
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 15px;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
        
        @media (max-width: 768px) {
            .sidebar {
                margin-left: -250px;
            }
            
            .sidebar.expanded {
                margin-left: 0;
            }
            
            .content {
                margin-left: 0;
            }
        }
        
        /* Event Severity Colors */
        .severity-high {
            background-color: #f56565;
            color: white;
        }
        
        .severity-medium {
            background-color: #ed8936;
            color: white;
        }
        
        .severity-low {
            background-color: #48bb78;
            color: white;
        }
    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="bi bi-stars me-2"></i>
                AstroShield Dashboard
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="#"><i class="bi bi-gear"></i></a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#"><i class="bi bi-bell"></i></a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#"><i class="bi bi-person-circle"></i></a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Sidebar -->
    <div class="sidebar" id="sidebar">
        <ul class="nav flex-column">
            <li class="nav-item">
                <a class="nav-link active" href="#">
                    <i class="bi bi-house"></i> Dashboard
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#">
                    <i class="bi bi-satellite"></i> Satellites
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#">
                    <i class="bi bi-exclamation-triangle"></i> Alerts
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#">
                    <i class="bi bi-graph-up"></i> Analytics
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#">
                    <i class="bi bi-gear"></i> Settings
                </a>
            </li>
        </ul>
    </div>

    <!-- Main Content -->
    <div class="content" id="content">
        <div class="container-fluid">
            <div class="row mb-4">
                <div class="col-12">
                    <h2>Space Situational Awareness Dashboard</h2>
                    <p class="text-muted">Real-time monitoring and analytics for satellite operations</p>
                </div>
            </div>

            <!-- Status Cards -->
            <div class="row">
                <div class="col-xl-3 col-md-6">
                    <div class="card status-card">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 class="card-title text-muted">System Status</h6>
                                    <h4 class="mb-0">Operational</h4>
                                </div>
                                <span class="badge bg-success">Online</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-xl-3 col-md-6">
                    <div class="card status-card">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 class="card-title text-muted">API Status</h6>
                                    <h4 class="mb-0" id="api-status">Checking...</h4>
                                </div>
                                <span class="badge bg-warning" id="api-badge">Pending</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-xl-3 col-md-6">
                    <div class="card status-card">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 class="card-title text-muted">Active Satellites</h6>
                                    <h4 class="mb-0" id="satellites-count">3</h4>
                                </div>
                                <i class="bi bi-satellite text-primary" style="font-size: 2rem;"></i>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-xl-3 col-md-6">
                    <div class="card status-card">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 class="card-title text-muted">Recent Events</h6>
                                    <h4 class="mb-0" id="events-count">2</h4>
                                </div>
                                <span class="badge bg-danger">1 High Priority</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Charts Section -->
            <div class="row mt-4">
                <div class="col-xl-6">
                    <div class="card mb-4">
                        <div class="card-header">
                            <h5 class="mb-0">Satellite Orbit Distribution</h5>
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="orbitChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-xl-6">
                    <div class="card mb-4">
                        <div class="card-header">
                            <h5 class="mb-0">Monthly Event Trends</h5>
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="eventChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Recent Events -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card mb-4">
                        <div class="card-header">
                            <h5 class="mb-0">Recent Events</h5>
                        </div>
                        <div class="card-body">
                            <div id="events-container">
                                <div class="d-flex justify-content-center">
                                    <div class="spinner-border text-primary" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- API Testing -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card mb-4">
                        <div class="card-header">
                            <h5 class="mb-0">API Testing</h5>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <button id="test-health" class="btn btn-primary me-2">Test Health</button>
                                <button id="test-satellites" class="btn btn-primary me-2">Get Satellites</button>
                                <button id="test-events" class="btn btn-primary">Get Events</button>
                            </div>
                            <div class="api-response" id="api-response">API response will appear here...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Toggle Sidebar Button -->
    <button class="toggle-sidebar" id="toggleSidebar">
        <i class="bi bi-list"></i>
    </button>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Toggle Sidebar
        document.getElementById('toggleSidebar').addEventListener('click', function() {
            document.getElementById('sidebar').classList.toggle('collapsed');
            document.getElementById('content').classList.toggle('expanded');
        });

        // Charts
        const orbitChart = new Chart(document.getElementById('orbitChart'), {
            type: 'bar',
            data: {
                labels: ['LEO', 'MEO', 'GEO', 'HEO'],
                datasets: [{
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
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        const eventChart = new Chart(document.getElementById('eventChart'), {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [
                    {
                        label: 'Proximity Events',
                        data: [12, 19, 15, 22, 28, 25],
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.3
                    },
                    {
                        label: 'Maneuver Events',
                        data: [8, 15, 12, 17, 10, 14],
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });

        // API Functions
        document.getElementById('test-health').addEventListener('click', function() {
            fetchData('/api/health');
        });

        document.getElementById('test-satellites').addEventListener('click', function() {
            fetchData('/api/satellites');
        });

        document.getElementById('test-events').addEventListener('click', function() {
            fetchData('/api/events');
        });

        function fetchData(endpoint) {
            const apiResponse = document.getElementById('api-response');
            apiResponse.textContent = 'Loading...';
            
            fetch(endpoint)
                .then(response => response.json())
                .then(data => {
                    apiResponse.textContent = JSON.stringify(data, null, 2);
                })
                .catch(error => {
                    apiResponse.textContent = `Error: ${error.message}`;
                });
        }

        // Load initial data
        function loadInitialData() {
            // Check API health
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    const apiStatus = document.getElementById('api-status');
                    const apiBadge = document.getElementById('api-badge');
                    
                    if (data.status === 'healthy') {
                        apiStatus.textContent = 'Healthy';
                        apiBadge.textContent = 'Online';
                        apiBadge.className = 'badge bg-success';
                    } else {
                        apiStatus.textContent = 'Degraded';
                        apiBadge.textContent = 'Issues';
                        apiBadge.className = 'badge bg-danger';
                    }
                })
                .catch(() => {
                    const apiStatus = document.getElementById('api-status');
                    const apiBadge = document.getElementById('api-badge');
                    apiStatus.textContent = 'Offline';
                    apiBadge.textContent = 'Error';
                    apiBadge.className = 'badge bg-danger';
                });
            
            // Load satellites
            fetch('/api/satellites')
                .then(response => response.json())
                .then(data => {
                    const satellitesCount = document.getElementById('satellites-count');
                    satellitesCount.textContent = data.length;
                })
                .catch(error => console.error('Error fetching satellites:', error));
            
            // Load events
            fetch('/api/events')
                .then(response => response.json())
                .then(data => {
                    const eventsCount = document.getElementById('events-count');
                    const eventsContainer = document.getElementById('events-container');
                    
                    eventsCount.textContent = data.length;
                    eventsContainer.innerHTML = '';
                    
                    data.forEach(event => {
                        const severityClass = event.severity === 'High' ? 'severity-high' : 
                                            event.severity === 'Medium' ? 'severity-medium' : 'severity-low';
                        
                        const eventCard = document.createElement('div');
                        eventCard.className = 'card mb-3';
                        eventCard.innerHTML = `
                            <div class="card-body">
                                <div class="row align-items-center">
                                    <div class="col-md-1 text-center">
                                        <i class="bi bi-exclamation-triangle-fill text-${event.severity === 'High' ? 'danger' : 'warning'} fs-3"></i>
                                    </div>
                                    <div class="col-md-7">
                                        <h6 class="mb-0">${event.description}</h6>
                                        <small class="text-muted">Event ID: ${event.id}</small>
                                    </div>
                                    <div class="col-md-2">
                                        <span class="badge ${severityClass}">${event.severity}</span>
                                    </div>
                                    <div class="col-md-2 text-end">
                                        <button class="btn btn-sm btn-outline-primary">Details</button>
                                    </div>
                                </div>
                            </div>
                        `;
                        
                        eventsContainer.appendChild(eventCard);
                    });
                })
                .catch(error => {
                    console.error('Error fetching events:', error);
                    const eventsContainer = document.getElementById('events-container');
                    eventsContainer.innerHTML = '<div class="alert alert-danger">Error loading events</div>';
                });
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', loadInitialData);
    </script>
</body>
</html>
EOT

# Create backend
mkdir -p backend
cat > backend/Dockerfile << 'EOT'
FROM node:16-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 9001
CMD ["node", "server.js"]
EOT

cat > backend/package.json << 'EOT'
{
  "name": "astroshield-api",
  "version": "1.0.0",
  "description": "AstroShield API Backend",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "cors": "^2.8.5",
    "express": "^4.18.2",
    "pg": "^8.10.0"
  }
}
EOT

cat > backend/server.js << 'EOT'
const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');

// Initialize Express
const app = express();
const PORT = process.env.PORT || 9001;

// Middleware
app.use(cors());
app.use(express.json());

// Database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    // Check database connection
    await pool.query('SELECT NOW()');
    
    res.json({ 
      status: 'healthy', 
      version: '1.0.0',
      services: {
        database: 'connected'
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
app.get('/satellites', async (req, res) => {
  try {
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
    
    res.json(satellites);
  } catch (error) {
    console.error('Error fetching satellites', error);
    res.status(500).json({ error: error.message });
  }
});

// Events endpoint
app.get('/events', async (req, res) => {
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

# Create PostgreSQL initialization script
mkdir -p init-scripts
cat > init-scripts/01-init.sql << 'EOT'
-- Create tables for the AstroShield database

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

# Start the deployment
echo "Starting the dashboard deployment..."
docker-compose down
docker-compose build
docker-compose up -d

# Check the status
echo "Checking service status..."
docker ps

echo "=== Dashboard deployment completed! ==="
echo "The application can now be accessed at: http://localhost:9000"
EOF

# Transfer the script to EC2
echo "Transferring deployment script to EC2..."
chmod +x ec2_simple_dashboard.sh
scp ec2_simple_dashboard.sh astroshield:~/

# Run the script on EC2
echo "Running dashboard deployment on EC2..."
ssh astroshield "chmod +x ~/ec2_simple_dashboard.sh && sudo ~/ec2_simple_dashboard.sh"

echo "Dashboard application deployment completed!"
echo "You can now access the dashboard with:"
echo "ssh -L 9000:localhost:9000 astroshield"
echo "Then access: http://127.0.0.1:9000" 