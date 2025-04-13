#!/bin/bash
set -e

echo "=== Creating AstroShield landing page ==="

cd /home/stardrive/astroshield/deployment

# Create a proper landing page
mkdir -p frontend/app/public
cat > frontend/app/public/index.html << 'EOT'
<!DOCTYPE html>
<html>
<head>
    <title>AstroShield Platform</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --primary-color: #0a2342;
            --secondary-color: #2c5282;
            --accent-color: #4fd1c5;
            --text-color: #2d3748;
            --light-bg: #f7fafc;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            color: var(--text-color);
            background: linear-gradient(135deg, #f6f9fc 0%, #edf2f7 100%);
            min-height: 100vh;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .container {
            width: 90%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        
        .logo-icon {
            margin-right: 10px;
            color: var(--accent-color);
        }
        
        .hero {
            text-align: center;
            padding: 4rem 0;
            background: rgba(255, 255, 255, 0.8);
            margin: 2rem 0;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            color: var(--primary-color);
        }
        
        .subtitle {
            font-size: 1.25rem;
            color: var(--secondary-color);
            margin-bottom: 2rem;
        }
        
        .status-card {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            max-width: 800px;
            margin: 0 auto;
        }
        
        .status-title {
            font-size: 1.5rem;
            color: var(--primary-color);
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid #edf2f7;
        }
        
        .status-item:last-child {
            border-bottom: none;
        }
        
        .status-label {
            flex: 1;
            font-weight: 500;
        }
        
        .status-value {
            padding: 0.25rem 0.75rem;
            border-radius: 16px;
            font-size: 0.875rem;
            font-weight: 600;
        }
        
        .status-success {
            background-color: #c6f6d5;
            color: #2f855a;
        }
        
        .status-warning {
            background-color: #feebc8;
            color: #c05621;
        }
        
        .status-error {
            background-color: #fed7d7;
            color: #c53030;
        }
        
        .message {
            background-color: #ebf8ff;
            color: #2b6cb0;
            padding: 1rem;
            border-radius: 4px;
            margin: 2rem 0;
            text-align: center;
        }
        
        footer {
            text-align: center;
            padding: 2rem 0;
            margin-top: 2rem;
            color: #718096;
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <nav>
                <div class="logo">
                    <span class="logo-icon">🛰️</span>
                    AstroShield
                </div>
            </nav>
        </div>
    </header>
    
    <div class="container">
        <div class="hero">
            <h1>AstroShield Platform</h1>
            <p class="subtitle">Space Situational Awareness & Satellite Protection System</p>
            
            <div class="status-card">
                <h2 class="status-title">System Status</h2>
                
                <div class="status-item">
                    <span class="status-label">Frontend Server</span>
                    <span class="status-value status-success">ONLINE</span>
                </div>
                
                <div class="status-item">
                    <span class="status-label">HTTPS Encryption</span>
                    <span class="status-value status-success">ACTIVE</span>
                </div>
                
                <div class="status-item">
                    <span class="status-label">API Backend</span>
                    <span class="status-value status-warning">IN PROGRESS</span>
                </div>
                
                <div class="status-item">
                    <span class="status-label">Database Connection</span>
                    <span class="status-value status-warning">PENDING</span>
                </div>
            </div>
            
            <div class="message">
                The deployment is in progress. The frontend is accessible, and we're currently configuring the backend services.
            </div>
        </div>
    </div>
    
    <footer>
        <div class="container">
            <p>&copy; 2025 AstroShield - Space Situational Awareness Platform</p>
        </div>
    </footer>
</body>
</html>
EOT

# Create a script to copy the landing page to the right place
cat > copy_landing.sh << 'EOT'
#!/bin/bash
sudo docker cp frontend/app/public/index.html deployment-frontend-1:/usr/share/nginx/html/index.html
EOT

# Make script executable
chmod +x copy_landing.sh

# Run the script to copy the landing page
./copy_landing.sh

echo "Landing page has been created and deployed."
echo "Visit https://astroshield.sdataplab.com/ to see it."
