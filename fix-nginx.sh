#!/bin/bash

# Create a backup of the original nginx.conf
cat > /tmp/nginx-fix.sh << "EOF"
#!/bin/bash

# Create a backup of the original nginx.conf
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.bak

# Remove the default server block from the main nginx.conf
sudo sed -i '/server {/,/}/d' /etc/nginx/nginx.conf

# Add include directive for conf.d directory if it doesn't exist
if ! grep -q "include /etc/nginx/conf.d/\*.conf;" /etc/nginx/nginx.conf; then
  sudo sed -i '/http {/a \    include /etc/nginx/conf.d/*.conf;' /etc/nginx/nginx.conf
fi

# Make sure our custom configuration is correct
cat > /tmp/astroshield.conf << "CONFEND"
server {
    listen 80;
    server_name astroshield.sdataplab.com localhost;

    location / {
        root /home/stardrive/astroshield/frontend/public;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://localhost:3002/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /docs {
        proxy_pass http://localhost:3002/docs;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /openapi.json {
        proxy_pass http://localhost:3002/openapi.json;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
CONFEND

sudo mv /tmp/astroshield.conf /etc/nginx/conf.d/astroshield.conf
sudo chmod 644 /etc/nginx/conf.d/astroshield.conf

# Make sure the public directory exists and has correct permissions
sudo mkdir -p /home/stardrive/astroshield/frontend/public
sudo chmod -R 755 /home/stardrive/astroshield/frontend/public
sudo chown -R nginx:nginx /home/stardrive/astroshield/frontend/public

# Create a simple HTML page
cat > /tmp/index.html << "HTMLEND"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AstroShield</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f0f2f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #1a1a2e;
            color: white;
            padding: 20px 0;
            text-align: center;
        }
        h1 {
            margin: 0;
        }
        .content {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .api-link {
            display: block;
            margin-top: 20px;
            padding: 10px;
            background-color: #e6f7ff;
            border-radius: 4px;
            text-align: center;
        }
        a {
            color: #0066cc;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>AstroShield</h1>
        </div>
    </header>
    <div class="container">
        <div class="content">
            <h2>Welcome to AstroShield</h2>
            <p>AstroShield is an application for tracking and visualizing asteroid trajectories and impact predictions.</p>
            <p>This is a temporary landing page. The full application is being deployed.</p>
            <div class="api-link">
                <a href="/docs" target="_blank">Access API Documentation</a>
            </div>
        </div>
    </div>
</body>
</html>
HTMLEND

sudo mv /tmp/index.html /home/stardrive/astroshield/frontend/public/index.html
sudo chmod 644 /home/stardrive/astroshield/frontend/public/index.html
sudo chown nginx:nginx /home/stardrive/astroshield/frontend/public/index.html

# Restart Nginx
sudo systemctl restart nginx

# Check Nginx status
sudo systemctl status nginx
EOF

chmod +x /tmp/nginx-fix.sh 