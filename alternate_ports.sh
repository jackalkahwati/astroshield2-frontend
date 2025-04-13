#!/bin/bash
set -e

echo "=== Using Alternate Ports for Deployment ==="
echo "This script will modify the deployment to use different port mappings"

# Create the script to run on the EC2 instance
cat > ec2_alternate_ports.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Setting up alternate ports for Docker services ==="

cd /home/stardrive/astroshield/deployment

# Backup original docker-compose.yml
cp docker-compose.yml docker-compose.yml.bak

# Create new docker-compose.yml with different port mappings
cat > docker-compose.yml << 'EOT'
version: '3'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    restart: always
    ports:
      - "3010:3000"  # Changed from 3000:3000
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:3001/api/v1
      - NODE_ENV=production
    networks:
      - astroshield-net
    depends_on:
      - backend

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    restart: always
    ports:
      - "3011:3001"  # Changed from 3001:3001
    environment:
      - PORT=3001
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/astroshield
    networks:
      - astroshield-net
    depends_on:
      - postgres

  postgres:
    image: postgres:14
    restart: always
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=astroshield
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5433:5432"  # Changed from 5432:5432
    networks:
      - astroshield-net

  nginx:
    image: nginx:latest
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    networks:
      - astroshield-net
    depends_on:
      - frontend
      - backend

networks:
  astroshield-net:
    driver: bridge

volumes:
  postgres_data:
EOT

# Also update the nginx configuration to point to the new ports
cat > nginx/nginx.conf << 'EOT'
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
        server_name astroshield.sdataplab.com;
        
        # Redirect to HTTPS
        location / {
            return 301 https://$host$request_uri;
        }
    }

    server {
        listen 443 ssl;
        server_name astroshield.sdataplab.com;

        ssl_certificate /etc/nginx/ssl/server.crt;
        ssl_certificate_key /etc/nginx/ssl/server.key;
        
        # Frontend - Note that inside Docker network we still use port 3000
        location / {
            proxy_pass http://frontend:3000;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }

        # Backend API - Note that inside Docker network we still use port 3001
        location /api/v1/ {
            proxy_pass http://backend:3001/api/v1/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
    }
}
EOT

# Stop any existing Docker containers
sudo docker-compose down

# Start the services with new port mappings
echo "Starting services with alternate ports..."
sudo docker-compose up -d

# Show running containers
echo "Running containers:"
sudo docker ps

echo "=== Alternate ports configuration complete ==="
echo "The application should now be accessible at https://astroshield.sdataplab.com/"
EOF

# Transfer the script to EC2
echo "Transferring script to EC2..."
chmod +x ec2_alternate_ports.sh
scp ec2_alternate_ports.sh astroshield:~/

# Run the script on EC2
echo "Running script on EC2..."
ssh astroshield "chmod +x ~/ec2_alternate_ports.sh && ~/ec2_alternate_ports.sh"

echo "Deployment with alternate ports completed."
echo "The application should now be accessible at https://astroshield.sdataplab.com/" 