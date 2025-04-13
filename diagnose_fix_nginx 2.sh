#!/bin/bash
set -e

echo "=== Nginx and Docker Diagnostic Script ==="
echo "This script will diagnose and fix common issues with Nginx and Docker"

# Create the script to run on the EC2 instance
cat > ec2_diagnose_fix.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Running diagnostics on AstroShield deployment ==="

# Check if Docker containers are running
echo "Checking Docker containers..."
docker ps

# Check Docker container logs
echo "Checking Docker container logs..."
for container in $(docker ps -q); do
  name=$(docker inspect --format='{{.Name}}' $container | sed 's/\///')
  echo "=== Logs for container: $name ==="
  docker logs --tail 20 $container
  echo ""
done

# Check Docker networks
echo "Checking Docker networks..."
docker network ls
echo ""

# Fix docker-compose network configuration
echo "Modifying docker-compose.yml to fix network configuration..."
cd /home/stardrive/astroshield/deployment

# Create a backup
cp docker-compose.yml docker-compose.yml.bak

# Update docker-compose.yml to fix network issues
cat > docker-compose.yml << 'EOT'
version: '3'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    restart: always
    ports:
      - "3000:3000"
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
      - "3001:3001"
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
      - "5432:5432"
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

# Fix nginx configuration
echo "Checking and fixing Nginx configuration..."
mkdir -p nginx
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
        
        # Frontend
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

        # Backend API
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

# Regenerate SSL certificate
echo "Regenerating SSL certificate..."
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/server.key -out nginx/ssl/server.crt \
    -subj '/CN=astroshield.sdataplab.com'

# Restart Docker containers
echo "Restarting Docker containers..."
docker-compose down
docker-compose up -d

# Check the new status
echo "New container status:"
docker ps

echo "=== Diagnostics and fixes completed ==="
echo "Try accessing https://astroshield.sdataplab.com/ again."
EOF

# Transfer the diagnostic script to EC2
echo "Transferring diagnostic script to EC2..."
chmod +x ec2_diagnose_fix.sh
scp ec2_diagnose_fix.sh astroshield:~/

# Run the diagnostic script on EC2
echo "Running diagnostic script on EC2..."
ssh astroshield "chmod +x ~/ec2_diagnose_fix.sh && ~/ec2_diagnose_fix.sh"

echo "Diagnostics and fixes have been applied."
echo "Try accessing https://astroshield.sdataplab.com/ again."
echo "If you still experience issues, it may take a few minutes for all services to fully start." 