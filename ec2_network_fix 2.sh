#!/bin/bash
set -e

echo "=== Applying network fixes to EC2 instance ==="

# The issue: EC2 instance is in a private subnet without a public IP
# It has a private IP (10.0.11.100) but no public IP is visible in metadata

# Check if this is an Amazon Linux instance
if [ -f /etc/system-release ]; then
  echo "Amazon Linux detected"
  OS_TYPE="amazon"
elif [ -f /etc/redhat-release ]; then
  echo "Red Hat based system detected"
  OS_TYPE="redhat"
elif [ -f /etc/debian_version ]; then
  echo "Debian based system detected"
  OS_TYPE="debian"
else
  echo "Unknown OS - proceeding with generic commands"
  OS_TYPE="unknown"
fi

# Ensure necessary packages are installed
echo "Installing necessary packages..."
if [ "$OS_TYPE" == "amazon" ] || [ "$OS_TYPE" == "redhat" ]; then
  sudo yum install -y curl wget nc || true
elif [ "$OS_TYPE" == "debian" ]; then
  sudo apt-get update
  sudo apt-get install -y curl wget netcat || true
fi

# Test outbound connectivity
echo "Testing outbound connectivity..."
curl -s -m 5 google.com > /dev/null && echo "Outbound connectivity works" || echo "Outbound connectivity issues detected"

# SOLUTION: Since we have confirmed that this is a private instance,
# we need to configure the application to use the private IP

PRIVATE_IP=$(hostname -I | awk '{print $1}')
echo "Using private IP: $PRIVATE_IP"

# Update the Docker configuration to use internal networking
cd /home/stardrive/astroshield/deployment

# Create a new docker-compose file that uses host networking
echo "Creating updated docker-compose.yml with host network mode..."
cat > docker-compose.yml << EOT
version: '3'

services:
  frontend:
    image: nginx:alpine
    restart: always
    network_mode: "host"  # Use host networking
    volumes:
      - ./simple-static:/usr/share/nginx/html
    ports:
      - "3010:80"

  nginx:
    image: nginx:latest
    restart: always
    network_mode: "host"  # Use host networking
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl

EOT

# Update the nginx configuration
echo "Updating nginx.conf..."
cat > nginx/nginx.conf << EOT
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log;
pid /run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    log_format  main  '\$remote_addr - \$remote_user [\$time_local] "\$request" '
                      '\$status \$body_bytes_sent "\$http_referer" '
                      '"\$http_user_agent" "\$http_x_forwarded_for"';

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
        
        # Serve content directly on HTTP
        location / {
            # When using host networking, frontend is available on localhost
            proxy_pass http://localhost:3010;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host \$host;
            proxy_cache_bypass \$http_upgrade;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
    }

    server {
        listen 443 ssl;
        server_name _;

        ssl_certificate /etc/nginx/ssl/server.crt;
        ssl_certificate_key /etc/nginx/ssl/server.key;
        
        # Static frontend - using the correct port for the container
        location / {
            # When using host networking, frontend is available on localhost
            proxy_pass http://localhost:3010;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host \$host;
            proxy_cache_bypass \$http_upgrade;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
    }
}
EOT

# Restart the containers with new configuration
echo "Restarting containers with new configuration..."
sudo docker-compose down
sudo docker-compose up -d

# Verify service is running
echo "Verifying services are running..."
sudo docker ps

# Print the instance's private IP so it can be used for SSH tunneling
echo "===== IMPORTANT ====="
echo "This EC2 instance is in a private subnet with no public IP."
echo "To access it, you'll need to create an SSH tunnel through a bastion host or use AWS Systems Manager."
echo ""
echo "The instance's private IP is: $PRIVATE_IP"
echo ""
echo "You can use the following command to tunnel through to port 80:"
echo "ssh -i your_key.pem -L 8080:$PRIVATE_IP:80 ec2-user@your-bastion-host"
echo ""
echo "Or connect via AWS SSM if it's configured:"
echo "aws ssm start-session --target <instance-id> --document-name AWS-StartPortForwardingSession --parameters 'portNumber=80,localPortNumber=8080'"
echo "===================="

echo "=== Network fixes applied ==="
