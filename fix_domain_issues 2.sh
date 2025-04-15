#!/bin/bash
set -e

echo "=== Fixing Domain and Security Issues ==="
echo "This script will create a new solution for accessing the site"

# Create the script to run on the EC2 instance
cat > ec2_fix_access.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Setting up direct access solution ==="

# Get the public IP
public_ip=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "Unknown")
echo "EC2 Public IP: $public_ip"

# Update Nginx config to work with the IP directly
cd /home/stardrive/astroshield/deployment

# Update nginx.conf to work with direct IP
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
            proxy_pass http://frontend:80;
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
            proxy_pass http://frontend:80;
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

# Restart containers
echo "Restarting containers with updated configuration..."
sudo docker-compose restart nginx

# Check if ports are accessible from the instance itself
echo "Checking local access to ports:"
echo "Port 80 on Nginx:"
curl -s -m 5 -o /dev/null -w "%{http_code}" http://localhost:80/ || echo " - Connection failed"
echo ""

echo "Port 443 on Nginx:"
curl -s -m 5 -o /dev/null -w "%{http_code}" -k https://localhost:443/ || echo " - Connection failed"
echo ""

# Verify the frontend content is accessible
echo "Testing frontend content:"
curl -s localhost:80 | grep -o "<title>.*</title>"

echo "=== Direct access solution setup complete ==="
echo "Now try accessing the site using the IP address directly:"
echo "http://$public_ip"
echo ""
echo "You should also verify if security groups allow incoming traffic on ports 80 and 443"
EOF

# Transfer the script to EC2
echo "Transferring script to EC2..."
chmod +x ec2_fix_access.sh
scp ec2_fix_access.sh astroshield:~/

# Run the script on EC2
echo "Running fix script on EC2..."
ssh astroshield "chmod +x ~/ec2_fix_access.sh && ~/ec2_fix_access.sh"

echo "Attempted to fix access issues."
echo "Please try accessing the site using the EC2 public IP directly." 