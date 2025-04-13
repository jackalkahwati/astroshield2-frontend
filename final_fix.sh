#!/bin/bash
set -e

echo "=== Final Nginx Configuration Fix ==="
echo "This script will update the Nginx configuration to use the correct frontend port"

# Create the script to run on the EC2 instance
cat > ec2_final_fix.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Fixing Nginx configuration ==="

cd /home/stardrive/astroshield/deployment

# Update nginx.conf to use port 3000
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
        
        # Static frontend - using port 3000 instead of 80
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
    }
}
EOT

# Restart the nginx container
echo "Restarting nginx..."
sudo docker-compose restart nginx

# Check logs
sleep 3
echo "Checking nginx logs:"
sudo docker logs deployment-nginx-1

# Check if the site is accessible
echo "Testing local access:"
curl -k https://localhost

echo "=== Nginx configuration updated ==="
EOF

# Transfer the script to EC2
echo "Transferring script to EC2..."
chmod +x ec2_final_fix.sh
scp ec2_final_fix.sh astroshield:~/

# Run the script on EC2
echo "Running script on EC2..."
ssh astroshield "chmod +x ~/ec2_final_fix.sh && ~/ec2_final_fix.sh"

echo "Nginx configuration has been updated."
echo "The site should now be accessible at https://astroshield.sdataplab.com/" 