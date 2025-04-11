# AstroShield Deployment Package

This package contains everything needed to deploy the AstroShield application to an EC2 instance.

## Quick Deployment

Run the following command as the ec2-user on your server:

```bash
chmod +x deploy-ec2.sh
./deploy-ec2.sh
```

This will:
1. Install all required dependencies
2. Configure Nginx with SSL
3. Set up the AstroShield application
4. Start the services

## Manual Deployment

If you prefer to deploy manually, follow these steps:

### 1. Install Dependencies

```bash
# Update packages
sudo yum update -y

# Install Node.js
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
source ~/.nvm/nvm.sh
nvm install 16
nvm use 16

# Install Python
sudo yum install -y python3 python3-pip

# Install Nginx
sudo amazon-linux-extras install -y nginx1
sudo systemctl enable nginx

# Install Certbot
sudo amazon-linux-extras install -y epel
sudo yum install -y certbot python-certbot-nginx
```

### 2. Set Up the Frontend

```bash
cd frontend
npm install
npm run build
```

### 3. Set Up the Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Nginx

Create a file at `/etc/nginx/conf.d/astroshield.conf` with the following content:

```
server {
    listen 80;
    server_name astroshield.sdataplab.com;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    location /api/v1 {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    location /maneuvers {
        proxy_pass http://localhost:3001/maneuvers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    location /satellites {
        proxy_pass http://localhost:3001/satellites;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 5. Set Up SSL

```bash
sudo certbot --nginx -d astroshield.sdataplab.com --non-interactive --agree-tos --email admin@sdataplab.com --redirect
```

### 6. Start the Services

```bash
# Start the backend
cd backend
source venv/bin/activate
nohup python3 minimal_server.py > backend.log 2>&1 &

# Start the frontend
cd frontend
PORT=3000 nohup npm start > frontend.log 2>&1 &
```

## Troubleshooting

### 502 Bad Gateway Error

If you're seeing a 502 Bad Gateway error, check the following:

1. Make sure the backend is running:
   ```bash
   ps aux | grep minimal_server.py
   ```

2. Make sure the frontend is running:
   ```bash
   ps aux | grep "npm start"
   ```

3. Check Nginx error logs:
   ```bash
   sudo tail -f /var/log/nginx/error.log
   ```

4. Restart Nginx:
   ```bash
   sudo systemctl restart nginx
   ```

### SSL Certificate Issues

If you're having issues with SSL:

1. Verify the certificate:
   ```bash
   sudo certbot certificates
   ```

2. Renew the certificate:
   ```bash
   sudo certbot renew --dry-run
   ```

## Security Groups

Ensure your EC2 instance has the following ports open:
- Port 80 (HTTP)
- Port 443 (HTTPS)

## Contact

For support, please contact admin@sdataplab.com. 