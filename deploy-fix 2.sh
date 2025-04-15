#!/bin/bash

# Exit on error
set -e

# Load configuration (optional - you'll need deploy-config.sh to exist)
if [ -f deploy-config.sh ]; then
  source deploy-config.sh
else
  # Default values if config file doesn't exist
  EC2_USER="ec2-user"
  EC2_HOST="astroshield.sdataplab.com"
  SSH_KEY_PATH="~/.ssh/id_rsa"  # Replace with actual path
  REMOTE_DIR="/home/ec2-user/astroshield"
fi

echo "========================================="
echo "AstroShield Nginx Config Update Script"
echo "========================================="
echo "Updating config on: $EC2_HOST"
echo "Using SSH key: $SSH_KEY_PATH"
echo "========================================="

# Verify SSH key exists
if [ ! -f "${SSH_KEY_PATH/#\~/$HOME}" ]; then
  echo "Error: SSH key not found at $SSH_KEY_PATH"
  echo "Please update SSH_KEY_PATH to point to a valid SSH key."
  exit 1
fi

# Create a temporary directory
mkdir -p /tmp/astroshield-fix

# Copy the config file
cp ssl-config.conf /tmp/astroshield-fix/

# Create the remote update script
cat > /tmp/astroshield-fix/update.sh << 'EOF'
#!/bin/bash

# Ensure SSL directory exists
sudo mkdir -p /etc/nginx/ssl

# Copy configuration file
sudo cp ssl-config.conf /etc/nginx/conf.d/astroshield.conf

# Verify configuration
echo "Checking Nginx configuration..."
sudo nginx -t

# If configuration test is successful, restart Nginx
if [ $? -eq 0 ]; then
    echo "Nginx configuration is valid. Restarting Nginx..."
    sudo systemctl restart nginx
    echo "Nginx restarted successfully!"
else
    echo "Error in Nginx configuration. Please check the syntax."
    exit 1
fi

# Check if astroshield service is running
if sudo systemctl is-active --quiet astroshield; then
    echo "AstroShield service is running."
else
    echo "AstroShield service is not running, starting it..."
    sudo systemctl start astroshield
fi

# Check Nginx status after restart
sudo systemctl status nginx

echo "Update complete! Configuration deployed successfully."
EOF

# Make the script executable
chmod +x /tmp/astroshield-fix/update.sh

# Create a tarball
tar -czf astroshield-fix.tar.gz -C /tmp/astroshield-fix .

# Upload to EC2
echo "Uploading to EC2..."
ssh -i "${SSH_KEY_PATH/#\~/$HOME}" -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST "mkdir -p /tmp/astroshield-fix"
scp -i "${SSH_KEY_PATH/#\~/$HOME}" -o StrictHostKeyChecking=no astroshield-fix.tar.gz $EC2_USER@$EC2_HOST:/tmp/

# Deploy on EC2
echo "Deploying on EC2..."
ssh -i "${SSH_KEY_PATH/#\~/$HOME}" -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST << EOF
cd /tmp
tar -xzf astroshield-fix.tar.gz -C /tmp/astroshield-fix
cd /tmp/astroshield-fix
chmod +x update.sh
./update.sh
EOF

# Clean up
rm -rf /tmp/astroshield-fix astroshield-fix.tar.gz

echo "==========================================================="
echo "Update completed! Please check https://${EC2_HOST} now."
echo "==========================================================="