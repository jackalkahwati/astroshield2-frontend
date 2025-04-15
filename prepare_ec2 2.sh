#!/bin/bash
set -e

echo "=== EC2 Preparation Script for AstroShield Deployment ==="
echo "This script will back up existing deployment and prepare for a new installation"

# Create the script to run on the EC2 instance
cat > ec2_prepare.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Preparing EC2 environment for AstroShield deployment ==="

# Create backup timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="$HOME/astroshield_backup_$TIMESTAMP"
echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Back up existing Docker container information
echo "Backing up Docker container information..."
docker ps -a > "$BACKUP_DIR/docker_containers.txt"
docker images > "$BACKUP_DIR/docker_images.txt"

# Back up existing Docker Compose files
if [ -d "/home/ec2-user/astroshield" ]; then
  echo "Backing up existing deployment from /home/ec2-user/astroshield"
  if [ -f "/home/ec2-user/astroshield/docker-compose.yml" ]; then
    cp /home/ec2-user/astroshield/docker-compose.yml "$BACKUP_DIR/"
  fi
  find /home/ec2-user/astroshield -name "*.conf" -exec cp {} "$BACKUP_DIR/" \;
fi

# Check if Nginx is running as a system service
if systemctl is-active --quiet nginx; then
  echo "System Nginx service is running. Stopping it..."
  sudo systemctl stop nginx
  sudo systemctl disable nginx
  echo "Nginx service stopped and disabled"
fi

# Check for ports in use
echo "Checking for port conflicts..."
for port in 80 443 3000 3001 5432; do
  if netstat -tuln | grep ":$port " > /dev/null; then
    echo "Port $port is in use. Identifying process..."
    pid=$(sudo lsof -t -i:$port)
    if [ -n "$pid" ]; then
      echo "Process using port $port: $(ps -p $pid -o comm=)"
    fi
  fi
done

# Create our deployment directory with proper permissions
echo "Creating deployment directory..."
mkdir -p /home/stardrive/astroshield
chmod 755 /home/stardrive/astroshield

echo "=== EC2 preparation complete ==="
echo "Ready for new deployment to /home/stardrive/astroshield"
EOF

# Transfer the preparation script to EC2
echo "Transferring preparation script to EC2..."
chmod +x ec2_prepare.sh
scp ec2_prepare.sh astroshield:~/

# Run the preparation script on EC2
echo "Running preparation script on EC2..."
ssh astroshield "chmod +x ~/ec2_prepare.sh && ~/ec2_prepare.sh"

echo "EC2 environment has been prepared for deployment."
echo "You can now run ./simplified_deploy.sh to deploy the application." 