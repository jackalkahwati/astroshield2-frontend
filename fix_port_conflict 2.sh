#!/bin/bash
set -e

echo "=== Port Conflict Resolution Script ==="
echo "This script will fix the port 3001 conflict on the EC2 instance"

# Create the script to run on the EC2 instance
cat > ec2_fix_ports.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Fixing port conflicts on EC2 ==="

# Find and kill process using port 3001
echo "Finding process using port 3001..."
PID=$(sudo lsof -ti:3001)
if [ -n "$PID" ]; then
  echo "Killing process $PID that's using port 3001..."
  sudo kill -9 $PID
  echo "Process terminated."
else
  echo "No process found using port 3001."
fi

# Also check for port 3000 (frontend) just to be safe
echo "Finding process using port 3000..."
PID=$(sudo lsof -ti:3000)
if [ -n "$PID" ]; then
  echo "Killing process $PID that's using port 3000..."
  sudo kill -9 $PID
  echo "Process terminated."
else
  echo "No process found using port 3000."
fi

# Restart Docker containers
echo "Restarting Docker containers..."
cd /home/stardrive/astroshield/deployment
sudo docker-compose down
sudo docker-compose up -d

# Verify services are running
echo "Verifying services..."
sudo docker ps

echo "=== Port conflicts resolved ==="
EOF

# Transfer the fix script to EC2
echo "Transferring fix script to EC2..."
chmod +x ec2_fix_ports.sh
scp ec2_fix_ports.sh astroshield:~/

# Run the fix script on EC2
echo "Running fix script on EC2..."
ssh astroshield "chmod +x ~/ec2_fix_ports.sh && ~/ec2_fix_ports.sh"

echo "Port conflicts have been resolved."
echo "Try accessing https://astroshield.sdataplab.com/ again." 