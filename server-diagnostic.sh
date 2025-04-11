#!/bin/bash

# Output header
echo "========================================================"
echo "AstroShield Server Diagnostic and Fix Script"
echo "========================================================"

# Check if services are running
echo -e "\n=== 1. Checking system processes ==="
echo "Looking for running services on ports 3000 and 3001:"
sudo netstat -tlpn | grep -E ':3000|:3001'

echo -e "\n=== 2. Checking Docker containers ==="
if command -v docker &> /dev/null; then
    echo "Listing all Docker containers:"
    sudo docker ps -a
else
    echo "Docker not installed or not available"
fi

echo -e "\n=== 3. Checking systemd services ==="
echo "Checking AstroShield service status:"
sudo systemctl status astroshield 2>/dev/null || echo "AstroShield service not found"

echo "Checking individual services:"
for service in astroshield-frontend astroshield-backend; do
    sudo systemctl status $service 2>/dev/null || echo "$service service not found"
done

echo -e "\n=== 4. Checking log files ==="
echo "Nginx error log (last 20 lines):"
sudo tail -n 20 /var/log/nginx/error.log 2>/dev/null || echo "Nginx error log not found"

echo -e "\nNginx access log (last 20 lines):"
sudo tail -n 20 /var/log/nginx/access.log 2>/dev/null || echo "Nginx access log not found"

echo -e "\nBackend log (if available):"
sudo find /home -name "backend.log" -type f -exec sudo tail -n 20 {} \; 2>/dev/null || echo "Backend log not found"

echo -e "\nFrontend log (if available):"
sudo find /home -name "frontend.log" -type f -exec sudo tail -n 20 {} \; 2>/dev/null || echo "Frontend log not found"

echo -e "\n=== 5. Checking file structure ==="
echo "Looking for minimal_server.py:"
sudo find /home -name "minimal_server.py" -type f 2>/dev/null

echo -e "\n=== 6. Attempting to fix issues ==="
echo "Would you like to:"
echo "1. Restart the backend service"
echo "2. Start the backend service manually"
echo "3. Fix Docker container issues"
echo "4. Exit without making changes"
read -p "Enter your choice [1-4]: " choice

case $choice in
    1)
        echo "Restarting AstroShield services..."
        sudo systemctl restart astroshield 2>/dev/null || echo "Could not restart main service"
        sudo systemctl restart astroshield-backend 2>/dev/null || echo "Could not restart backend service"
        sudo systemctl restart astroshield-frontend 2>/dev/null || echo "Could not restart frontend service"
        echo "Restarting Nginx..."
        sudo systemctl restart nginx
        ;;
    2)
        echo "Starting backend service manually..."
        
        # Find minimal_server.py and start it
        MIN_SERVER=$(sudo find /home -name "minimal_server.py" -type f | head -n 1)
        if [ -n "$MIN_SERVER" ]; then
            DIR=$(dirname "$MIN_SERVER")
            echo "Found minimal_server.py at $MIN_SERVER"
            echo "Stopping any existing processes..."
            sudo pkill -f "python.*minimal_server.py" || echo "No running processes found"
            
            echo "Starting minimal_server.py in the background..."
            cd $DIR
            sudo nohup python3 minimal_server.py > backend.log 2>&1 &
            echo $! > backend.pid
            echo "Started backend with PID: $!"
        else
            echo "Could not find minimal_server.py"
        fi
        ;;
    3)
        echo "Fixing Docker container issues..."
        if command -v docker &> /dev/null; then
            echo "Restarting Docker containers..."
            sudo docker-compose down || echo "docker-compose down failed"
            sudo docker-compose up -d || echo "docker-compose up failed"
        else
            echo "Docker not installed or not available"
        fi
        ;;
    4)
        echo "Exiting without making changes"
        ;;
    *)
        echo "Invalid choice. Exiting."
        ;;
esac

echo -e "\n=== 7. Final status check ==="
echo "Checking if services are now running on ports 3000 and 3001:"
sudo netstat -tlpn | grep -E ':3000|:3001'

echo -e "\nChecking Nginx status:"
sudo systemctl status nginx | head -n 5

echo -e "\n========================================================"
echo "Diagnostic complete. Please check if the website is accessible now."
echo "========================================================"