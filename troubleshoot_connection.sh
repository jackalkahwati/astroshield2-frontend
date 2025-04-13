#!/bin/bash
set -e

echo "=== Troubleshooting Port Forwarding Connection Issues ==="
echo "This script will help diagnose why the connection to localhost:8443 is failing"

# Step 1: Verify SSH connection
echo -e "\n== Step 1: Verifying SSH connection =="
echo "Checking SSH connection to astroshield..."
ssh -q astroshield exit
if [ $? -eq 0 ]; then
  echo "✅ SSH connection working!"
else
  echo "❌ SSH connection failed. Please check your SSH configuration."
  exit 1
fi

# Step 2: Check if required services are running on the EC2 instance
echo -e "\n== Step 2: Checking if web services are running on EC2 =="
echo "Checking if nginx/web servers are running on the EC2 instance..."
WEB_SERVICES=$(ssh astroshield "sudo docker ps | grep -E 'nginx|web|http' || echo 'No web services found'")
echo "$WEB_SERVICES"

# Step 3: Test port 80 first (HTTP) since it's more likely to work than HTTPS
echo -e "\n== Step 3: Testing if port 80 is accessible on EC2 =="
HTTP_TEST=$(ssh astroshield "curl -s -o /dev/null -w '%{http_code}' http://localhost:80 || echo 'Failed'")
echo "HTTP response from localhost:80 on EC2: $HTTP_TEST"

if [[ "$HTTP_TEST" == "200" || "$HTTP_TEST" == "301" || "$HTTP_TEST" == "302" ]]; then
  echo "✅ HTTP server is responding on EC2!"
else
  echo "❌ HTTP server is not responding on port 80. Let's check container logs:"
  LOGS=$(ssh astroshield "sudo docker logs deployment-nginx-1 2>&1 | tail -20")
  echo "$LOGS"
fi

# Step 4: Test local port availability
echo -e "\n== Step 4: Testing if ports 8080 and 8443 are already in use locally =="
if lsof -i:8080 > /dev/null 2>&1; then
  echo "❌ Port 8080 is already in use on your local machine."
else
  echo "✅ Port 8080 is available."
fi

if lsof -i:8443 > /dev/null 2>&1; then
  echo "❌ Port 8443 is already in use on your local machine."
else
  echo "✅ Port 8443 is available."
fi

# Step 5: Perform troubleshooting for macOS permissions
echo -e "\n== Step 5: Checking macOS permissions =="
echo "macOS might be blocking network access. Please check System Settings > Privacy & Security > Network."
echo "Make sure your terminal app has 'Local Network' permissions enabled."

# Step 6: Try a simpler port forwarding configuration
echo -e "\n== Step 6: Trying alternative port forwarding configuration =="
echo "Let's try a different approach with explicit settings:"
echo -e "\nOpen a new terminal window and run this command (keeping it open):\n"
echo -e "  ssh -v -L 8080:127.0.0.1:80 astroshield\n"
echo -e "Then try accessing http://127.0.0.1:8080 in your browser (note: using 127.0.0.1 instead of localhost)."
echo -e "\nIf that doesn't work, try with port 3010 (the direct frontend port):"
echo -e "  ssh -v -L 9010:127.0.0.1:3010 astroshield\n"
echo -e "Then access http://127.0.0.1:9010 in your browser."

# Step 7: Test port forwarding for diagnostic purposes
echo -e "\n== Step 7: Testing port forwarding with netcat =="
echo "Let's set up a test listener on the EC2 instance and try to connect to it locally."
echo "This will help us determine if port forwarding works at all."

echo "Starting a netcat listener on EC2 port 9999..."
ssh astroshield "nohup nc -l 9999 > /dev/null 2>&1 &" || echo "Could not start netcat listener"

echo "Setting up port forwarding to the netcat listener..."
echo "Opening connection for 5 seconds..."
ssh -f -N -L 9999:localhost:9999 astroshield sleep 5

echo "Testing connection to the forwarded port..."
if nc -z localhost 9999 2>/dev/null; then
  echo "✅ Test port forwarding successful! This confirms SSH tunneling works."
else
  echo "❌ Test port forwarding failed. This suggests a fundamental issue with SSH tunneling."
fi

# Step 8: Summary and recommendations
echo -e "\n== Step 8: Summary and Recommendations =="
echo "Based on the diagnostics, here are recommendations:"
echo "1. Try using 127.0.0.1 instead of localhost in your browser"
echo "2. Make sure no firewall is blocking local connections"
echo "3. Try a different local port (e.g., 9090 instead of 8080)"
echo "4. Check macOS Privacy & Security settings for your terminal application"
echo ""
echo "Alternative connection command:"
echo "ssh -v -N -L 9090:127.0.0.1:80 astroshield"
echo "Then browse to: http://127.0.0.1:9090" 