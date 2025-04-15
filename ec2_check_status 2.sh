#!/bin/bash
set -e

echo "=== Verifying AstroShield Deployment on EC2 ==="

# Check if Docker containers are running
echo "1. Checking Docker containers:"
sudo docker ps

# Check Docker container logs
echo "2. Checking Nginx logs:"
sudo docker logs deployment-nginx-1

# Check if the frontend container is accessible
echo "3. Testing frontend container accessibility:"
curl -s http://localhost:3010 | head -n 10

# Check if Nginx can access the frontend
echo "4. Testing Nginx to frontend connection:"
curl -k -s https://localhost | head -n 10

# Check DNS resolution
echo "5. Checking DNS resolution for the domain:"
nslookup astroshield.sdataplab.com || echo "DNS lookup failed"

# Check if port 443 is open
echo "6. Checking if port 443 is accessible:"
nc -zv astroshield.sdataplab.com 443 || echo "Port 443 not accessible"

# Check certificate information
echo "7. Checking SSL certificate:"
echo | openssl s_client -connect localhost:443 2>/dev/null | grep "subject\|issuer"

# Check for any firewall rules that might be blocking traffic
echo "8. Checking firewall rules:"
sudo iptables -L | grep -E "(ACCEPT|DROP|REJECT)" | head -n 10

echo "=== Deployment Status Check Complete ==="
