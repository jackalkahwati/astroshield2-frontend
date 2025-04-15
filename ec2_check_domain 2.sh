#!/bin/bash
set -e

echo "=== Checking Domain and AWS Configuration ==="

# Get instance metadata
echo "1. Getting EC2 instance information:"
curl -s http://169.254.169.254/latest/meta-data/instance-id
echo ""
curl -s http://169.254.169.254/latest/meta-data/public-ipv4
echo ""

# Check public IP resolution
echo "2. Checking public IP address:"
public_ip=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo "Public IP: $public_ip"

# Check if domain points to this IP
echo "3. Checking domain resolution:"
domain_ip=$(dig +short astroshield.sdataplab.com)
echo "Domain IP: $domain_ip"

if [ "$public_ip" = "$domain_ip" ]; then
  echo "Domain resolves to this instance's IP ✓"
else
  echo "Domain does not resolve to this instance's IP ✗"
  echo "This could be causing the 502 error - DNS not properly configured"
fi

# Check if security groups allow traffic
echo "4. Testing if specific ports are open publicly:"
echo "HTTP (80):"
curl -s -m 5 -o /dev/null -w "%{http_code}" http://$public_ip/ || echo " - Connection failed"
echo ""

echo "HTTPS (443):"
curl -s -m 5 -o /dev/null -w "%{http_code}" -k https://$public_ip/ || echo " - Connection failed"
echo ""

# Check security groups (requires aws cli, may not work if not configured)
echo "5. Trying to check security groups (may not work if AWS CLI not configured):"
instance_id=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 describe-instance-attribute --instance-id $instance_id --attribute groupSet 2>/dev/null || echo "AWS CLI not configured"

echo "=== Domain and AWS Configuration Check Complete ==="

# If public access is failing, try to check if the server is actually running locally
echo "6. Final check - testing frontend directly:"
curl -s localhost:3010 | grep -o "<title>.*</title>"

echo "7. Testing Nginx access to frontend:"
curl -s -k https://localhost | grep -o "<title>.*</title>"

