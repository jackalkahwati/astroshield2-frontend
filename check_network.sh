#!/bin/bash
set -e

echo "=== Checking EC2 Network Configuration ==="
echo "This script will check network settings and security configurations"

# Create the script to run on the EC2 instance
cat > ec2_network_check.sh << 'EOF'
#!/bin/bash
set -e

echo "=== EC2 Network Configuration Check ==="

# Check instance metadata and network info
echo "1. Instance information:"
echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "Public IP: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "None - This is a problem")"
echo "Private IP: $(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)"
echo "MAC: $(curl -s http://169.254.169.254/latest/meta-data/mac)"

# Check network interfaces
echo -e "\n2. Network interfaces:"
ifconfig || ip addr

# Check listening ports
echo -e "\n3. Checking listening ports:"
sudo netstat -tulpn | grep LISTEN || echo "netstat not found, trying ss"
sudo ss -tulpn | grep LISTEN || echo "Neither netstat nor ss available"

# Check security groups via instance metadata
echo -e "\n4. Checking instance metadata for security groups:"
curl -s http://169.254.169.254/latest/meta-data/security-groups

# Check iptables rules
echo -e "\n5. Checking iptables rules:"
sudo iptables -L -n

# Check if ports are accessible from the instance itself
echo -e "\n6. Testing local port access:"
echo "Port 80 on nginx container:"
curl -s -m 5 -o /dev/null -w "%{http_code}" http://localhost:80/ || echo " - Connection failed"
echo ""

echo "Port 443 on nginx container:"
curl -s -m 5 -o /dev/null -w "%{http_code}" -k https://localhost:443/ || echo " - Connection failed"
echo ""

# Check Docker network configuration
echo -e "\n7. Docker network configuration:"
sudo docker network ls
sudo docker network inspect deployment_astroshield-net

# Check if nginx is correctly binding to all interfaces
echo -e "\n8. Checking nginx binding:"
sudo docker exec deployment-nginx-1 nginx -T | grep -A5 "listen"

# Test outbound connectivity
echo -e "\n9. Testing outbound connectivity:"
curl -s -m 5 -o /dev/null -w "Google HTTP status: %{http_code}\n" http://www.google.com || echo " - Connection failed"
curl -s -m 5 -o /dev/null -w "Google HTTPS status: %{http_code}\n" https://www.google.com || echo " - Connection failed"

# Check for AWS specific networking issues
echo -e "\n10. Checking cloud-init logs for network issues:"
grep -i "network\|eth0\|dhcp\|route" /var/log/cloud-init.log | tail -20 || echo "No cloud-init logs found"

echo -e "\n=== Network Configuration Check Complete ==="
EOF

# Transfer the script to EC2
echo "Transferring script to EC2..."
chmod +x ec2_network_check.sh
scp ec2_network_check.sh astroshield:~/

# Run the script on EC2
echo "Running network check on EC2..."
ssh astroshield "chmod +x ~/ec2_network_check.sh && sudo ~/ec2_network_check.sh" 