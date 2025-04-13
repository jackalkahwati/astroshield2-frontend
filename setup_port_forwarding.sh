#!/bin/bash
set -e

echo "=== Setting Up SSH Port Forwarding ==="
echo "This script will help you access the application from your local machine"

# Check if the SSH connection is working
echo "Checking SSH connection to astroshield..."
ssh -q astroshield exit
if [ $? -eq 0 ]; then
  echo "SSH connection working! ✅"
else
  echo "SSH connection failed. Please check your SSH configuration."
  exit 1
fi

# Get current SSH connection details from ~/.ssh/config
echo "Checking your SSH configuration..."
if [ -f ~/.ssh/config ]; then
  echo "Found SSH config file."
  ASTROSHIELD_CONFIG=$(grep -A 10 "Host astroshield" ~/.ssh/config || echo "Host astroshield not found")
  echo "Your current astroshield SSH configuration:"
  echo "$ASTROSHIELD_CONFIG"
else
  echo "No SSH config file found. Using default SSH settings."
fi

# Instructions for port forwarding
echo -e "\n=== PORT FORWARDING INSTRUCTIONS ==="
echo "To access your application locally, open a new terminal and run this command:"
echo -e "\n  ssh -L 8080:localhost:80 astroshield\n"
echo "Then open http://localhost:8080 in your browser."
echo -e "\nTo access with HTTPS (may show certificate warnings), run:"
echo -e "\n  ssh -L 8443:localhost:443 astroshield\n"
echo "Then open https://localhost:8443 in your browser."
echo -e "\nKeep the terminal with the SSH connection open while browsing."
echo -e "Press Ctrl+C to close the SSH tunnel when you're done."
echo -e "\n=== END INSTRUCTIONS ===\n"

# Offer to start a port forwarding session
read -p "Would you like to start an HTTP port forwarding session now? (y/n): " START_FORWARDING

if [[ "$START_FORWARDING" == "y" || "$START_FORWARDING" == "Y" ]]; then
  echo "Starting port forwarding on port 8080. Access your application at http://localhost:8080"
  echo "Press Ctrl+C to stop."
  ssh -L 8080:localhost:80 astroshield
else
  echo "You can start port forwarding manually using the command above."
fi 