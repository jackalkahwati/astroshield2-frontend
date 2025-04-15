#!/bin/bash
set -e

# Create directory for SSH config
mkdir -p ~/.ssh

# Add SSH config
cat >> ~/.ssh/config << 'EOT'

Host ub
  HostName ec2-3-30-215-137.us-gov-west-1.compute.amazonaws.com
  User jackal
  IdentityFile ~/.ssh/jackal_ec2_key
  IdentitiesOnly yes

Host astroshield
  HostName 10.0.11.100
  User stardrive
  IdentityFile ~/.ssh/jackal_ec2_key
  IdentitiesOnly yes
  ProxyJump ub
EOT

# Set permissions
chmod 600 ~/.ssh/config

echo "SSH config created successfully." 