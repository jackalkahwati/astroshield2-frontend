#!/bin/bash
set -e

echo "Setting up SSH for EC2 deployment..."

# Create or append to SSH config
mkdir -p ~/.ssh
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

# Make sure config permissions are secure
chmod 600 ~/.ssh/config

echo "SSH config created successfully."

# You need to manually copy the key content to ~/.ssh/jackal_ec2_key
# and set permissions to 400 (chmod 400 ~/.ssh/jackal_ec2_key)
echo "IMPORTANT: Please copy your private key to ~/.ssh/jackal_ec2_key and run: chmod 400 ~/.ssh/jackal_ec2_key"

# Create a temporary directory for packaging
PACKAGE_DIR=$(mktemp -d)
echo "Created temporary directory for packaging: $PACKAGE_DIR"

# Copy UDL integration files
echo "Packaging UDL integration files..."
cp -r astroshield-integration-package "$PACKAGE_DIR/"

# Create deployment config
cat > "$PACKAGE_DIR/config.yaml" << 'EOT'
udl:
  base_url: "https://unifieddatalibrary.com"
  timeout: 30
  max_retries: 3
  sample_period: 0.5
  use_secure_messaging: true

kafka:
  bootstrap_servers: "kafka.astroshield.local:9092"
  security_protocol: "SASL_SSL"
  client_id: "udl_integration"

monitoring:
  enabled: true
  log_metrics: true
  prometheus_port: 8000
  metrics_interval: 60

topics:
  state_vector:
    udl_params: 
      maxResults: 100
    transform_func: "transform_state_vector"
    kafka_topic: "udl.state_vectors"
    polling_interval: 60
    cache_ttl: 300
  conjunction:
    udl_params: 
      maxResults: 50
    transform_func: "transform_conjunction"
    kafka_topic: "udl.conjunctions"
    polling_interval: 120
    cache_ttl: 600
  maneuver:
    udl_params: 
      maxResults: 25
    transform_func: "transform_maneuver"
    kafka_topic: "udl.maneuvers"
    polling_interval: 300
    cache_ttl: 1800
EOT

# Create README
cat > "$PACKAGE_DIR/README.md" << 'EOT'
# AstroShield UDL Integration

This package contains the UDL integration for AstroShield. It is designed to fetch data from the Unified Data Library (UDL) API and transform it to the AstroShield format for further processing.

## Installation

1. Install dependencies:
   ```
   pip install -e .
   ```

2. Set up environment variables (or use config file):
   ```
   export UDL_USERNAME=your_username
   export UDL_PASSWORD=your_password
   export KAFKA_BOOTSTRAP_SERVERS=kafka.astroshield.local:9092
   ```

3. Run the integration:
   ```
   python -m asttroshield.udl_integration --config ./config.yaml
   ```

## Configuration

See `config.yaml` for configuration options.
EOT

# Create requirements.txt
cat > "$PACKAGE_DIR/requirements.txt" << 'EOT'
requests>=2.25.0
pyyaml>=5.4.0
confluent-kafka>=1.5.0
python-dotenv>=0.15.0
prometheus-client>=0.9.0
EOT

# Create deployment script for remote execution
cat > "$PACKAGE_DIR/setup.sh" << 'EOT'
#!/bin/bash
set -e

echo "Setting up UDL Integration..."

# Create virtual environment
python3 -m venv udl_venv
source udl_venv/bin/activate

# Install dependencies and package
pip install -r requirements.txt
pip install -e astroshield-integration-package/

# Create .env file for credentials
cat > .env << 'EOF'
UDL_USERNAME=your_username
UDL_PASSWORD=your_password
KAFKA_BOOTSTRAP_SERVERS=kafka.astroshield.local:9092
KAFKA_SASL_USERNAME=astroshield
KAFKA_SASL_PASSWORD=your_kafka_password
EOF

echo "Installation complete. Edit .env with your credentials before starting."
echo "To start the integration, run: source udl_venv/bin/activate && python -m asttroshield.udl_integration --config ./config.yaml"
EOT

chmod +x "$PACKAGE_DIR/setup.sh"

# Create tarball
TARBALL="udl_integration_deploy.tar.gz"
echo "Creating tarball $TARBALL..."
tar -czf $TARBALL -C $PACKAGE_DIR .

echo "Deployment package created: $TARBALL"

echo "To deploy to EC2:"
echo "1. scp $TARBALL astroshield:~/"
echo "2. ssh astroshield 'mkdir -p ~/udl_integration && tar -xzf ~/udl_integration_deploy.tar.gz -C ~/udl_integration && cd ~/udl_integration && ./setup.sh'"

# Clean up temporary directory
rm -rf $PACKAGE_DIR
echo "Temporary directory removed."
