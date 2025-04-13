#!/bin/bash
set -e

echo "=== AstroShield Environment Setup and Configuration ==="
echo "This script will set up and test the AstroShield environment"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Set up Python virtual environment
setup_python_env() {
  echo "=== Setting up Python virtual environment ==="
  python3 -m venv venv
  source venv/bin/activate
  
  echo "Installing base requirements..."
  pip install --upgrade pip
  pip install -r requirements.txt
  
  if [ -f ml/requirements.txt ]; then
    echo "Installing ML requirements..."
    pip install -r ml/requirements.txt
  fi
  
  echo "Python environment set up with venv activated"
  python --version
  pip --version
}

# Create UDL configuration file
setup_udl_config() {
  echo "=== Setting up UDL Configuration ==="
  
  if [ -f .env ]; then
    echo "Found existing .env file. Do you want to update it? (y/n)"
    read update_env
    if [ "$update_env" != "y" ]; then
      echo "Keeping existing .env file"
      return
    fi
  fi
  
  echo "Please enter your UDL credentials:"
  echo -n "UDL_USERNAME: "
  read udl_username
  echo -n "UDL_PASSWORD: "
  read -s udl_password
  echo
  echo -n "UDL_BASE_URL (default: https://unifieddatalibrary.com/udl/api/v1): "
  read udl_base_url
  
  # Use default if empty
  if [ -z "$udl_base_url" ]; then
    udl_base_url="https://unifieddatalibrary.com/udl/api/v1"
  fi
  
  # Create or update .env file
  cat > .env << EOT
# UDL Configuration
UDL_USERNAME=${udl_username}
UDL_PASSWORD=${udl_password}
UDL_BASE_URL=${udl_base_url}

# Kafka Configuration (if needed)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_SECURITY_PROTOCOL=PLAINTEXT
EOT

  echo "UDL Configuration saved to .env file"
}

# Test UDL authentication
test_udl_auth() {
  echo "=== Testing UDL Authentication ==="
  
  if [ ! -f .env ]; then
    echo "No .env file found. Please run setup_udl_config first."
    return 1
  fi
  
  # Find the test_udl_auth.py script
  if [ -f test_udl_auth.py ]; then
    TEST_SCRIPT="test_udl_auth.py"
  elif [ -f astroshield-integration-package/src/asttroshield/udl_integration/test_udl_auth.py ]; then
    TEST_SCRIPT="astroshield-integration-package/src/asttroshield/udl_integration/test_udl_auth.py"
  else
    echo "Could not find test_udl_auth.py in the current directory or in astroshield-integration-package."
    return 1
  fi
  
  echo "Running UDL authentication test using ${TEST_SCRIPT}..."
  python ${TEST_SCRIPT}
  
  if [ $? -eq 0 ]; then
    echo "UDL Authentication successful!"
    return 0
  else
    echo "UDL Authentication failed. Please check your credentials in .env"
    return 1
  fi
}

# Check subsystem0 data ingestion
check_subsystem0() {
  echo "=== Checking Subsystem 0 (Data Ingestion) ==="
  
  if [ -f src/example_app.py ]; then
    echo "Found example_app.py. Running subsystem 0 test..."
    python src/example_app.py --subsystem 0 &
    SUBSYSTEM_PID=$!
    
    echo "Subsystem 0 is running with PID ${SUBSYSTEM_PID}"
    echo "Letting it run for 10 seconds to test data ingestion..."
    sleep 10
    
    # Stop the subsystem
    kill ${SUBSYSTEM_PID}
    echo "Subsystem 0 test complete"
    return 0
  else
    echo "Could not find example_app.py to test subsystem 0"
    return 1
  fi
}

# Start the API server
start_api() {
  echo "=== Starting API Server ==="
  
  if [ -f backend/app/server.py ]; then
    echo "Found backend/app/server.py. Starting API server..."
    cd backend
    uvicorn app.server:app --reload --host 0.0.0.0 --port 5000 &
    API_PID=$!
    
    echo "API is running with PID ${API_PID}"
    echo "You can access the API at http://localhost:5000/api/docs"
    
    # Keep a reference to the PID so we can stop it later
    echo ${API_PID} > .api.pid
    
    return 0
  elif [ -f app.py ]; then
    echo "Found app.py. Starting API server..."
    uvicorn app:create_app --reload --host 0.0.0.0 --port 5000 &
    API_PID=$!
    
    echo "API is running with PID ${API_PID}"
    echo "You can access the API at http://localhost:5000/api/docs"
    
    # Keep a reference to the PID so we can stop it later
    echo ${API_PID} > .api.pid
    
    return 0
  else
    echo "Could not find API server file to start"
    return 1
  fi
}

# Test the API
test_api() {
  echo "=== Testing API Server ==="
  
  # Wait a bit for the API to start
  sleep 5
  
  # Test API health endpoint
  echo "Testing API health endpoint..."
  curl -s http://localhost:5000/api/health || curl -s http://localhost:5000/health
  
  echo -e "\nAPI test complete"
}

# Main execution
echo "What would you like to do?"
echo "1. Set up Python environment"
echo "2. Configure UDL"
echo "3. Test UDL authentication"
echo "4. Check Subsystem 0 (Data Ingestion)"
echo "5. Start API server"
echo "6. Test API"
echo "7. Run all steps in sequence"
echo "8. Stop API server"
echo -n "Enter your choice (1-8): "
read choice

case $choice in
  1)
    setup_python_env
    ;;
  2)
    setup_udl_config
    ;;
  3)
    test_udl_auth
    ;;
  4)
    check_subsystem0
    ;;
  5)
    start_api
    ;;
  6)
    test_api
    ;;
  7)
    setup_python_env
    setup_udl_config
    test_udl_auth
    check_subsystem0
    start_api
    test_api
    ;;
  8)
    if [ -f .api.pid ]; then
      API_PID=$(cat .api.pid)
      echo "Stopping API server with PID ${API_PID}..."
      kill ${API_PID}
      rm .api.pid
      echo "API server stopped"
    else
      echo "No API server PID file found"
    fi
    ;;
  *)
    echo "Invalid choice"
    ;;
esac

echo "=== Script completed ===" 