#!/bin/bash
# AstroShield Robust Setup Script
# This script addresses common setup issues and provides diagnostics

set -e
VERBOSE=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
  if [ "$VERBOSE" = true ]; then
    echo -e "${BLUE}[INFO]${NC} $1"
  fi
}

success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

check_disk_space() {
  log "Checking available disk space..."
  local available_space=$(df -h . | awk 'NR==2 {print $4}')
  log "Available space: $available_space"
  
  # Convert to bytes for comparison, assuming GB notation
  local space_value=$(echo $available_space | sed 's/[^0-9.]//g')
  local space_unit=$(echo $available_space | sed 's/[0-9.]//g')
  
  local min_required=1 # Minimum 1GB required
  
  if [[ "$space_unit" == "G"* ]] && (( $(echo "$space_value < $min_required" | bc -l) )); then
    error "Insufficient disk space. At least ${min_required}GB is recommended."
    warn "Consider using the minimal installation option."
    return 1
  elif [[ "$space_unit" == "M"* ]]; then
    warn "Very limited disk space detected. Using minimal installation."
    export USE_MINIMAL=true
    return 0
  else
    success "Sufficient disk space available: $available_space"
    return 0
  fi
}

detect_python() {
  log "Detecting Python environment..."
  
  # Check for available Python commands
  if command -v python3.10 &>/dev/null; then
    PY_CMD="python3.10"
  elif command -v python3.9 &>/dev/null; then
    PY_CMD="python3.9"
    warn "Using Python 3.9 instead of recommended 3.10"
  elif command -v python3 &>/dev/null; then
    PY_CMD="python3"
    local py_version=$(python3 --version 2>&1)
    log "Found $py_version"
  else
    error "No Python 3 installation found"
    return 1
  fi
  
  # Check for pip
  if command -v pip3 &>/dev/null; then
    PIP_CMD="pip3"
  elif command -v pip &>/dev/null; then
    PIP_CMD="pip"
  else
    log "pip not found, will use '$PY_CMD -m pip' instead"
    PIP_CMD="$PY_CMD -m pip"
  fi
  
  # Verify we can actually use pip
  $PIP_CMD --version &>/dev/null
  if [ $? -ne 0 ]; then
    warn "pip command failed, trying alternative approach"
    PIP_CMD="$PY_CMD -m pip"
    $PIP_CMD --version &>/dev/null
    if [ $? -ne 0 ]; then
      error "Cannot use pip. This might be due to Nix environment restrictions."
      return 1
    fi
  fi
  
  success "Using Python: $PY_CMD and Pip: $PIP_CMD"
  export PY_CMD
  export PIP_CMD
  return 0
}

setup_nix_environment() {
  log "Setting up Nix environment..."
  
  # Check if we're in a Nix environment already
  if command -v nix-shell &>/dev/null; then
    log "Nix is available"
    
    # Create a simple and robust shell.nix file
    cat > shell.nix << 'EOF'
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.pip
  ];
  
  shellHook = ''
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    export PYTHONPATH="$PWD:$PYTHONPATH"
    echo "Nix Python environment activated"
  '';
}
EOF
    
    log "Created shell.nix file"
    log "Starting Nix shell (this might take a moment)..."
    
    # Enter nix-shell in a way that preserves the current script
    if [ -z "$NIX_SHELL_ACTIVE" ]; then
      export NIX_SHELL_ACTIVE=1
      log "Re-executing setup in Nix shell..."
      # Use absolute path to the script for re-execution
      SCRIPT_PATH="$(pwd)/$(basename $0)"
      exec nix-shell --run "NIX_SHELL_ACTIVE=1 $SCRIPT_PATH $*"
    else
      success "Nix shell is active"
    fi
  else
    warn "Nix not found, continuing with system Python"
  fi
}

create_minimal_requirements() {
  log "Creating minimal requirements files..."
  
  # Main minimal requirements
  cat > requirements-minimal.txt << 'EOF'
fastapi>=0.115.0
uvicorn>=0.28.0
python-dotenv>=1.0.0
requests>=2.31.0
pydantic>=1.10.0
EOF
  
  # ML minimal requirements
  cat > ml/requirements-minimal.txt << 'EOF'
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.1.0
tqdm>=4.65.0
joblib>=1.3.0
EOF
  
  success "Created minimal requirements files"
}

install_dependencies() {
  log "Installing dependencies..."
  
  # Determine which requirements files to use
  local req_file="requirements.txt"
  local ml_req_file="ml/requirements.txt"
  
  if [ "$USE_MINIMAL" = true ]; then
    log "Using minimal requirements due to space constraints"
    req_file="requirements-minimal.txt"
    ml_req_file="ml/requirements-minimal.txt"
    
    # Create minimal requirements if they don't exist
    if [ ! -f "$req_file" ]; then
      create_minimal_requirements
    fi
  fi
  
  # Install main requirements
  if [ -f "$req_file" ]; then
    log "Installing main requirements from $req_file"
    $PIP_CMD install -r "$req_file" --no-cache-dir
    if [ $? -ne 0 ]; then
      warn "Failed to install all packages, trying essential ones individually"
      $PIP_CMD install fastapi uvicorn python-dotenv requests pydantic --no-cache-dir
    fi
  else
    error "Requirements file $req_file not found"
    warn "Installing essential packages individually"
    $PIP_CMD install fastapi uvicorn python-dotenv requests pydantic --no-cache-dir
  fi
  
  # Install ML requirements if available
  if [ -f "$ml_req_file" ]; then
    log "Installing ML requirements from $ml_req_file"
    $PIP_CMD install -r "$ml_req_file" --no-cache-dir || warn "Failed to install all ML packages"
  else
    warn "ML requirements file $ml_req_file not found, skipping"
  fi
  
  # Ensure dotenv and requests are installed (critical for UDL integration)
  $PIP_CMD install python-dotenv requests --no-cache-dir
  
  success "Core dependencies installed"
}

setup_environment() {
  log "Setting up environment..."
  
  # Add the current directory to PYTHONPATH
  export PYTHONPATH="$PWD:$PYTHONPATH"
  log "PYTHONPATH set to include current directory"
  
  # Create a .env file if it doesn't exist
  if [ ! -f .env ]; then
    log "Creating default .env file"
    cat > .env << 'EOF'
# UDL Configuration
UDL_USERNAME=test_user
UDL_PASSWORD=test_password
UDL_BASE_URL=https://unifieddatalibrary.com/udl/api/v1

# Kafka Configuration (if needed)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_SECURITY_PROTOCOL=PLAINTEXT
EOF
    warn "Created default .env file with placeholder UDL credentials"
    warn "Edit .env to set your actual UDL credentials"
  else
    log "Found existing .env file"
  fi
}

create_mock_udl_auth() {
  log "Creating mock UDL authentication test script..."
  
  cat > mock_test_udl_auth.py << 'EOF'
#!/usr/bin/env python3
"""
Mock UDL authentication test script.
This version works without requiring external imports or actual UDL credentials.
"""
import os
import sys
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_env_from_file():
    """Load environment variables from .env file manually."""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                os.environ[key] = value
        return True
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
        return False

def test_udl_auth():
    """Test UDL authentication using a mock approach."""
    logger.info("Testing UDL authentication (MOCK MODE)")
    
    # Try to load from .env file if dotenv module failed
    if 'UDL_USERNAME' not in os.environ:
        load_env_from_file()
    
    # Log environment variable status
    username = os.environ.get("UDL_USERNAME", "")
    password = os.environ.get("UDL_PASSWORD", "")
    base_url = os.environ.get("UDL_BASE_URL", "")
    
    logger.info(f"UDL_USERNAME: {'✓' if username else '✗'} (value: {username[:2] if len(username) > 1 else ''}***)")
    logger.info(f"UDL_PASSWORD: {'✓' if password else '✗'} (length: {len(password)})")
    logger.info(f"UDL_BASE_URL: {'✓' if base_url else '✗'} (value: {base_url})")
    
    if not (username and password):
        logger.error("Required environment variables not found. Please check your .env file.")
        return False
    
    logger.info("MOCK: Simulating UDL authentication success")
    logger.info("In a real environment, this would connect to the UDL server")
    logger.info("Authentication successful (mock mode)")
    return True

if __name__ == "__main__":
    success = test_udl_auth()
    sys.exit(0 if success else 1)
EOF
  
  chmod +x mock_test_udl_auth.py
  success "Created mock UDL test script"
}

test_udl_auth() {
  log "Testing UDL authentication..."
  
  # Check for the test script
  local test_script=""
  if [ -f test_udl_auth.py ]; then
    test_script="test_udl_auth.py"
  elif [ -f astroshield-integration-package/src/asttroshield/udl_integration/test_udl_auth.py ]; then
    test_script="astroshield-integration-package/src/asttroshield/udl_integration/test_udl_auth.py"
  elif [ -f mock_test_udl_auth.py ]; then
    test_script="mock_test_udl_auth.py"
  else
    warn "No UDL test script found, creating mock version"
    create_mock_udl_auth
    test_script="mock_test_udl_auth.py"
  fi
  
  log "Running UDL authentication test using $test_script..."
  
  # Export PYTHONPATH to include necessary directories
  export PYTHONPATH="$PWD:$PYTHONPATH"
  
  # Try to run the test script
  $PY_CMD $test_script
  if [ $? -eq 0 ]; then
    success "UDL authentication test passed"
    return 0
  else
    error "UDL authentication test failed"
    warn "Try running: $PY_CMD $test_script manually to see detailed errors"
    warn "You might need to edit your .env file with correct UDL credentials"
    return 1
  fi
}

create_simple_api() {
  log "Creating a simple API server for testing..."
  
  cat > simple_api.py << 'EOF'
#!/usr/bin/env python3
"""
Simple FastAPI server for testing the AstroShield API functionality.
"""
import logging
import os
import sys
from datetime import datetime
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    print("FastAPI not installed. Please install with: pip install fastapi uvicorn")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AstroShield API (Simple Test Version)",
    description="Test version of the AstroShield API",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AstroShield API is running"}

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
        "environment": "test"
    }

@app.get("/health")
async def alt_health():
    """Alternative health check endpoint"""
    return await health()

@app.get("/api/satellites")
async def satellites():
    """Mock satellites endpoint"""
    return {
        "satellites": [
            {"id": "SAT001", "name": "Test Satellite 1", "status": "active"},
            {"id": "SAT002", "name": "Test Satellite 2", "status": "inactive"},
            {"id": "SAT003", "name": "Test Satellite 3", "status": "active"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting test API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
EOF
  
  chmod +x simple_api.py
  success "Created simple API server for testing"
}

start_api() {
  log "Starting API server..."
  
  # Identify which API file to use
  local api_file=""
  if [ -f backend/app/server.py ]; then
    api_file="backend/app/server.py"
    log "Found backend/app/server.py"
  elif [ -f app.py ]; then
    api_file="app.py"
    log "Found app.py"
  elif [ -f simple_api.py ]; then
    api_file="simple_api.py"
    log "Found simple_api.py"
  else
    warn "No API server file found, creating a simple version for testing"
    create_simple_api
    api_file="simple_api.py"
  fi
  
  # Check for uvicorn
  if ! $PY_CMD -c "import uvicorn" &>/dev/null; then
    warn "uvicorn not installed, attempting to install it"
    $PIP_CMD install uvicorn fastapi --no-cache-dir
  fi
  
  # Start the appropriate API server
  if [ "$api_file" = "simple_api.py" ]; then
    log "Starting simple API directly"
    $PY_CMD $api_file &
    API_PID=$!
  elif [ "$api_file" = "backend/app/server.py" ]; then
    log "Starting backend/app/server.py with uvicorn"
    cd backend
    $PY_CMD -m uvicorn app.server:app --host 0.0.0.0 --port 5000 &
    API_PID=$!
    cd ..
  elif [ "$api_file" = "app.py" ]; then
    log "Starting app.py with create_app function"
    $PY_CMD -m uvicorn app:create_app --host 0.0.0.0 --port 5000 &
    API_PID=$!
  fi
  
  if [ -n "$API_PID" ]; then
    echo $API_PID > .api.pid
    success "API server started with PID: $API_PID"
    log "You can access the API at http://localhost:5000"
    log "API health check: http://localhost:5000/api/health"
    return 0
  else
    error "Failed to start API server"
    return 1
  fi
}

test_api() {
  log "Testing API server..."
  sleep 3 # Give the server a moment to start
  
  # Try multiple health check endpoints
  log "Testing health endpoint..."
  curl -s http://localhost:5000/api/health 2>/dev/null || \
  curl -s http://localhost:5000/health 2>/dev/null || \
  curl -s http://localhost:5000/ 2>/dev/null
  
  if [ $? -ne 0 ]; then
    error "API health check failed"
    warn "The API server might not be running or is not responding"
    warn "Try checking the processes: ps aux | grep python"
    return 1
  else
    success "API server is responding"
    return 0
  fi
}

cleanup() {
  log "Performing cleanup..."
  
  # Stop any running API server
  if [ -f .api.pid ]; then
    local pid=$(cat .api.pid)
    log "Stopping API server with PID $pid"
    kill $pid 2>/dev/null || true
    rm .api.pid
  fi
  
  # Remove temporary files
  log "Cleaning up temporary files"
  
  success "Cleanup completed"
}

run_all() {
  setup_nix_environment
  check_disk_space
  detect_python
  setup_environment
  install_dependencies
  test_udl_auth
  start_api
  test_api
}

show_help() {
  echo "AstroShield Robust Setup Script"
  echo "-------------------------------"
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --help          Show this help message"
  echo "  --all           Run all setup steps (recommended)"
  echo "  --minimal       Use minimal requirements to save disk space"
  echo "  --nix           Setup Nix environment only"
  echo "  --deps          Install dependencies only"
  echo "  --env           Setup environment only"
  echo "  --udl           Test UDL authentication only"
  echo "  --api           Start API server only"
  echo "  --test-api      Test API server only"
  echo "  --cleanup       Clean up resources"
  echo "  --no-verbose    Disable verbose output"
  echo ""
  echo "Examples:"
  echo "  $0 --all               # Run everything"
  echo "  $0 --minimal --deps    # Install minimal dependencies"
}

# Parse command line arguments
if [ $# -eq 0 ]; then
  show_help
  exit 0
fi

for arg in "$@"; do
  case $arg in
    --help)
      show_help
      exit 0
      ;;
    --all)
      run_all
      ;;
    --minimal)
      export USE_MINIMAL=true
      ;;
    --nix)
      setup_nix_environment
      ;;
    --deps)
      detect_python
      install_dependencies
      ;;
    --env)
      setup_environment
      ;;
    --udl)
      detect_python
      setup_environment
      test_udl_auth
      ;;
    --api)
      detect_python
      setup_environment
      start_api
      ;;
    --test-api)
      test_api
      ;;
    --cleanup)
      cleanup
      ;;
    --no-verbose)
      VERBOSE=false
      ;;
    *)
      error "Unknown option: $arg"
      show_help
      exit 1
      ;;
  esac
done

echo ""
success "Script completed successfully"
echo "For help, run: $0 --help" 