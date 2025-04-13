#!/bin/bash
set -e

echo "=== AstroShield Environment Setup for macOS ==="
echo "This script will set up a complete development environment for AstroShield on macOS"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

echo "=== Step 1: Installing Homebrew if not already installed ==="
if ! command_exists brew; then
  echo "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
  echo "Homebrew is already installed. Updating..."
  brew update
fi

echo "=== Step 2: Installing Essential Packages ==="
brew install \
  git \
  curl \
  wget \
  openssl \
  || echo "Some packages may have failed to install. Continuing..."

echo "=== Step 3: Python Setup ==="
# Install Python 3.9 with Homebrew
if ! command_exists python3.9; then
  echo "Installing Python 3.9..."
  brew install python@3.9
else
  echo "Python 3.9 is already installed."
fi

# Update PATH to ensure Python 3.9 is used
export PATH="/usr/local/opt/python@3.9/bin:$PATH"

# Create symlinks if needed
if [ ! -f /usr/local/bin/python ]; then
  ln -sf /usr/local/bin/python3.9 /usr/local/bin/python
fi
if [ ! -f /usr/local/bin/python3 ]; then
  ln -sf /usr/local/bin/python3.9 /usr/local/bin/python3
fi
if [ ! -f /usr/local/bin/pip3 ]; then
  ln -sf /usr/local/bin/pip3.9 /usr/local/bin/pip3
fi

# Make sure pip is updated
pip3 install --upgrade pip

echo "=== Step 4: Nix Setup ==="
# Install Nix if not already installed
if ! command_exists nix-shell; then
  echo "Installing Nix..."
  sh <(curl -L https://nixos.org/nix/install) --darwin-use-unencrypted-nix-store-volume
  
  # Source Nix
  . ~/.nix-profile/etc/profile.d/nix.sh
else
  echo "Nix is already installed."
fi

# Create a basic dev.nix if it doesn't exist or is broken
mkdir -p .idx
cat > .idx/dev.nix << 'EOT'
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python
    python39
    python39Packages.pip
    python39Packages.setuptools
    python39Packages.wheel
    
    # System tools
    git
    curl
    wget
    
    # Build dependencies
    gcc
    gnumake
    
    # Other tools
    nodejs
    yarn
  ];
  
  shellHook = ''
    export PATH=$PATH:$HOME/.local/bin
    echo "AstroShield development environment loaded"
    python --version
    pip --version
  '';
}
EOT

echo "=== Step 5: Installing Docker (if needed) ==="
if ! command_exists docker; then
  echo "Docker not found. Please install Docker Desktop for Mac from:"
  echo "https://www.docker.com/products/docker-desktop"
  echo "After installation, please restart this script to continue setup."
  exit 0
else
  echo "Docker is already installed."
fi

echo "=== Step 6: Testing the Environment ==="
# Test Python
echo "Testing Python installation..."
python3 --version
pip3 --version

# Test Nix-shell if available
if command_exists nix-shell; then
  echo "Testing nix-shell..."
  nix-shell .idx/dev.nix --run "python --version && pip --version" || echo "Nix-shell test failed, but installation may still be OK."
fi

echo "=== Environment Setup Complete ==="
echo ""
echo "To start using the environment, run:"
echo "  nix-shell .idx/dev.nix"
echo ""
echo "For Docker-based development:"
echo "  1. Build the Docker image: docker-compose build"
echo "  2. Run containers: docker-compose up -d"
echo ""
echo "If you encounter any issues, please try running the following commands:"
echo "  1. brew update && brew upgrade"
echo "  2. python3 --version"
echo "  3. pip3 --version"
echo "  4. nix-shell .idx/dev.nix"
echo ""
echo "If problems persist, consider rebuilding your Docker environment using the provided Dockerfile." 