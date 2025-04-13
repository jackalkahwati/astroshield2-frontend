#!/bin/bash
set -e

echo "=== AstroShield Environment Setup ==="
echo "This script will set up a complete development environment for AstroShield"

# Check if we're running as root or with sudo
if [ "$(id -u)" -ne 0 ]; then
  echo "This script needs to be run with sudo. Please run 'sudo $0'"
  exit 1
fi

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

echo "=== Step 1: System Update ==="
apt-get update || { echo "Failed to update package lists. This could indicate a corrupted apt database."; }

# Fix potential apt issues
echo "Fixing potential apt issues..."
apt-get install --reinstall apt || echo "Could not reinstall apt, continuing anyway..."

# Update system
echo "Updating system packages..."
apt-get update
apt-get upgrade -y

echo "=== Step 2: Installing Essential Packages ==="
apt-get install -y \
  build-essential \
  curl \
  wget \
  git \
  software-properties-common \
  ca-certificates \
  apt-transport-https

echo "=== Step 3: Python Setup ==="
# Remove potentially broken Python installations
echo "Removing potentially broken Python installations..."
apt-get remove -y python3 python3-pip || echo "No existing Python installation found"

# Install Python 3.9
echo "Installing Python 3.9..."
apt-get install -y python3.9 python3.9-dev python3.9-distutils

# Set Python 3.9 as the default
update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install pip
echo "Installing pip..."
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Create symlinks if needed
ln -sf /usr/bin/python3.9 /usr/local/bin/python
ln -sf /usr/bin/python3.9 /usr/local/bin/python3
ln -sf /usr/local/bin/pip /usr/local/bin/pip3

echo "=== Step 4: Nix Setup ==="
# Install Nix if not already installed
if ! command_exists nix-shell; then
  echo "Installing Nix..."
  curl -L https://nixos.org/nix/install | sh
  
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

echo "=== Step 5: Testing the Environment ==="
# Test Python
echo "Testing Python installation..."
python --version
pip --version

# Test apt-get
echo "Testing apt-get..."
apt-get install -y curl

# Test Nix-shell
echo "Testing nix-shell..."
nix-shell .idx/dev.nix --run "python --version && pip --version"

echo "=== Environment Setup Complete ==="
echo ""
echo "To start using the environment, run:"
echo "  nix-shell .idx/dev.nix"
echo ""
echo "If you encounter any issues, please try running the following commands:"
echo "  1. sudo apt-get update && sudo apt-get upgrade -y"
echo "  2. python --version"
echo "  3. pip --version"
echo "  4. nix-shell .idx/dev.nix"
echo ""
echo "If problems persist, consider rebuilding your Docker environment using the provided Dockerfile." 