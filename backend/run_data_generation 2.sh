#!/bin/bash
# Run the UDL data collection and training example generation

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -q requests python-dotenv

# Check if .env file exists and has credentials
if [ ! -f ".env" ]; then
  echo "ERROR: .env file not found!"
  echo "Please create a .env file with UDL credentials before running this script."
  exit 1
fi

# Check if credentials are set in .env
if ! grep -q "UDL_USERNAME" .env || ! grep -q "UDL_PASSWORD" .env; then
  echo "ERROR: UDL credentials not found in .env file!"
  echo "Please set UDL_USERNAME and UDL_PASSWORD in your .env file."
  exit 1
fi

# Make sure the password is not the placeholder
if grep -q "your-password-here" .env; then
  echo "ERROR: Please set your actual UDL password in the .env file."
  echo "The current password is still set to the placeholder value."
  exit 1
fi

# Display the current UDL connection settings
echo "UDL Connection Settings:"
echo "  Username: $(grep UDL_USERNAME .env | cut -d= -f2)"
echo "  Auth URL: $(grep UDL_AUTH_URL .env | cut -d= -f2)"
echo "  Base URL: $(grep UDL_BASE_URL .env | cut -d= -f2)"
echo "  Mock Mode: $(grep UDL_USE_MOCK .env | cut -d= -f2)"

# Ask for confirmation before proceeding
echo
echo "Do you want to proceed with data collection? (y/n)"
read -r response
if [[ "$response" != "y" ]]; then
  echo "Data collection cancelled by user."
  exit 0
fi

# Run the data collection script
echo "Starting UDL data collection and training example generation..."
python generate_training_data.py

# Check if execution was successful
if [ $? -eq 0 ]; then
  echo "Data collection and training example generation completed successfully!"
else
  echo "ERROR: Data collection and training example generation failed!"
  exit 1
fi

# Deactivate the virtual environment
deactivate