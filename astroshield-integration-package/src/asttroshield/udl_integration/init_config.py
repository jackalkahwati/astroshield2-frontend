#!/usr/bin/env python3
"""
UDL Configuration Initialization Script

This script creates a new UDL configuration file with default settings.
It can be used to create a starting point for configuring the UDL integration.
"""

import os
import sys
import argparse
import shutil
import yaml
import json

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def initialize_config(output_path: str, format: str = "yaml") -> bool:
    """
    Initialize a new UDL configuration file.
    
    Args:
        output_path: Path to write the configuration file
        format: Format of the configuration file ("yaml" or "json")
        
    Returns:
        True if successful, False otherwise
    """
    # Default config file path
    default_config_path = os.path.join(SCRIPT_DIR, "udl_config.yaml")
    
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if format.lower() == "yaml":
            # Copy the default YAML file
            shutil.copy(default_config_path, output_path)
            print(f"Created YAML configuration file at: {output_path}")
            return True
        elif format.lower() == "json":
            # Convert YAML to JSON
            with open(default_config_path, 'r') as yaml_file:
                config_data = yaml.safe_load(yaml_file)
            
            with open(output_path, 'w') as json_file:
                json.dump(config_data, json_file, indent=2)
            
            print(f"Created JSON configuration file at: {output_path}")
            return True
        else:
            print(f"Unsupported format: {format}. Use 'yaml' or 'json'.")
            return False
            
    except Exception as e:
        print(f"Error initializing configuration: {str(e)}")
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Initialize UDL configuration file")
    parser.add_argument("--output", "-o", 
                        default="./udl_config.yaml",
                        help="Path to output configuration file")
    parser.add_argument("--format", "-f", 
                        choices=["yaml", "json"], 
                        default="yaml",
                        help="Format of the configuration file")
    
    args = parser.parse_args()
    
    success = initialize_config(args.output, args.format)
    if success:
        print("Configuration initialized successfully.")
        print("\nNext steps:")
        print("1. Edit the configuration file with your UDL credentials and settings.")
        print("2. Initialize the UDL integration with the configuration file:")
        print("\n   from asttroshield.udl_integration import UDLIntegration")
        print("   integration = UDLIntegration(config_file='path/to/config.yaml')")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 