#!/usr/bin/env python3
"""
Script to publish AstroShield API documentation to SwaggerHub
"""

import os
import sys
import json
import argparse
import requests

def check_file_exists(file_path):
    """Check if the specified file exists."""
    if not os.path.isfile(file_path):
        print(f"Error: File {file_path} not found.")
        return False
    return True

def publish_to_swaggerhub(username, api_key, api_file, api_name="AstroShield", api_version="1.0.0"):
    """Attempt to publish an OpenAPI specification to SwaggerHub."""
    if not check_file_exists(api_file):
        return False
    
    # Load the OpenAPI specification
    try:
        with open(api_file, 'r') as f:
            openapi_spec = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {api_file} is not a valid JSON file.")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    # Check if this is a valid OpenAPI spec
    if 'openapi' not in openapi_spec and 'swagger' not in openapi_spec:
        print("Error: The file does not appear to be a valid OpenAPI specification.")
        return False
    
    # Headers for API requests
    headers = {
        'Authorization': api_key,
        'Content-Type': 'application/json'
    }
    
    # Try multiple endpoints
    endpoints = [
        f"https://api.swaggerhub.com/apis/{username}/{api_name}/{api_version}",
        f"https://api.swaggerhub.com/apis/{username}/{api_name}/{api_version}?isPrivate=false",
        "https://api.swaggerhub.com/specs"
    ]
    
    methods = ["POST", "PUT", "POST"]
    payloads = [
        json.dumps(openapi_spec),
        json.dumps(openapi_spec),
        json.dumps({
            "name": api_name,
            "version": api_version,
            "specification": openapi_spec
        })
    ]
    
    # Try each endpoint
    for i, (endpoint, method, payload) in enumerate(zip(endpoints, methods, payloads)):
        print(f"Attempt {i+1}: Using {method} to {endpoint}")
        try:
            if method == "POST":
                response = requests.post(endpoint, headers=headers, data=payload)
            else:
                response = requests.put(endpoint, headers=headers, data=payload)
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code in (200, 201):
                print("Success! API uploaded successfully.")
                return True
            else:
                print(f"Failed with response: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
    
    # If all attempts failed, provide alternative instructions
    print("\nAll automatic upload attempts failed.")
    print("\nPlease follow the manual process:")
    print("1. Go to app.swaggerhub.com and log in")
    print("2. Click 'Create New' > 'Create New API'")
    print("3. Select 'Import and Document API'")
    print(f"4. Upload the file: {api_file}")
    print("5. Complete the form and click 'Create API'")
    
    return False

def main():
    parser = argparse.ArgumentParser(description='Publish OpenAPI specification to SwaggerHub')
    parser.add_argument('--username', required=True, help='Your SwaggerHub username')
    parser.add_argument('--api-key', default='d8533cd1-a315-408f-bf4f-fcd898863daf', help='Your SwaggerHub API key')
    parser.add_argument('--file', default='openapi.json', help='Path to the OpenAPI specification file')
    parser.add_argument('--name', default='AstroShield', help='API name')
    parser.add_argument('--version', default='1.0.0', help='API version')
    
    args = parser.parse_args()
    
    success = publish_to_swaggerhub(
        username=args.username,
        api_key=args.api_key,
        api_file=args.file,
        api_name=args.name,
        api_version=args.version
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 