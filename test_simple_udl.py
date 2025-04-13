#!/usr/bin/env python3
"""
Simple test for UDL client with minimal dependencies.
"""
import os
import sys
import requests
import json
from datetime import datetime

def test_udl_service():
    """Test the UDL service connection"""
    base_url = os.environ.get("UDL_BASE_URL", "http://localhost:8888")
    token = None # Initialize token
    
    # Test the root endpoint
    try:
        print(f"Testing connection to {base_url}...")
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✓ UDL service is responding")
        else:
            print(f"! UDL service returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"! Error connecting to UDL service: {str(e)}")
        return False
    
    # Test authentication
    try:
        # Use test_user from .env or fallback to defaults
        username = os.environ.get("UDL_USERNAME", "test_user")
        password = os.environ.get("UDL_PASSWORD", "test_password")
        
        print(f"Testing authentication with username: {username}...")
        auth_url = f"{base_url}/auth/token"
        auth_response = requests.post(auth_url, json={
            "username": username, 
            "password": password
        })
        
        if auth_response.status_code == 200 and "token" in auth_response.json():
            token = auth_response.json()["token"]
            print(f"✓ Authentication successful. Token: {token[:10]}...")
        else:
            print(f"! Authentication failed with status: {auth_response.status_code}")
            print(f"! Response: {auth_response.text}")
            # For testing purposes, create a mock token to continue
            print("Using mock token to continue test")
            token = "mock-token-for-testing"
    except Exception as e:
        print(f"! Authentication error: {str(e)}")
        # For testing purposes, use a mock token to continue
        print("Using mock token to continue test")
        token = "mock-token-for-testing"
    
    # Test getting state vectors
    try:
        print("Testing state vectors endpoint...")
        vectors_url = f"{base_url}/statevector"
        vectors_response = requests.get(
            vectors_url,
            headers={"Authorization": f"Bearer {token}"},
            params={"epoch": "now"}
        )
        
        if vectors_response.status_code == 200:
            data = vectors_response.json()
            if "stateVectors" in data and len(data["stateVectors"]) > 0:
                print(f"✓ Received {len(data['stateVectors'])} state vectors")
                
                # Print the first vector as an example
                sv = data["stateVectors"][0]
                print(f"  Example: {sv['id']} - {sv['name']}")
                print(f"    Position: x={sv['position']['x']:.1f}, y={sv['position']['y']:.1f}, z={sv['position']['z']:.1f}")
                return True
            else:
                print("! No state vectors found in response")
                return False
        else:
            print(f"! State vector request failed with status: {vectors_response.status_code}")
            return False
    except Exception as e:
        print(f"! Error getting state vectors: {str(e)}")
        return False

if __name__ == "__main__":
    # Load environment variables from .env file
    if os.path.exists(".env"):
        print("Loading environment variables from .env file")
        with open(".env", "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    try:
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value
                        if key == "UDL_USERNAME" or key == "UDL_PASSWORD" or key == "UDL_BASE_URL":
                            print(f"Loaded {key}={value}")
                    except ValueError:
                        print(f"Warning: Skipping malformed line in .env: {line.strip()}")
    
    success = test_udl_service()
    sys.exit(0 if success else 1)
