#!/usr/bin/env python3
"""
Simple script to test UDL API connectivity and authentication.
"""

import os
import requests
import base64
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_api_key_auth():
    """Test authentication with API key."""
    api_key = os.environ.get("UDL_API_KEY")
    if not api_key:
        print("No UDL_API_KEY found in environment variables.")
        return False
    
    base_url = os.environ.get("UDL_BASE_URL", "https://unifieddatalibrary.com")
    url = f"{base_url}/udl/statevector"
    
    headers = {"X-API-Key": api_key}
    
    try:
        response = requests.get(url, headers=headers, params={"maxResults": 1}, timeout=30)
        print(f"API Key Auth - Status: {response.status_code}")
        
        if response.ok:
            print("API Key Authentication successful!")
            print(f"Response: {json.dumps(response.json(), indent=2)[:200]}...")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error testing API key auth: {e}")
        return False

def test_basic_auth():
    """Test authentication with username/password using basic auth."""
    username = os.environ.get("UDL_USERNAME")
    password = os.environ.get("UDL_PASSWORD")
    
    if not username or not password:
        print("No UDL_USERNAME or UDL_PASSWORD found in environment variables.")
        return False
    
    base_url = os.environ.get("UDL_BASE_URL", "https://unifieddatalibrary.com")
    
    # Include required 'epoch' parameter as mentioned in the error message
    url = f"{base_url}/udl/statevector"
    params = {"epoch": "now"}  # Using 'now' as a relative time
    
    # Create basic auth header
    auth_str = f"{username}:{password}"
    auth_bytes = auth_str.encode('ascii')
    base64_bytes = base64.b64encode(auth_bytes)
    base64_auth = base64_bytes.decode('ascii')
    
    headers = {"Authorization": f"Basic {base64_auth}"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        print(f"Basic Auth - Status: {response.status_code}")
        
        if response.ok:
            print("Basic Authentication successful!")
            print(f"Response: {json.dumps(response.json(), indent=2)[:200]}...")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error testing basic auth: {e}")
        return False

def test_token_auth():
    """Test authentication with username/password using token auth."""
    username = os.environ.get("UDL_USERNAME")
    password = os.environ.get("UDL_PASSWORD")
    
    if not username or not password:
        print("No UDL_USERNAME or UDL_PASSWORD found in environment variables.")
        return False
    
    base_url = os.environ.get("UDL_BASE_URL", "https://unifieddatalibrary.com")
    
    # Try multiple auth URL variants
    auth_urls = [
        f"{base_url}/auth/token",
        f"{base_url}/api/auth/token",
        f"{base_url}/udl/auth/token",
        f"{base_url}/udl/api/auth/token",
        f"{base_url}/api/v1/auth/token",
        f"{base_url}/udl/api/v1/auth/token",
        f"{base_url}/oauth/token",
        f"{base_url}/api/oauth/token"
    ]
    
    success = False
    token = None
    
    for auth_url in auth_urls:
        try:
            print(f"Trying auth URL: {auth_url}")
            
            # Try as JSON payload
            auth_response = requests.post(
                auth_url,
                json={"username": username, "password": password},
                timeout=10
            )
            
            print(f"  JSON Auth - Status: {auth_response.status_code}")
            
            if not auth_response.ok:
                # Try as form data
                auth_response = requests.post(
                    auth_url,
                    data={"username": username, "password": password},
                    timeout=10
                )
                print(f"  Form Auth - Status: {auth_response.status_code}")
            
            if auth_response.ok:
                try:
                    auth_data = auth_response.json()
                    token = auth_data.get("token", auth_data.get("access_token"))
                    
                    if token:
                        print(f"  Successfully obtained token from {auth_url}")
                        success = True
                        break
                    else:
                        print(f"  No token found in response: {auth_data}")
                except Exception as e:
                    print(f"  Error parsing response: {e}")
            else:
                print(f"  Auth failed: {auth_response.text[:100]}...")
                
        except Exception as e:
            print(f"  Error with URL {auth_url}: {e}")
    
    if not success or not token:
        print("Failed to obtain auth token from any endpoint.")
        return False
        
    # Test API access with token
    url = f"{base_url}/udl/statevector"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"epoch": "now"}  # Using 'now' as a relative time
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        print(f"API Access with Token - Status: {response.status_code}")
        
        if response.ok:
            print("Token Authentication successful!")
            print(f"Response: {json.dumps(response.json(), indent=2)[:200]}...")
            return True
        else:
            print(f"Error accessing API with token: {response.text}")
            return False
    except Exception as e:
        print(f"Error testing API access with token: {e}")
        return False

def test_api_endpoints():
    """Test connectivity to various UDL API endpoints."""
    base_url = os.environ.get("UDL_BASE_URL", "https://unifieddatalibrary.com")
    
    endpoints = [
        "/",
        "/udl",
        "/udl/openapi.json",
        "/udl/statevector/queryhelp",
        "/udl/conjunction/queryhelp",
        "/udl/sensor/queryhelp"
    ]
    
    print("\nTesting API endpoints without authentication:")
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            print(f"{url} - Status: {response.status_code}")
        except Exception as e:
            print(f"{url} - Error: {e}")

if __name__ == "__main__":
    print("=== UDL API Connectivity Test ===\n")
    
    print("Testing API Key Authentication:")
    api_key_success = test_api_key_auth()
    
    print("\nTesting Basic Authentication:")
    basic_auth_success = test_basic_auth()
    
    print("\nTesting Token Authentication:")
    token_auth_success = test_token_auth()
    
    # Test API endpoints
    test_api_endpoints()
    
    print("\n=== Summary ===")
    print(f"API Key Authentication: {'Success' if api_key_success else 'Failed'}")
    print(f"Basic Authentication: {'Success' if basic_auth_success else 'Failed'}")
    print(f"Token Authentication: {'Success' if token_auth_success else 'Failed'}")
    
    if not (api_key_success or basic_auth_success or token_auth_success):
        print("\nAll authentication methods failed.")
        print("Please verify your UDL credentials and the UDL API base URL.")
        print("You may also need to contact the UDL support team for assistance.")
    else:
        print("\nAt least one authentication method succeeded.")
        print("You should be able to use the UDL Integration with the working auth method.") 