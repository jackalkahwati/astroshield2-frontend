import requests
import json

def test_health_endpoint():
    response = requests.get("http://localhost:3001/api/v1/health")
    print("Health response:", response.status_code, response.json())
    assert response.status_code == 200

def test_trajectory_endpoint():
    # Check if the trajectory endpoint works
    data = {
        "config": {
            "object_name": "Test Object",
            "atmospheric_model": "exponential",
            "wind_model": "custom",
            "monte_carlo_samples": 100,
            "object_properties": {
                "mass": 100,
                "area": 1.2,
                "cd": 2.2
            },
            "breakup_model": {
                "enabled": True,
                "fragmentation_threshold": 50
            }
        },
        "initial_state": [0, 0, 400000, 7800, 0, 0]
    }
    response = requests.post("http://localhost:3001/api/trajectory/analyze", json=data)
    print("Trajectory response:", response.status_code)
    if response.status_code != 200:
        print("Error:", response.text)
    else:
        print("Response data:", response.json())
    
    # Try listing available endpoint URLs
    print("\nTrying to get API documentation...")
    docs_response = requests.get("http://localhost:3001/api/v1/openapi.json")
    if docs_response.status_code == 200:
        api_docs = docs_response.json()
        if "paths" in api_docs:
            print("Available endpoints:")
            for path in api_docs["paths"]:
                print(f"- {path}")
        else:
            print("No paths found in API documentation")
    else:
        print("Could not retrieve API documentation:", docs_response.status_code)

if __name__ == "__main__":
    print("Running health test...")
    test_health_endpoint()
    print("\nRunning trajectory test...")
    test_trajectory_endpoint() 