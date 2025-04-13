import http.client
import json

def test_endpoint(host, port, path):
    conn = http.client.HTTPConnection(host, port)
    conn.request("GET", path)
    response = conn.getresponse()
    print(f"Status: {response.status} {response.reason}")
    if response.status == 200:
        data = response.read().decode()
        print(f"Response: {data[:100]}...")  # Show first 100 chars
    else:
        print(f"Error: {response.read().decode()}")
    conn.close()

if __name__ == "__main__":
    print("Testing API endpoints...")
    test_endpoint("localhost", 3002, "/api/v1/maneuvers")