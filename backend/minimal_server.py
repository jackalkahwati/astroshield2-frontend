from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class MockHandler(BaseHTTPRequestHandler):
    def log_request(self, code='-', size='-'):
        logger.info(f"Request: {self.command} {self.path} {code}")
    
    def _set_headers(self, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        # Allow requests from any origin
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        logger.info(f"Set headers for {self.path}")

    def do_OPTIONS(self):
        logger.info(f"OPTIONS request received for {self.path}")
        self._set_headers()
        self.wfile.write(b'')
        
    def do_GET(self):
        logger.info(f"GET request received for {self.path}")
        
        if self.path == "/api/v1/maneuvers":
            self._handle_maneuvers()
        elif self.path == "/api/v1/satellites":
            self._handle_satellites()
        elif self.path == "/api/v1/health":
            self._handle_health()
        else:
            logger.warning(f"Unknown path: {self.path}")
            self._handle_not_found()

    def do_POST(self):
        logger.info(f"POST request received for {self.path}")
        self._set_headers()
        self.wfile.write(json.dumps({"success": True}).encode())

    def _handle_maneuvers(self):
        logger.info("Handling maneuvers request")
        self._set_headers()
        now = datetime.now()
        
        maneuvers = [
            {
                "id": "mnv-001",
                "satellite_id": "sat-001",
                "status": "completed",
                "type": "collision_avoidance",
                "scheduledTime": (now - timedelta(hours=12)).isoformat(),
                "completedTime": (now - timedelta(hours=11)).isoformat(),
                "details": {
                    "delta_v": 0.02,
                    "duration": 15.0,
                    "fuel_required": 5.2
                },
                "created_by": "user@example.com",
                "created_at": (now - timedelta(hours=24)).isoformat()
            },
            {
                "id": "mnv-002",
                "satellite_id": "sat-001",
                "status": "scheduled",
                "type": "station_keeping",
                "scheduledTime": (now + timedelta(hours=5)).isoformat(),
                "completedTime": None,
                "details": {
                    "delta_v": 0.01,
                    "duration": 10.0,
                    "fuel_required": 2.1
                },
                "created_by": "user@example.com",
                "created_at": (now - timedelta(hours=3)).isoformat()
            },
            {
                "id": "mnv-003",
                "satellite_id": "sat-002",
                "status": "in_progress",
                "type": "orbital_adjustment",
                "scheduledTime": (now - timedelta(minutes=30)).isoformat(),
                "completedTime": None,
                "details": {
                    "delta_v": 0.15,
                    "duration": 45.0,
                    "fuel_required": 12.7
                },
                "created_by": "admin@example.com",
                "created_at": (now - timedelta(hours=2)).isoformat()
            }
        ]
        
        response_json = json.dumps(maneuvers)
        logger.info(f"Sending maneuvers response: {response_json[:100]}...")
        self.wfile.write(response_json.encode())

    def _handle_satellites(self):
        self._set_headers()
        satellites = [
            {
                "id": "sat-001",
                "name": "AstroShield-1",
                "status": "operational",
                "orbit": {
                    "altitude": 500.2,
                    "inclination": 45.0,
                    "eccentricity": 0.001
                }
            },
            {
                "id": "sat-002",
                "name": "AstroShield-2",
                "status": "operational",
                "orbit": {
                    "altitude": 525.7,
                    "inclination": 52.5,
                    "eccentricity": 0.002
                }
            }
        ]
        self.wfile.write(json.dumps(satellites).encode())

    def _handle_health(self):
        self._set_headers()
        health = {
            "status": "healthy",
            "uptime": "1d 5h 22m",
            "memory": {
                "used": 512,
                "total": 2048,
                "percent": 25.0
            },
            "cpu": {
                "usage": 12.5
            },
            "version": "1.0.0"
        }
        self.wfile.write(json.dumps(health).encode())

    def _handle_not_found(self):
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        error_response = {"error": "Not found", "path": self.path}
        self.wfile.write(json.dumps(error_response).encode())

def run(server_class=HTTPServer, handler_class=MockHandler, port=3002):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logger.info(f"Starting mock server on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        httpd.server_close()
        logger.info("Server closed")

if __name__ == "__main__":
    run()