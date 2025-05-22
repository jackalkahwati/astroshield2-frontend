# Deployment Instructions for AstroShield on Amazon Linux EC2

1. Configuration:
   - Ensure that environment variables (e.g., ASTROSHIELD_CONFIG) are properly set on your EC2 instance.
   - Confirm that all configured paths in start_astroshield.sh and related config files are absolute and correct.

2. Systemd Service Setup:
   - A systemd service file has been created at /etc/systemd/system/astroshield.service.
   - On your Amazon Linux EC2 instance, run the following commands:
     sudo systemctl daemon-reload
     sudo systemctl start astroshield.service
     sudo systemctl enable astroshield.service

3. Dependency Installation:
   - Activate your virtual environment on the EC2 instance.
   - Ensure that all required Python packages are installed (uvicorn, fastapi, python-jose, passlib, sqlalchemy, email-validator, redis, numpy, scipy, pydantic_settings, etc.).

4. Reverse Proxy and Network Configuration:
   - Verify that your reverse proxy (e.g., nginx) is correctly configured to forward requests to the backend service.
   - Ensure that firewall/port settings allow communication between the reverse proxy and AstroShield.

5. Validation:
   - Test critical endpoints (such as /health and /openapi.json) using curl or a browser to confirm the backend is accessible.
   - Run the full test suite (with pytest) on the EC2 instance to verify that there are no import or execution errors.

6. Monitoring and Logging:
   - Confirm that log files are being created with correct permissions.
   - Set up any additional monitoring tools (e.g., Prometheus, Grafana) as needed.

Follow these steps to complete the deployment of AstroShield on your Amazon Linux EC2 instance behind the reverse proxy. 