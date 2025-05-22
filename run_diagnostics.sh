#!/bin/bash

# This script automates the diagnostic steps for the AstroShield deployment on EC2.
# It will log outputs from each major command for review in diagnose_astroshield.log

LOGFILE="diagnose_astroshield.log"
echo "Starting AstroShield Diagnostic Script" | tee "$LOGFILE"

echo "\nStep 2: Checking astroshield.service status" | tee -a "$LOGFILE"
sudo systemctl status astroshield.service | tee -a "$LOGFILE"

echo "\nStep 3: Checking detailed logs for astroshield.service" | tee -a "$LOGFILE"
sudo journalctl -xeu astroshield.service | tee -a "$LOGFILE"

echo "\nStep 4: Verifying permissions of ~/astroshield/start.sh" | tee -a "$LOGFILE"
ls -l ~/astroshield/start.sh | tee -a "$LOGFILE"
echo "Setting execute permission for start.sh" | tee -a "$LOGFILE"
chmod +x ~/astroshield/start.sh

echo "Running start.sh..." | tee -a "$LOGFILE"
~/astroshield/start.sh 2>&1 | tee -a "$LOGFILE"

echo "\nStep 5: Checking Docker container status" | tee -a "$LOGFILE"
docker ps -a | tee -a "$LOGFILE"

# Attempt to locate the AstroShield container by filtering with name 'astroshield'
CONTAINER_ID=$(docker ps -a --filter "name=astroshield" --format "{{.ID}}")
if [ -n "$CONTAINER_ID" ]; then
  echo "\nFound AstroShield container with ID: $CONTAINER_ID. Fetching logs:" | tee -a "$LOGFILE"
  docker logs "$CONTAINER_ID" | tee -a "$LOGFILE"
else
  echo "No AstroShield container found." | tee -a "$LOGFILE"
fi

echo "\nStep 6: Verifying .env file presence and contents" | tee -a "$LOGFILE"
ls -la ~/astroshield/.env | tee -a "$LOGFILE"
echo "Contents of .env file:" | tee -a "$LOGFILE"
cat ~/astroshield/.env | tee -a "$LOGFILE"

echo "\nStep 7: Reviewing systemd service file at /etc/systemd/system/astroshield.service" | tee -a "$LOGFILE"
sudo cat /etc/systemd/system/astroshield.service | tee -a "$LOGFILE"
echo "Please verify manually that the paths and user are correct (should be /home/stardrive/astroshield and user stardrive)." | tee -a "$LOGFILE"

echo "Reloading systemd daemon and restarting astroshield.service" | tee -a "$LOGFILE"
sudo systemctl daemon-reload
sudo systemctl restart astroshield.service
sudo systemctl status astroshield.service | tee -a "$LOGFILE"

echo "\nStep 8: Verifying internal application accessibility on port 8080" | tee -a "$LOGFILE"
echo "Testing localhost:8080" | tee -a "$LOGFILE"
curl -v localhost:8080 2>&1 | tee -a "$LOGFILE"
if [ -n "$CONTAINER_ID" ]; then
  echo "\nInspecting Docker container port bindings for container $CONTAINER_ID" | tee -a "$LOGFILE"
  docker inspect "$CONTAINER_ID" | grep -i port | tee -a "$LOGFILE"
fi

echo "\nDiagnostic script completed. Review the log file ($LOGFILE) for detailed outputs." | tee -a "$LOGFILE" 