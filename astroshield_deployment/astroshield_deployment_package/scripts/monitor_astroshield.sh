#!/bin/bash

# Configuration
LOG_FILE="/var/log/astroshield/monitor.log"
ALERT_EMAIL="admin@sdataplab.com"
CHECK_INTERVAL=300  # 5 minutes

# Function to send alert email
send_alert() {
    local subject="$1"
    local message="$2"
    echo "$message" | mail -s "AstroShield Alert: $subject" "$ALERT_EMAIL"
}

# Function to check service status
check_service() {
    if ! systemctl is-active --quiet astroshield; then
        send_alert "Service Down" "The AstroShield service is not running. Attempting to restart..."
        systemctl restart astroshield
        sleep 10
        if ! systemctl is-active --quiet astroshield; then
            send_alert "Service Restart Failed" "Failed to restart the AstroShield service. Manual intervention required."
        fi
    fi
}

# Function to check disk space
check_disk_space() {
    local disk_usage=$(df -h /var/www/astroshield | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 85 ]; then
        send_alert "High Disk Usage" "Disk usage is at ${disk_usage}%. Please check and clean up if necessary."
    fi
}

# Function to check log file size
check_log_size() {
    local log_size=$(du -m /var/log/astroshield/astroshield.error.log | awk '{print $1}')
    if [ "$log_size" -gt 100 ]; then
        send_alert "Large Log File" "Error log file is larger than 100MB. Consider rotating logs."
    fi
}

# Main monitoring loop
while true; do
    echo "$(date) - Starting monitoring checks..." >> "$LOG_FILE"
    
    check_service
    check_disk_space
    check_log_size
    
    echo "$(date) - Monitoring checks completed." >> "$LOG_FILE"
    sleep "$CHECK_INTERVAL"
done 