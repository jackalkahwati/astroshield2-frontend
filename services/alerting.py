"""Alerting system for CCDM service critical errors."""
import threading
import smtplib
import requests
import json
import time
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Set

# Import configuration
from services.config_loader import get_config

logger = logging.getLogger(__name__)

class AlertManager:
    """Alert system for critical errors."""
    
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config["alerting"]["enabled"]
        self.email_config = self.config["alerting"]["email"]
        self.slack_config = self.config["alerting"]["slack"]
        
        # Maintain a set of recent alert IDs to prevent duplicates
        self.recent_alerts: Set[str] = set()
        self.lock = threading.RLock()
        
        # Start alert cleaner thread
        if self.enabled:
            self._start_alert_cleaner()
        
        logger.info(f"Alert manager initialized. Enabled: {self.enabled}")
            
    def _start_alert_cleaner(self):
        """Start background thread to clean old alerts."""
        def clean_alerts():
            while True:
                time.sleep(3600)  # Run every hour
                with self.lock:
                    old_size = len(self.recent_alerts)
                    self.recent_alerts.clear()
                    logger.debug(f"Cleared {old_size} alert deduplication entries")
                    
        thread = threading.Thread(target=clean_alerts, daemon=True)
        thread.start()
        logger.debug("Alert cleaner thread started")
        
    def _is_duplicate(self, alert_id: str) -> bool:
        """Check if alert is a duplicate."""
        with self.lock:
            if alert_id in self.recent_alerts:
                return True
            self.recent_alerts.add(alert_id)
            return False
            
    def send_alert(self, level: str, title: str, message: str, context: Dict[str, Any] = None):
        """
        Send alert via configured channels.
        
        Args:
            level: Alert level (critical, error, warning)
            title: Alert title
            message: Alert message
            context: Additional context for the alert
        """
        if not self.enabled:
            return
            
        # Only alert for critical and error levels
        if level.lower() not in ("critical", "error"):
            return
            
        # Create alert ID to prevent duplicates
        if context:
            alert_id = f"{level}:{title}:{context.get('error_type', '')}"
        else:
            alert_id = f"{level}:{title}"
            
        if self._is_duplicate(alert_id):
            logger.debug(f"Suppressing duplicate alert: {alert_id}")
            return
            
        logger.info(f"Sending {level} alert: {title}")
            
        # Prepare alert data
        alert_data = {
            "level": level,
            "title": title,
            "message": message,
            "service": self.config["service"]["name"],
            "version": self.config["service"]["version"],
            "environment": self.config.get("environment", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if context:
            alert_data["context"] = context
            
        # Send via configured channels
        if self.email_config["enabled"]:
            self._send_email_alert(alert_data)
            
        if self.slack_config["enabled"]:
            self._send_slack_alert(alert_data)
            
    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send alert via email."""
        try:
            config = self.email_config
            
            # Check if email configuration is valid
            if not config.get("smtp_server") or not config.get("from_address") or not config.get("to_addresses"):
                logger.warning("Email alert configuration is incomplete")
                return
                
            # Create message
            msg = MIMEMultipart()
            msg['Subject'] = f"[{alert_data['level'].upper()}] {alert_data['service']} - {alert_data['title']}"
            msg['From'] = config['from_address']
            msg['To'] = ", ".join(config['to_addresses'])
            
            # Create message body
            body = f"""
            <html>
            <body>
                <h2>{alert_data['title']}</h2>
                <p><strong>Level:</strong> {alert_data['level']}</p>
                <p><strong>Service:</strong> {alert_data['service']} v{alert_data['version']}</p>
                <p><strong>Environment:</strong> {alert_data['environment']}</p>
                <p><strong>Time:</strong> {alert_data['timestamp']}</p>
                <h3>Message:</h3>
                <p>{alert_data['message']}</p>
            """
            
            if 'context' in alert_data:
                body += "<h3>Context:</h3><pre>" + json.dumps(alert_data['context'], indent=2) + "</pre>"
                
            body += """
            </body>
            </html>
            """
            
            # Attach body
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                if config.get('username') and config.get('password'):
                    server.login(config['username'], config['password'])
                server.send_message(msg)
                
            logger.info(f"Email alert sent: {alert_data['title']}")
                
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            
    def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Send alert via Slack webhook."""
        try:
            webhook_url = self.slack_config.get('webhook_url')
            if not webhook_url:
                logger.warning("Slack webhook URL not configured")
                return
                
            # Create color based on level
            color = "#ff0000" if alert_data['level'].lower() == "critical" else "#ffa500"
            
            # Create attachment
            attachment = {
                "fallback": f"{alert_data['level'].upper()}: {alert_data['title']}",
                "color": color,
                "title": f"{alert_data['level'].upper()}: {alert_data['title']}",
                "text": alert_data['message'],
                "fields": [
                    {
                        "title": "Service",
                        "value": f"{alert_data['service']} v{alert_data['version']}",
                        "short": True
                    },
                    {
                        "title": "Environment",
                        "value": alert_data['environment'],
                        "short": True
                    },
                    {
                        "title": "Time",
                        "value": alert_data['timestamp'],
                        "short": True
                    }
                ],
                "footer": "CCDM Service Alerts"
            }
            
            # Add context if available
            if 'context' in alert_data:
                context_text = json.dumps(alert_data['context'], indent=2)
                attachment["fields"].append({
                    "title": "Context",
                    "value": f"```{context_text}```",
                    "short": False
                })
                
            # Create payload
            payload = {
                "text": f"Alert from {alert_data['service']} ({alert_data['environment']})",
                "attachments": [attachment]
            }
            
            # Send to Slack
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to send Slack alert: {response.status_code} {response.text}")
            else:
                logger.info(f"Slack alert sent: {alert_data['title']}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")

# Global alert manager instance
_alert_manager = None

def get_alert_manager():
    """
    Get alert manager singleton instance.
    
    Returns:
        AlertManager instance
    """
    global _alert_manager
    
    if _alert_manager is None:
        _alert_manager = AlertManager()
        
    return _alert_manager

def send_alert(level: str, title: str, message: str, context: Dict[str, Any] = None):
    """
    Send alert via configured channels.
    
    Args:
        level: Alert level (critical, error, warning)
        title: Alert title
        message: Alert message
        context: Additional context for the alert
    """
    alert_manager = get_alert_manager()
    alert_manager.send_alert(level, title, message, context) 