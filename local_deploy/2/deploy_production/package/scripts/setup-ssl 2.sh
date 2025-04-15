#!/bin/bash
# Setup SSL for AstroShield using Certbot - AWS Gov Cloud version
# Attempts multiple approaches since certbot might be restricted

set -e

# Get domain from parameter or use default
DOMAIN=${1:-"astroshield.sdataplab.com"}

echo "Setting up SSL for $DOMAIN..."

# Check if certbot is installed
if command -v certbot &> /dev/null; then
    echo "Certbot found, attempting to obtain certificate..."
    
    # First try: automatic Nginx configuration
    if sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN; then
        echo "Successfully obtained certificate with Nginx plugin!"
    else
        # Second try: standalone mode (requires port 80 to be free)
        echo "Nginx plugin failed. Trying standalone mode..."
        sudo systemctl stop nginx || true
        if sudo certbot certonly --standalone -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN; then
            echo "Successfully obtained certificate with standalone mode!"
            
            # Update Nginx configuration manually
            echo "Updating Nginx configuration to use the new certificate..."
            # Make sure paths in nginx config match certbot paths
            sudo sed -i "s|ssl_certificate .*|ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;|g" /etc/nginx/conf.d/astroshield.conf
            sudo sed -i "s|ssl_certificate_key .*|ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;|g" /etc/nginx/conf.d/astroshield.conf
        else
            # Third try: webroot mode
            echo "Standalone mode failed. Trying webroot mode..."
            sudo systemctl start nginx || true
            sudo mkdir -p /var/www/html/.well-known
            if sudo certbot certonly --webroot -w /var/www/html -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN; then
                echo "Successfully obtained certificate with webroot mode!"
                
                # Update Nginx configuration manually
                echo "Updating Nginx configuration to use the new certificate..."
                sudo sed -i "s|ssl_certificate .*|ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;|g" /etc/nginx/conf.d/astroshield.conf
                sudo sed -i "s|ssl_certificate_key .*|ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;|g" /etc/nginx/conf.d/astroshield.conf
            else
                echo "All automatic certificate methods failed."
                echo "You may need to obtain a certificate manually or check if wildcard certificate is already available."
                
                # Check if a wildcard certificate might exist
                if [ -d "/etc/letsencrypt/live/*.sdataplab.com" ]; then
                    echo "Wildcard certificate for *.sdataplab.com might be available. Checking..."
                    sudo sed -i "s|ssl_certificate .*|ssl_certificate /etc/letsencrypt/live/*.sdataplab.com/fullchain.pem;|g" /etc/nginx/conf.d/astroshield.conf
                    sudo sed -i "s|ssl_certificate_key .*|ssl_certificate_key /etc/letsencrypt/live/*.sdataplab.com/privkey.pem;|g" /etc/nginx/conf.d/astroshield.conf
                    echo "Updated configuration to use potential wildcard certificate."
                fi
            fi
        fi
    fi
    
    # Ensure nginx is restarted to load new certificate
    sudo systemctl restart nginx || echo "Warning: Failed to restart Nginx"
else
    echo "Certbot not found."
    
    # Check if corporate wildcard certificate is available
    if [ -f "/etc/ssl/certs/sdataplab-wildcard.pem" ]; then
        echo "Using corporate wildcard certificate..."
        sudo sed -i "s|ssl_certificate .*|ssl_certificate /etc/ssl/certs/sdataplab-wildcard.pem;|g" /etc/nginx/conf.d/astroshield.conf
        sudo sed -i "s|ssl_certificate_key .*|ssl_certificate_key /etc/ssl/private/sdataplab-wildcard.key;|g" /etc/nginx/conf.d/astroshield.conf
        echo "Updated configuration to use corporate wildcard certificate."
    else
        echo "No certificates available. You will need to manually configure SSL."
        echo "For now, creating a self-signed certificate for testing..."
        
        # Create self-signed certificate
        sudo mkdir -p /etc/ssl/private
        sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
          -keyout /etc/ssl/private/astroshield-selfsigned.key \
          -out /etc/ssl/certs/astroshield-selfsigned.crt \
          -subj "/C=US/ST=State/L=City/O=SDATestPlatform/CN=$DOMAIN"
        
        # Update Nginx configuration
        sudo sed -i "s|ssl_certificate .*|ssl_certificate /etc/ssl/certs/astroshield-selfsigned.crt;|g" /etc/nginx/conf.d/astroshield.conf
        sudo sed -i "s|ssl_certificate_key .*|ssl_certificate_key /etc/ssl/private/astroshield-selfsigned.key;|g" /etc/nginx/conf.d/astroshield.conf
    fi
fi

echo "SSL setup process completed. Please check certificates and Nginx configuration."