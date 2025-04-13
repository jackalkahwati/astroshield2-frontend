#!/bin/bash
sudo rm -f /etc/nginx/conf.d/astroshield-ssl.conf.disabled
echo "Creating HTTP-only configuration..."
cat << EOC | sudo tee /etc/nginx/conf.d/astroshield-http.conf
