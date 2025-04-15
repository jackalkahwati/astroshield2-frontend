#!/bin/bash
sudo nginx -t
sudo systemctl restart nginx
echo "Checking services on ports 3000 and 3001:"
sudo lsof -i :3000,3001
echo "Recent Nginx error logs:"
sudo tail -n 20 /var/log/nginx/error.log
