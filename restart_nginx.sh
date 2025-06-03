#!/bin/bash

echo "🔄 Restarting AstroShield services to apply nginx configuration changes..."

# Stop current services
echo "📛 Stopping current services..."
docker-compose down

# Rebuild and restart services
echo "🚀 Starting services with updated configuration..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check service status
echo "🔍 Checking service status..."
echo ""
echo "=== Service Status ==="
docker-compose ps

echo ""
echo "=== Testing Endpoints ==="
echo "🌐 Main site: https://astroshield.sdataplab.com/"
echo "📊 Grafana: https://astroshield.sdataplab.com/grafana"
echo "📈 Prometheus: https://astroshield.sdataplab.com/prometheus"
echo ""

# Test health endpoints
echo "=== Health Checks ==="
echo "Testing main health endpoint..."
curl -s -o /dev/null -w "Health endpoint: %{http_code}\n" https://astroshield.sdataplab.com/health || echo "❌ Health endpoint not accessible"

echo ""
echo "✅ Configuration updated successfully!"
echo "📝 Changes made:"
echo "   • Added charset=UTF-8 to fix emoji display"
echo "   • Added /grafana proxy route"
echo "   • Added /prometheus proxy route"
echo "   • Updated service URLs in landing page" 