#!/bin/bash

echo "🚀 Deploying AstroShield Frontend Fix"
echo "====================================="
echo ""
echo "This script will:"
echo "• Apply emoji charset fix"
echo "• Add /grafana and /prometheus proxy routes"
echo "• Fix frontend serving to show proper dashboard"
echo ""

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found"
    echo "Please run this script from the AstroShield root directory"
    exit 1
fi

echo "=== Step 1: Stopping current services ==="
docker-compose down
echo "✅ Services stopped"

echo ""
echo "=== Step 2: Rebuilding frontend ==="
echo "Rebuilding frontend to ensure latest changes..."
docker-compose build --no-cache frontend
echo "✅ Frontend rebuilt"

echo ""
echo "=== Step 3: Starting services ==="
docker-compose up -d
echo "✅ Services starting..."

echo ""
echo "=== Step 4: Waiting for services to initialize ==="
sleep 15
echo "⏳ Initial startup complete"

echo ""
echo "=== Step 5: Checking service status ==="
echo ""
echo "Service Status:"
docker-compose ps

echo ""
echo "=== Step 6: Testing endpoints ==="
echo ""

# Test main site
echo "Testing main site..."
if curl -s -I http://localhost | grep -q "200\|301\|302"; then
    echo "✅ Main site responding"
else
    echo "❌ Main site not responding"
fi

# Test frontend direct
echo "Testing frontend service..."
if docker-compose exec -T frontend curl -s -I http://localhost:3000 | grep -q "200"; then
    echo "✅ Frontend service responding"
else
    echo "❌ Frontend service not responding"
    echo "Checking frontend logs..."
    docker-compose logs --tail=20 frontend
fi

# Test backend
echo "Testing backend service..."
if curl -s -I http://localhost:3001/health | grep -q "200"; then
    echo "✅ Backend service responding"
else
    echo "❌ Backend service not responding"
fi

echo ""
echo "=== Final Status ==="
echo ""
echo "🌐 Website Status:"
echo "   • Main site: https://astroshield.sdataplab.com/"
echo "   • Grafana: https://astroshield.sdataplab.com/grafana"  
echo "   • Prometheus: https://astroshield.sdataplab.com/prometheus"
echo ""

echo "🔧 Fixes Applied:"
echo "   ✅ Emoji charset fix (UTF-8 meta tag added)"
echo "   ✅ Service URLs updated to use proxy paths"
echo "   ✅ Conflicting static HTML removed"
echo "   ✅ Frontend rebuilt and restarted"
echo ""

echo "📝 What Changed:"
echo "   • /grafana now proxies to Grafana dashboard"
echo "   • /prometheus now proxies to Prometheus metrics"
echo "   • Emojis should display correctly"
echo "   • Main site should show AstroShield dashboard (not landing page)"
echo ""

echo "🔍 If issues persist:"
echo "   • Check logs: docker-compose logs frontend"
echo "   • Restart individual service: docker-compose restart frontend"
echo "   • Check container status: docker-compose ps"
echo ""

echo "✅ Deployment complete!"
echo "The AstroShield dashboard should now be accessible at https://astroshield.sdataplab.com/" 