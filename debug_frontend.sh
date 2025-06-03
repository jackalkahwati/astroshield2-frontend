#!/bin/bash

echo "🔍 Debugging Frontend Service Issue"
echo "=================================="
echo ""

# Check if we have the static landing file taking precedence
echo "=== Checking for static files ==="
if [ -f "index.html" ]; then
    echo "❌ Found index.html in root directory - this might be overriding the frontend"
    ls -la index.html
else
    echo "✅ No conflicting index.html in root"
fi

if [ -f "simple_landing.html" ]; then
    echo "📄 Found simple_landing.html"
    ls -la simple_landing.html
fi

if [ -f "astroshield_landing.html" ]; then
    echo "📄 Found astroshield_landing.html"
    ls -la astroshield_landing.html
fi

echo ""
echo "=== Checking nginx volumes ==="
echo "Current nginx configuration mounts:"
grep -A 5 "volumes:" docker-compose.yml | grep nginx -A 10

echo ""
echo "=== Checking if there's a static file server ==="
# Check if nginx is serving static files instead of proxying
if grep -q "root " nginx/nginx.conf; then
    echo "❌ Found 'root' directive in nginx config - static files being served"
    grep -n "root " nginx/nginx.conf
else
    echo "✅ No static file serving in nginx config"
fi

echo ""
echo "=== Docker container status ==="
echo "Note: Docker not running locally, but checking configuration..."

echo ""
echo "=== Frontend package.json check ==="
if [ -f "frontend/package.json" ]; then
    echo "✅ Frontend package.json exists"
    echo "Scripts:"
    cat frontend/package.json | grep -A 10 '"scripts"'
else
    echo "❌ No frontend/package.json found"
fi

echo ""
echo "=== Recommendations ==="
echo "1. Check if frontend container is running: docker-compose ps"
echo "2. Check frontend logs: docker-compose logs frontend"
echo "3. Verify frontend is listening on port 3000: docker-compose exec frontend netstat -tlnp"
echo "4. Test direct frontend access: curl http://frontend:3000 (from within nginx container)"
echo ""
echo "🎯 Most likely issues:"
echo "   • Frontend container failed to start"
echo "   • Frontend build failed"
echo "   • Static HTML file being served instead of Next.js app"
echo "   • nginx proxying to wrong service" 