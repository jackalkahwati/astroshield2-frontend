#!/bin/bash

echo "🔧 Fixing Frontend Serving Issue"
echo "================================"
echo ""

# Step 1: Backup and remove conflicting static files
echo "=== Step 1: Removing conflicting static files ==="
if [ -f "index.html" ]; then
    echo "📦 Backing up index.html to index.html.backup"
    mv index.html index.html.backup
    echo "✅ Removed conflicting index.html from root"
else
    echo "✅ No conflicting index.html found"
fi

# Step 2: Check if there are any nginx static file mounts
echo ""
echo "=== Step 2: Checking nginx configuration ==="
if grep -q "try_files" nginx/nginx.conf; then
    echo "❌ Found try_files directive in nginx config - might be serving static files"
    echo "Current nginx config should only proxy to frontend service"
else
    echo "✅ No static file serving detected in nginx config"
fi

# Step 3: Ensure frontend service is properly configured
echo ""
echo "=== Step 3: Checking frontend configuration ==="
if [ -f "frontend/package.json" ]; then
    echo "✅ Frontend package.json exists"
    
    # Check if Next.js is configured properly
    if grep -q "next start" frontend/package.json; then
        echo "✅ Next.js start script configured"
    else
        echo "❌ Next.js start script missing"
    fi
    
    # Check if we have a proper Next.js app structure
    if [ -f "frontend/app/page.tsx" ] || [ -f "frontend/src/app/page.tsx" ]; then
        echo "✅ Next.js App Router structure detected"
    elif [ -f "frontend/pages/index.tsx" ] || [ -f "frontend/pages/index.js" ]; then
        echo "✅ Next.js Pages Router structure detected" 
    else
        echo "❌ No valid Next.js page structure found"
    fi
else
    echo "❌ No frontend package.json found"
fi

# Step 4: Check for any other static HTML files that might interfere
echo ""
echo "=== Step 4: Checking for other static files ==="
static_files=("simple_landing.html" "astroshield_landing.html" "index-static.html")
for file in "${static_files[@]}"; do
    if [ -f "$file" ]; then
        echo "📄 Found $file - this should NOT be served as the main frontend"
    fi
done

# Step 5: Create proper nginx test
echo ""
echo "=== Step 5: Nginx configuration validation ==="
echo "Current frontend proxy configuration:"
grep -A 8 "Frontend routes" nginx/nginx.conf || echo "No frontend routes section found"

# Step 6: Provide restart instructions
echo ""
echo "=== Step 6: Restart Instructions ==="
echo "To apply fixes, run these commands on the server:"
echo ""
echo "1. Stop services:"
echo "   docker-compose down"
echo ""
echo "2. Rebuild frontend (in case of build issues):"
echo "   docker-compose build frontend"
echo ""
echo "3. Start services:"
echo "   docker-compose up -d"
echo ""
echo "4. Check service status:"
echo "   docker-compose ps"
echo "   docker-compose logs frontend"
echo ""

# Step 7: Create verification commands
echo "=== Step 7: Verification Commands ==="
echo "After restart, verify with these commands:"
echo ""
echo "# Check if frontend container is running:"
echo "docker-compose ps frontend"
echo ""
echo "# Check frontend logs:"
echo "docker-compose logs -f frontend"
echo ""
echo "# Test direct frontend access (from server):"
echo "curl -I http://localhost:3000"
echo ""
echo "# Test through nginx (from server):"
echo "curl -I http://localhost"
echo ""

echo "✅ Frontend serving fix completed!"
echo ""
echo "🎯 Summary:"
echo "   • Removed conflicting index.html from root"
echo "   • Nginx should now properly proxy to Next.js frontend"
echo "   • Frontend service needs to be rebuilt and restarted"
echo "   • Use verification commands to confirm fix" 