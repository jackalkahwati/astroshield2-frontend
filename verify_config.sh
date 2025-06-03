#!/bin/bash

echo "🔍 Verifying AstroShield configuration changes..."
echo ""

# Check if charset meta tag was added
echo "=== Checking HTML Files ==="
if grep -q 'meta charset="UTF-8"' astroshield_landing.html; then
    echo "✅ Charset meta tag added to astroshield_landing.html"
else
    echo "❌ Charset meta tag missing in astroshield_landing.html"
fi

# Check if URLs were updated
if grep -q '/grafana' astroshield_landing.html; then
    echo "✅ Grafana URL updated to use proxy path"
else
    echo "❌ Grafana URL still using direct IP"
fi

if grep -q '/prometheus' astroshield_landing.html; then
    echo "✅ Prometheus URL updated to use proxy path"
else
    echo "❌ Prometheus URL still using direct IP"
fi

echo ""
echo "=== Checking Nginx Configuration ==="
if grep -q 'upstream grafana' nginx/nginx.conf; then
    echo "✅ Grafana upstream configured"
else
    echo "❌ Grafana upstream missing"
fi

if grep -q 'upstream prometheus' nginx/nginx.conf; then
    echo "✅ Prometheus upstream configured"
else
    echo "❌ Prometheus upstream missing"
fi

if grep -q 'location /grafana/' nginx/nginx.conf; then
    echo "✅ Grafana proxy route configured"
else
    echo "❌ Grafana proxy route missing"
fi

if grep -q 'location /prometheus/' nginx/nginx.conf; then
    echo "✅ Prometheus proxy route configured"
else
    echo "❌ Prometheus proxy route missing"
fi

echo ""
echo "=== Docker Compose Services ==="
if grep -q 'grafana:' docker-compose.yml; then
    echo "✅ Grafana service defined in docker-compose.yml"
else
    echo "❌ Grafana service missing from docker-compose.yml"
fi

if grep -q 'prometheus:' docker-compose.yml; then
    echo "✅ Prometheus service defined in docker-compose.yml"
else
    echo "❌ Prometheus service missing from docker-compose.yml"
fi

echo ""
echo "🎯 Summary of changes made:"
echo "   1. Added <meta charset=\"UTF-8\"> to fix emoji display"
echo "   2. Updated Grafana URL from http://56.136.120.99:3000 → /grafana"
echo "   3. Updated Prometheus URL from http://56.136.120.99:9090 → /prometheus"
echo "   4. Added nginx upstream definitions for grafana and prometheus"
echo "   5. Added nginx location blocks for /grafana/ and /prometheus/"
echo ""
echo "📋 After restart, services will be available at:"
echo "   • Main site: https://astroshield.sdataplab.com/"
echo "   • Grafana: https://astroshield.sdataplab.com/grafana"
echo "   • Prometheus: https://astroshield.sdataplab.com/prometheus" 