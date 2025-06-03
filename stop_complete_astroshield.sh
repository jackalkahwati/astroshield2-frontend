#!/bin/bash

# AstroShield Complete Platform Stop Script
# This stops all backend services and frontend

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping AstroShield Platform...${NC}"

# Stop frontend process
if [ -f /tmp/astroshield_frontend.pid ]; then
    FRONTEND_PID=$(cat /tmp/astroshield_frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}🔪 Stopping frontend (PID: $FRONTEND_PID)...${NC}"
        kill $FRONTEND_PID
        sleep 2
        # Force kill if still running
        if ps -p $FRONTEND_PID > /dev/null 2>&1; then
            kill -9 $FRONTEND_PID 2>/dev/null || true
        fi
    fi
    rm -f /tmp/astroshield_frontend.pid
fi

# Kill any remaining Next.js processes
echo -e "${YELLOW}🧹 Cleaning up remaining Node.js processes...${NC}"
pkill -f "npm.*dev" || true
pkill -f "next.*dev" || true
pkill -f "node.*next" || true

# Stop Docker services
echo -e "${YELLOW}🐳 Stopping Docker services...${NC}"
docker-compose -f docker-compose.backend-stack.yml down --remove-orphans

# Optional: Clean up Docker volumes (uncomment if you want to reset data)
# echo -e "${YELLOW}🗑️  Cleaning up Docker volumes...${NC}"
# docker-compose -f docker-compose.backend-stack.yml down -v

echo -e "${GREEN}✅ AstroShield Platform stopped successfully!${NC}"
echo
echo -e "${BLUE}📋 All services have been stopped:${NC}"
echo "  • Frontend Dashboard (port 3000)"
echo "  • Backend API (port 3001)"
echo "  • Grafana (port 3002)"
echo "  • Prometheus (port 9090)"
echo "  • Kafka UI (port 8080)"
echo "  • PostgreSQL Database (port 5432)"
echo "  • Redis Cache (port 6379)"
echo
echo -e "${BLUE}🚀 To restart: ./start_complete_astroshield.sh${NC}" 