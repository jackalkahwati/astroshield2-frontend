#!/bin/bash

# AstroShield Complete Platform Startup Script
# This starts all backend services + the full-featured frontend

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Complete AstroShield Platform...${NC}"

# Function to check if port is in use
check_port() {
    if lsof -ti :$1 > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Port $1 is already in use${NC}"
        return 1
    fi
    return 0
}

# Stop any existing processes
echo -e "${YELLOW}🧹 Cleaning up existing processes...${NC}"
docker-compose -f docker-compose.backend-stack.yml down --remove-orphans 2>/dev/null || true
pkill -f "npm.*dev" || true
pkill -f "next.*dev" || true

# Kill processes on key ports to prevent conflicts
echo -e "${YELLOW}🔌 Freeing up ports...${NC}"
for port in 3000 3001 3002 5432 6379 8080 9090 9092 2181; do
    if lsof -ti :$port > /dev/null 2>&1; then
        echo -e "${YELLOW}  Killing process on port $port...${NC}"
        lsof -ti :$port | xargs kill -9 2>/dev/null || true
    fi
done
sleep 2

# Start backend services with Docker
echo -e "${BLUE}🔧 Starting backend services (Docker)...${NC}"
docker-compose -f docker-compose.backend-stack.yml up -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for backend services to initialize...${NC}"
sleep 10

# Check backend health
echo -e "${BLUE}🏥 Checking backend health...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:3001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend API is healthy${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Backend API failed to start properly${NC}"
        exit 1
    fi
    sleep 2
done

# Start the correct frontend
echo -e "${BLUE}🖥️  Starting AstroShield Frontend...${NC}"
cd astroshield-production/frontend

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
    npm install
fi

# Start frontend in background
echo -e "${GREEN}🎯 Launching frontend on http://localhost:3000${NC}"
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to be ready
echo -e "${YELLOW}⏳ Waiting for frontend to initialize...${NC}"
sleep 8

# Check frontend health
for i in {1..15}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Frontend is ready${NC}"
        break
    fi
    if [ $i -eq 15 ]; then
        echo -e "${RED}❌ Frontend failed to start${NC}"
        kill $FRONTEND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# Save frontend PID for cleanup
echo $FRONTEND_PID > /tmp/astroshield_frontend.pid

echo
echo -e "${GREEN}🎉 AstroShield Platform Successfully Started!${NC}"
echo
echo -e "${BLUE}📊 Available Services:${NC}"
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  🖥️  Frontend Dashboard:     http://localhost:3000          │"
echo "│  🔧 Backend API:             http://localhost:3001          │"
echo "│  📚 API Documentation:       http://localhost:3001/docs     │"
echo "│  📊 Grafana Monitoring:      http://localhost:3002          │"
echo "│       └─ Login: admin/admin                                 │"
echo "│  📈 Prometheus Metrics:      http://localhost:9090          │"
echo "│  📨 Kafka UI:                http://localhost:8080          │"
echo "│  🗄️  PostgreSQL Database:    localhost:5432                │"
echo "│  ⚡ Redis Cache:             localhost:6379                │"
echo "└─────────────────────────────────────────────────────────────┘"
echo
echo -e "${BLUE}🛑 To stop all services:${NC} ./stop_complete_astroshield.sh"
echo -e "${BLUE}📝 View logs:${NC} docker-compose -f docker-compose.backend-stack.yml logs -f"
echo
echo -e "${GREEN}✨ Ready to explore AstroShield! Open http://localhost:3000${NC}" 