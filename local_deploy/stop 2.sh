#!/bin/bash
cd $(dirname "$0")

# Stop backend
if [ -f backend/backend.pid ]; then
  kill -9 $(cat backend/backend.pid) 2>/dev/null || true
  rm backend/backend.pid
fi

# Stop frontend
if [ -f frontend/frontend.pid ]; then
  kill -9 $(cat frontend/frontend.pid) 2>/dev/null || true
  rm frontend/frontend.pid
fi

echo "AstroShield services stopped"