#!/bin/bash
cd $(dirname "$0")

# Start backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
nohup python3 minimal_server.py > backend.log 2>&1 &
echo $! > backend.pid
cd ..

# Start frontend
cd frontend
npm install
PORT=3000 nohup npm start > frontend.log 2>&1 &
echo $! > frontend.pid
cd ..

echo "AstroShield deployment completed!"
echo "Backend running at http://localhost:3001"
echo "Frontend running at http://localhost:3000"