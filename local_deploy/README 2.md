# AstroShield Local Deployment

This is a local deployment of the AstroShield application. It includes both the backend and frontend components.

## Starting the Application

Run the start script to start both the backend and frontend:

```bash
./start.sh
```

After starting, you can access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:3001

## Stopping the Application

Run the stop script to stop all services:

```bash
./stop.sh
```

## Logs

Logs are available in:
- Backend: `backend/backend.log`
- Frontend: `frontend/frontend.log`

## Structure

- `backend/` - Contains the Python backend server
- `frontend/` - Contains the Next.js frontend application