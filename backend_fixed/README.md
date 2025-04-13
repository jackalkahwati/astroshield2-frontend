# AstroShield Backend

Backend API for the AstroShield satellite protection system, built with FastAPI and PostgreSQL.

## Features

- REST API for satellite management, analysis, and CCDM functionality
- PostgreSQL database for data persistence
- JWT authentication for secure API access
- Docker integration for easy deployment

## Prerequisites

- Python 3.9+
- PostgreSQL 14+
- Docker and Docker Compose (optional, for containerized setup)

## Quick Start

### 1. Setup Environment

Clone the repository and navigate to the backend directory:

```bash
cd backend_fixed
```

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example .env file:

```bash
cp .env.example .env
```

Edit the .env file with your configuration values.

### 3. Setup Database

Run the database setup script:

```bash
./setup_database.sh
```

This script will:
- Start PostgreSQL if it's not running (using Docker)
- Create the database if it doesn't exist
- Create tables and load sample data

Alternatively, you can manually initialize the database:

```bash
# Set the DATABASE_URL environment variable
export DATABASE_URL="postgresql://postgres:password@localhost:5432/astroshield"

# Create tables and load sample data
python scripts/init_database.py
```

### 4. Run the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

### 5. API Documentation

- Swagger UI: http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc

## Docker Deployment

To run the entire application with Docker Compose:

```bash
docker-compose up -d
```

This will start the following services:
- Backend API
- PostgreSQL database
- Frontend (if available)
- Nginx for serving the frontend

## Directory Structure

```
backend_fixed/
├── app/                    # Application package
│   ├── core/               # Core functionality
│   ├── db/                 # Database models and session
│   ├── models/             # Pydantic models and SQLAlchemy ORM
│   ├── routers/            # API endpoints
│   ├── services/           # Business logic
│   └── main.py             # FastAPI application
├── scripts/                # Utility scripts
│   └── init_database.py    # Database initialization script
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
└── setup_database.sh       # Database setup script
```

## Database Schema

The database includes the following tables:
- `users`: User accounts and authentication
- `ccdm_analyses`: CCDM analysis records
- `threat_assessments`: Threat assessment records
- `analysis_results`: Individual analysis results
- `historical_analyses`: Historical analysis records
- `shape_changes`: Shape change detection records

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Proprietary - Copyright (c) 2024 AstroShield 