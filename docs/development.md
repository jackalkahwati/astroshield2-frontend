# Development Guide

## Overview

This guide provides instructions for setting up and developing the AstroShield platform locally.

## Prerequisites

### Required Software

1. Development Tools
   ```bash
   # macOS (using Homebrew)
   brew install git node python@3.11 poetry docker docker-compose kubectl

   # Ubuntu/Debian
   apt update && apt install -y git nodejs python3.11 python3-pip docker.io docker-compose kubectl
   
   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. IDE Setup
   - VSCode with extensions:
     - Python
     - TypeScript
     - ESLint
     - Prettier
     - Docker
     - Kubernetes
     - GitLens

3. Environment Setup
   ```bash
   # Clone repository
   git clone https://github.com/your-org/astroshield.git
   cd astroshield
   
   # Copy environment files
   cp .env.example .env
   cp frontend/.env.example frontend/.env
   cp backend/.env.example backend/.env
   ```

## Local Development

### Backend Setup

1. Install Dependencies
   ```bash
   # Navigate to backend directory
   cd backend
   
   # Install dependencies
   poetry install
   
   # Activate virtual environment
   poetry shell
   ```

2. Database Setup
   ```bash
   # Start PostgreSQL
   docker-compose up -d postgres
   
   # Apply migrations
   poetry run alembic upgrade head
   
   # Seed initial data
   poetry run python scripts/seed.py
   ```

3. Run Backend
   ```bash
   # Start backend server
   poetry run uvicorn app.main:app --reload --port 8000
   
   # Run tests
   poetry run pytest
   
   # Run linting
   poetry run flake8
   poetry run black .
   ```

### Frontend Setup

1. Install Dependencies
   ```bash
   # Navigate to frontend directory
   cd frontend
   
   # Install dependencies
   npm install
   ```

2. Development Server
   ```bash
   # Start development server
   npm run dev
   
   # Run tests
   npm test
   
   # Run linting
   npm run lint
   ```

### Docker Development

1. Build Images
   ```bash
   # Build all services
   docker-compose build
   
   # Build specific service
   docker-compose build backend
   docker-compose build frontend
   ```

2. Run Services
   ```bash
   # Start all services
   docker-compose up -d
   
   # View logs
   docker-compose logs -f
   
   # Stop services
   docker-compose down
   ```

## Development Workflow

### Git Workflow

1. Branch Naming
   ```bash
   # Feature branch
   git checkout -b feature/add-new-endpoint
   
   # Bug fix branch
   git checkout -b fix/resolve-auth-issue
   
   # Hotfix branch
   git checkout -b hotfix/critical-security-fix
   ```

2. Commit Messages
   ```bash
   # Feature commit
   git commit -m "feat: add new endpoint for satellite tracking"
   
   # Fix commit
   git commit -m "fix: resolve authentication token expiry issue"
   
   # Docs commit
   git commit -m "docs: update API documentation"
   ```

### Code Style

1. Python Style
   ```python
   # Example function with type hints
   def calculate_orbit(
       satellite_id: str,
       timestamp: datetime,
       *,
       include_velocity: bool = False
   ) -> Dict[str, Any]:
       """
       Calculate satellite orbit parameters.
       
       Args:
           satellite_id: Unique identifier of the satellite
           timestamp: Time at which to calculate orbit
           include_velocity: Whether to include velocity vectors
           
       Returns:
           Dictionary containing orbit parameters
       """
       # Function implementation
   ```

2. TypeScript Style
   ```typescript
   // Example interface and component
   interface SatelliteData {
     id: string;
     name: string;
     orbit: {
       altitude: number;
       inclination: number;
     };
   }
   
   const SatelliteDisplay: React.FC<{ data: SatelliteData }> = ({ data }) => {
     const { name, orbit } = data;
     
     return (
       <div className="satellite-info">
         <h2>{name}</h2>
         <p>Altitude: {orbit.altitude} km</p>
         <p>Inclination: {orbit.inclination}°</p>
       </div>
     );
   };
   ```

### Testing

1. Backend Tests
   ```python
   # Example test case
   def test_calculate_orbit():
       # Arrange
       satellite_id = "TEST-SAT-001"
       timestamp = datetime.utcnow()
       
       # Act
       result = calculate_orbit(satellite_id, timestamp)
       
       # Assert
       assert "altitude" in result
       assert "inclination" in result
       assert isinstance(result["altitude"], float)
   ```

2. Frontend Tests
   ```typescript
   // Example test case
   describe('SatelliteDisplay', () => {
     it('renders satellite information correctly', () => {
       // Arrange
       const testData = {
         id: 'TEST-SAT-001',
         name: 'Test Satellite',
         orbit: {
           altitude: 500,
           inclination: 45,
         },
       };
       
       // Act
       render(<SatelliteDisplay data={testData} />);
       
       // Assert
       expect(screen.getByText('Test Satellite')).toBeInTheDocument();
       expect(screen.getByText('Altitude: 500 km')).toBeInTheDocument();
     });
   });
   ```

## API Development

### Adding New Endpoints

1. Router Setup
   ```python
   # backend/app/routers/satellites.py
   from fastapi import APIRouter, Depends
   from typing import List
   
   router = APIRouter(prefix="/satellites", tags=["satellites"])
   
   @router.get("/")
   async def list_satellites(
       orbit_type: str = None,
       status: str = None
   ) -> List[dict]:
       """List all satellites with optional filtering."""
       # Implementation
   ```

2. Schema Definition
   ```python
   # backend/app/schemas/satellite.py
   from pydantic import BaseModel
   from datetime import datetime
   
   class SatelliteBase(BaseModel):
       name: str
       norad_id: str
       launch_date: datetime
       
   class SatelliteCreate(SatelliteBase):
       pass
       
   class Satellite(SatelliteBase):
       id: str
       created_at: datetime
       updated_at: datetime
       
       class Config:
           orm_mode = True
   ```

### Error Handling

1. Custom Exceptions
   ```python
   # backend/app/core/exceptions.py
   from fastapi import HTTPException
   
   class SatelliteNotFound(HTTPException):
       def __init__(self, satellite_id: str):
           super().__init__(
               status_code=404,
               detail=f"Satellite {satellite_id} not found"
           )
   ```

2. Error Handlers
   ```python
   # backend/app/core/handlers.py
   from fastapi import Request
   from fastapi.responses import JSONResponse
   
   async def satellite_not_found_handler(
       request: Request,
       exc: SatelliteNotFound
   ) -> JSONResponse:
       return JSONResponse(
           status_code=exc.status_code,
           content={"error": exc.detail}
       )
   ```

## Database Management

### Migrations

1. Create Migration
   ```bash
   # Generate new migration
   poetry run alembic revision --autogenerate -m "add satellite table"
   
   # Apply migration
   poetry run alembic upgrade head
   
   # Rollback migration
   poetry run alembic downgrade -1
   ```

2. Migration Script
   ```python
   # backend/migrations/versions/xxx_add_satellite_table.py
   def upgrade():
       op.create_table(
           'satellites',
           sa.Column('id', sa.String(), nullable=False),
           sa.Column('name', sa.String(), nullable=False),
           sa.Column('norad_id', sa.String(), nullable=False),
           sa.Column('launch_date', sa.DateTime(), nullable=False),
           sa.PrimaryKeyConstraint('id')
       )
   
   def downgrade():
       op.drop_table('satellites')
   ```

## Troubleshooting

### Common Issues

1. Database Connection
   ```bash
   # Check database status
   docker-compose ps postgres
   
   # View database logs
   docker-compose logs postgres
   
   # Reset database
   docker-compose down -v postgres
   docker-compose up -d postgres
   ```

2. Frontend Issues
   ```bash
   # Clear node modules
   rm -rf node_modules
   npm install
   
   # Clear next.js cache
   rm -rf .next
   npm run dev
   ```

### Debugging

1. Backend Debugging
   ```python
   # Add debug logging
   import logging
   
   logger = logging.getLogger(__name__)
   logger.setLevel(logging.DEBUG)
   
   logger.debug("Processing satellite data", extra={
       "satellite_id": satellite_id,
       "timestamp": timestamp
   })
   ```

2. Frontend Debugging
   ```typescript
   // Add debug logging
   const debug = require('debug')('app:satellite');
   
   debug('Rendering satellite data', {
     id: satellite.id,
     name: satellite.name
   });
   ```

## Performance Optimization

### Backend Optimization

1. Query Optimization
   ```python
   # Use select_related for related fields
   satellites = (
       db.query(Satellite)
       .select_related("orbit")
       .filter(Satellite.status == "active")
       .all()
   )
   
   # Use pagination
   satellites = (
       db.query(Satellite)
       .offset(skip)
       .limit(limit)
       .all()
   )
   ```

2. Caching
   ```python
   # Use Redis for caching
   @router.get("/{satellite_id}")
   async def get_satellite(
       satellite_id: str,
       redis: Redis = Depends(get_redis)
   ):
       # Check cache
       cached = await redis.get(f"satellite:{satellite_id}")
       if cached:
           return json.loads(cached)
           
       # Fetch and cache
       satellite = await get_satellite_data(satellite_id)
       await redis.set(
           f"satellite:{satellite_id}",
           json.dumps(satellite),
           ex=3600
       )
       return satellite
   ```

### Frontend Optimization

1. Component Optimization
   ```typescript
   // Use memo for expensive calculations
   const orbitParameters = useMemo(() => {
     return calculateOrbitParameters(satellite.data);
   }, [satellite.data]);
   
   // Use callback for event handlers
   const handleOrbitUpdate = useCallback(() => {
     updateOrbit(satellite.id);
   }, [satellite.id]);
   ```

2. Data Fetching
   ```typescript
   // Use SWR for data fetching
   const { data, error } = useSWR(
     `/api/satellites/${id}`,
     fetcher,
     {
       revalidateOnFocus: false,
       refreshInterval: 60000
     }
   );
   ``` 