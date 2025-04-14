# AstroShield Modular Backend

This directory contains the modular backend for the AstroShield application. The architecture is designed with clean separation of concerns and domain boundaries.

## Modules

### Trajectory Analysis (`app/trajectory`)
- Handles trajectory analysis and predictions
- Provides endpoints for analyzing satellite trajectories
- Simulates object reentry and breakup events

### Maneuvers Management (`app/maneuvers`)
- Manages spacecraft maneuver planning and execution
- Provides CRUD operations for maneuver records
- Supports station-keeping, collision avoidance, and other maneuver types

### CCDM - Conjunction and Collision Data Management (`app/ccdm`)
- Handles conjunction data messages and collision predictions
- Provides endpoints for retrieving and filtering conjunction events
- Manages primary and secondary object data

### Common Utilities (`app/common`)
- Shared utilities like logging, error handling, etc.
- Common functionality used across modules

## Running the Application

### Using Docker
```bash
# Build and start the backend only
docker-compose -f docker-compose.backend.yml up -d

# Build and start the complete application
docker-compose -f docker-compose.simple.yml up -d
```

### Using Scripts
```bash
# Start the application
./docker-start.sh

# Stop the application
./docker-stop.sh
```

## API Endpoints

- Health Check: `/health`
- API Health: `/api/v1/health`
- Trajectory Analysis: `/api/trajectory/analyze`
- Maneuvers: `/api/v1/maneuvers`
- Conjunctions: `/api/v1/conjunctions`

## Design Principles

This modular architecture follows these principles:
1. **Separation of Concerns**: Each module has its own models, services, and routers
2. **Domain-Driven Design**: Modules are organized around business domains
3. **Clean Interfaces**: Clear and well-defined interfaces between modules
4. **Modularity**: Easily extendable with new modules

## Future Extensions

The modular architecture provides a foundation for future microservices if needed:
1. Each module could be extracted into its own microservice
2. Shared models could be moved into a common library
3. Data persistence could be separated by domain 