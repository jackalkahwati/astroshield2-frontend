# AstroShield Development Guide

This document provides comprehensive guidelines for developers working on the AstroShield platform.

## Development Environment Setup

### Prerequisites

- Python 3.8+
- Node.js 18+
- Docker and Docker Compose
- Git LFS
- PostgreSQL 14+ (local or containerized)
- Kafka (local or containerized)

### Setting Up Your Environment

1. Clone the repository
   ```bash
   git clone https://github.com/your-organization/astroshield.git
   cd astroshield
   ```

2. Install Git LFS and pull large files
   ```bash
   git lfs install
   git lfs pull
   ```

3. Install Python dependencies
   ```bash
   # Install with development extras
   pip install -e ".[dev]"
   ```

4. Install frontend dependencies
   ```bash
   cd frontend
   npm install
   ```

5. Set up environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. Start development services (optional)
   ```bash
   docker-compose up -d postgres kafka redis
   ```

## Running the Application

### Backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 3001
```

The API will be available at http://localhost:3001 with documentation at http://localhost:3001/docs

### Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at http://localhost:3000

## Project Structure

### Backend Structure

```
backend/
├── app/                       # Main application package
│   ├── __init__.py           
│   ├── main.py                # Application entry point
│   ├── core/                  # Core utilities
│   │   ├── config.py          # Configuration
│   │   ├── security.py        # Authentication and authorization
│   │   └── ...
│   ├── db/                    # Database utilities
│   │   ├── session.py         # Database session handling
│   │   └── ...
│   ├── models/                # SQLAlchemy models
│   │   ├── user.py
│   │   └── ...
│   ├── routers/               # API route definitions
│   │   ├── ccdm.py
│   │   ├── dashboard.py
│   │   └── ...
│   ├── services/              # Business logic
│   │   ├── ccdm.py
│   │   └── ...
│   └── middleware/            # Middleware components
│       ├── auth.py
│       └── ...
├── migrations/                # Alembic migrations
└── tests/                     # Backend tests
```

### Frontend Structure

```
frontend/
├── app/                       # Next.js application
│   ├── layout.tsx             # Root layout
│   ├── page.tsx               # Home page
│   ├── dashboard/             # Dashboard route
│   │   └── page.tsx
│   └── ...
├── components/                # React components
│   ├── dashboard/             # Dashboard components
│   │   ├── overview.tsx
│   │   └── ...
│   ├── ui/                    # Shared UI components
│   │   ├── button.tsx
│   │   └── ...
│   └── ...
├── lib/                       # Utilities and helpers
│   ├── api-client.ts          # API client
│   └── ...
└── public/                    # Static assets
```

## Development Workflow

### Branching Strategy

1. Create feature branches from `main`
   ```bash
   git checkout main
   git pull
   git checkout -b feature/my-feature
   ```

2. Make small, focused commits
   ```bash
   git add .
   git commit -m "feat: Add spacecraft trajectory prediction"
   ```

3. Push your branch and create a pull request
   ```bash
   git push -u origin feature/my-feature
   ```

### Code Style

#### Python (Backend)

- Follow PEP 8 guidelines
- Use type hints
- Format code with Black
- Sort imports with isort
- Lint with ruff

```python
"""Example module docstring explaining purpose."""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from fastapi import Depends, HTTPException

from app.core.config import settings
from app.models.spacecraft import Spacecraft


def calculate_trajectory(
    spacecraft_id: str,
    start_time: datetime,
    duration_hours: float = 24.0
) -> Dict[str, List[float]]:
    """
    Calculate spacecraft trajectory over time.
    
    Args:
        spacecraft_id: Unique identifier of the spacecraft
        start_time: Starting time for the calculation
        duration_hours: Duration of trajectory prediction in hours
        
    Returns:
        Dictionary containing position vectors at each time step
        
    Raises:
        HTTPException: If spacecraft not found
    """
    # Implementation details here
    return {"positions": [...], "velocities": [...]}
```

#### TypeScript (Frontend)

- Use ESLint and Prettier
- Leverage TypeScript types
- Use functional components with hooks
- Follow React best practices

```typescript
import { useState, useEffect } from 'react';
import { TrajectoryData, Spacecraft } from '@/types';
import { fetchTrajectory } from '@/lib/api-client';

interface TrajectoryViewerProps {
  spacecraftId: string;
  startTime: Date;
  duration?: number;
}

export const TrajectoryViewer: React.FC<TrajectoryViewerProps> = ({
  spacecraftId,
  startTime,
  duration = 24
}) => {
  const [trajectoryData, setTrajectoryData] = useState<TrajectoryData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const loadTrajectory = async () => {
      try {
        setIsLoading(true);
        const data = await fetchTrajectory(spacecraftId, startTime, duration);
        setTrajectoryData(data);
      } catch (err) {
        setError('Failed to load trajectory data');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadTrajectory();
  }, [spacecraftId, startTime, duration]);
  
  if (isLoading) return <div>Loading trajectory data...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!trajectoryData) return <div>No trajectory data available</div>;
  
  return (
    <div className="trajectory-viewer">
      {/* Render trajectory visualization */}
    </div>
  );
};
```

## Testing Guidelines

### Backend Tests

- Write unit tests using pytest
- Aim for 85% code coverage
- Test edge cases and error handling
- Use fixtures and mocks for external dependencies

Example test:
```python
import pytest
from datetime import datetime
from app.services.ccdm import CCDMService
from unittest.mock import Mock, patch

@pytest.fixture
def ccdm_service():
    """Fixture providing a CCDMService instance."""
    return CCDMService()

def test_analyze_conjunction(ccdm_service):
    """Test successful conjunction analysis."""
    spacecraft_id = "test_spacecraft_1"
    other_spacecraft_id = "test_spacecraft_2"
    
    result = ccdm_service.analyze_conjunction(spacecraft_id, other_spacecraft_id)
    
    assert result['status'] == 'operational'
    assert 'indicators' in result
    assert 'analysis_timestamp' in result
```

### Frontend Tests

- Write tests using Jest and React Testing Library
- Focus on component behavior, not implementation details
- Test user interactions and state changes

Example test:
```typescript
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TrajectoryViewer } from '../components/TrajectoryViewer';
import { fetchTrajectory } from '@/lib/api-client';

// Mock API client
jest.mock('@/lib/api-client', () => ({
  fetchTrajectory: jest.fn()
}));

describe('TrajectoryViewer', () => {
  const mockTrajectoryData = {
    positions: [[1, 2, 3], [2, 3, 4]],
    velocities: [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
  };
  
  beforeEach(() => {
    (fetchTrajectory as jest.Mock).mockResolvedValue(mockTrajectoryData);
  });
  
  it('should render loading state initially', () => {
    render(
      <TrajectoryViewer 
        spacecraftId="test-1" 
        startTime={new Date()} 
      />
    );
    
    expect(screen.getByText(/loading trajectory data/i)).toBeInTheDocument();
  });
  
  it('should render trajectory data when loaded', async () => {
    render(
      <TrajectoryViewer 
        spacecraftId="test-1" 
        startTime={new Date()} 
      />
    );
    
    // Wait for loading to complete
    await waitFor(() => {
      expect(screen.queryByText(/loading trajectory data/i)).not.toBeInTheDocument();
    });
    
    // Verify correct data is displayed
    expect(fetchTrajectory).toHaveBeenCalledWith(
      'test-1', 
      expect.any(Date), 
      24
    );
  });
});
```

## API Design Guidelines

### RESTful API Principles

- Use nouns for resource names, not verbs
- Use plural forms for resource collections
- Use proper HTTP methods:
  - GET: Retrieve resources
  - POST: Create resources
  - PUT: Update resources (full update)
  - PATCH: Update resources (partial update)
  - DELETE: Remove resources

### API Structure

- Group related endpoints in routers
- Version the API (e.g., `/api/v1/resource`)
- Use consistent query parameter naming
- Provide comprehensive documentation

### Response Format

Use a consistent response format:

```json
{
  "data": {
    // Resource data or collection
  },
  "meta": {
    "pagination": {
      "page": 1,
      "per_page": 25,
      "total": 100,
      "total_pages": 4
    }
  },
  "error": null
}
```

For errors:

```json
{
  "data": null,
  "meta": {},
  "error": {
    "code": "resource_not_found",
    "message": "The requested resource was not found",
    "details": {
      // Additional error details
    }
  }
}
```

## Database Management

### Migrations

Using Alembic for database migrations:

```bash
# Create a new migration
alembic revision --autogenerate -m "Add spacecraft table"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

### ORM Usage

Using SQLAlchemy:

```python
from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime
from sqlalchemy.orm import relationship
from app.db.base_class import Base

class Spacecraft(Base):
    __tablename__ = "spacecraft"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    norad_id = Column(String, unique=True, index=True)
    launch_date = Column(DateTime, nullable=True)
    mass_kg = Column(Float, nullable=True)
    operator_id = Column(Integer, ForeignKey("operators.id"))
    
    # Relationships
    operator = relationship("Operator", back_populates="spacecraft")
    maneuvers = relationship("Maneuver", back_populates="spacecraft")
    state_vectors = relationship("StateVector", back_populates="spacecraft")
```

## Message-Driven Architecture

### Kafka Message Structure

All messages should follow the standard format:

```json
{
  "header": {
    "messageId": "uuid-example-12345",
    "traceId": "original-message-uuid-67890",
    "source": "subsystem_name",
    "messageType": "message_category.specific_type",
    "parentMessageIds": ["parent-uuid-1", "parent-uuid-2"],
    "timestamp": "2025-01-01T12:00:00Z"
  },
  "payload": {
    // Message-specific data
  }
}
```

### Producing Messages

```python
from src.asttroshield.common.message_headers import MessageFactory
from src.kafka_client.kafka_publish import KafkaProducer

# Create a message
message = MessageFactory.create_message(
    message_type="ss2.state.estimate",
    source="state_estimator",
    payload={
        "objectId": "spacecraft-1",
        "timestamp": datetime.utcnow().isoformat(),
        "position": [1000.5, 2000.3, 3000.1],
        "velocity": [1.5, -0.3, 0.1]
    }
)

# Publish the message
producer = KafkaProducer(bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS)
producer.publish(
    topic="state-estimates",
    key=message["payload"]["objectId"],
    value=message
)
```

### Consuming Messages

```python
from src.kafka_client.kafka_consume import KafkaConsumer

# Create a consumer
consumer = KafkaConsumer(
    bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
    group_id="ccdm-detection-group",
    topics=["state-estimates"]
)

# Process messages
for message in consumer.consume():
    header = message.get("header", {})
    payload = message.get("payload", {})
    
    # Process the message
    object_id = payload.get("objectId")
    position = payload.get("position")
    
    # Create a response message
    response = MessageFactory.create_derived_message(
        parent_message=message,
        message_type="ss4.ccdm.detection",
        source="ccdm_service",
        payload={
            "objectId": object_id,
            "detectionTime": datetime.utcnow().isoformat(),
            "detectionType": "maneuver",
            "confidence": 0.95
        }
    )
    
    # Publish the response
    producer.publish(
        topic="ccdm-detections",
        key=object_id,
        value=response
    )
```

## Performance Optimization

### Backend Optimization

- Use async endpoints for I/O-bound operations
- Implement caching for expensive operations
- Use database indices for frequently queried fields
- Implement pagination for large data sets
- Profile endpoints to identify bottlenecks

### Frontend Optimization

- Implement code splitting for faster initial load
- Use React.memo and useMemo for expensive components
- Optimize bundle size with tree shaking
- Use image optimization techniques
- Implement virtualization for long lists

## Security Guidelines

### Authentication and Authorization

- Implement JWT-based authentication
- Define and enforce role-based access control
- Use short-lived tokens with refresh mechanism
- Implement proper logout mechanism

### API Security

- Validate all input
- Implement rate limiting
- Use secure headers
- Implement CORS properly
- Protect against common attacks (SQLi, XSS, CSRF)

### Data Security

- Encrypt sensitive data at rest
- Use TLS for all communications
- Implement proper data access controls
- Follow the principle of least privilege

## Troubleshooting and Common Issues

### Database Issues

- Connection pool exhaustion
  - Check for unclosed connections
  - Verify connection pool settings
- Slow queries
  - Check for missing indices
  - Examine query plans with EXPLAIN
- Migration errors
  - Check for conflicts in migration history
  - Use alembic history to diagnose issues

### API Issues

- Rate limiting
  - Verify rate limit configuration
  - Implement proper backoff in clients
- Authentication failures
  - Check token validity and expiration
  - Verify secret keys
- Performance issues
  - Profile endpoints to identify bottlenecks
  - Check for N+1 query problems

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)