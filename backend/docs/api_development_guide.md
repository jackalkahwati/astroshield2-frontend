# AstroShield API Development Guide

This guide provides information for developers who are contributing to or extending the AstroShield API.

## Development Environment Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Git
- Docker and Docker Compose (optional, for containerized development)

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-org/astroshield.git
cd astroshield
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Installs development dependencies
```

4. **Set up environment variables**

Create a `.env` file in the root directory:

```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/astroshield
SECRET_KEY=your_secret_key_here
DEBUG=True
ENVIRONMENT=development
```

5. **Database setup**

```bash
# Create the database
createdb astroshield

# Run migrations
alembic upgrade head
```

6. **Run the development server**

```bash
uvicorn backend.app.main:app --reload --port 3001
```

## Project Structure

```
backend/
├── app/
│   ├── api/            # API endpoints organized by resource
│   ├── core/           # Core functionality, config, security
│   ├── db/             # Database models and session management
│   ├── models/         # Pydantic models (request/response schemas)
│   ├── services/       # Business logic
│   ├── tests/          # Tests
│   └── main.py         # Application entry point
├── docs/               # Documentation
├── scripts/            # Utility scripts
└── alembic/            # Database migrations
```

## API Design Guidelines

### RESTful Principles

- Use nouns, not verbs in endpoint paths
- Use HTTP methods appropriately:
  - GET: Retrieve resources
  - POST: Create resources
  - PUT: Update resources (full update)
  - PATCH: Partial updates
  - DELETE: Remove resources
- Return appropriate status codes
- Use plural nouns for collection endpoints

### Versioning

- All API endpoints should be versioned (e.g., `/api/v1/satellites`)
- Maintain backward compatibility within a version

### Authentication

- Use JWT-based authentication
- Include the token in the Authorization header: `Authorization: Bearer <token>`
- Implement token refresh mechanism

### Response Format

All API responses should follow a consistent format:

```json
{
  "data": {}, // The resource or collection (may be null for non-200 responses)
  "meta": {}, // Pagination info, counts, etc.
  "error": {} // Error details (null for successful responses)
}
```

### Pagination

For collection endpoints, implement pagination with the following query parameters:
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20, max: 100)

Response should include pagination metadata:

```json
{
  "data": [...],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 145,
    "pages": 8
  }
}
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=backend

# Run specific test file
pytest backend/app/tests/api/test_satellites.py
```

### Writing Tests

- Write unit tests for service functions
- Write integration tests for API endpoints
- Use fixtures for common setup
- Mock external dependencies

## Documentation

- All API endpoints must be documented with OpenAPI/Swagger
- Document all function parameters and return values
- Keep the Swagger UI documentation up to date

## Error Handling

- Use custom exception classes
- Return appropriate HTTP status codes
- Provide clear error messages
- Log detailed error information for debugging

## Continuous Integration

Our CI pipeline includes:

- Linting with flake8
- Type checking with mypy
- Running tests
- Building Docker images
- Deploying to staging/production

## Deployment

### Building Docker Images

```bash
docker build -t astroshield-api:latest .
```

### Environment Variables for Production

Set the following environment variables in production:

- `ENVIRONMENT=production`
- `DEBUG=False`
- `SECRET_KEY=<secure-random-string>`
- `DATABASE_URL=<production-db-url>`
- `ALLOWED_HOSTS=api.astroshield.com`

## Contributing

1. Create a feature branch from `develop`
2. Implement your changes
3. Add or update tests
4. Update documentation
5. Submit a pull request
6. Ensure CI checks pass

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Use docstrings for all public functions and classes
- Run `black` and `isort` before committing code

## Performance Considerations

- Use database indexes for frequently queried fields
- Implement caching for expensive operations
- Use async handlers for I/O-bound operations
- Optimize database queries (use select_related/prefetch_related)

## Security Best Practices

- Never commit secrets to the repository
- Validate all user input
- Use parameterized queries to prevent SQL injection
- Implement rate limiting
- Keep dependencies updated
- Perform regular security audits

## Troubleshooting

### Common Issues

#### API Server Won't Start

Check if:
- The database is running
- Environment variables are set correctly
- Required ports are available

#### Slow API Responses

- Check database query performance
- Look for N+1 query issues
- Enable query logging in development

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)

## Support

For questions or issues:
- Open an issue on GitHub
- Contact the development team at dev@astroshield.com 