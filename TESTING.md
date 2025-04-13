# AstroShield Testing Strategy

This document outlines the testing strategy for the AstroShield project.

## Test Types

### 1. Unit Tests

Unit tests are used to test individual components in isolation. These tests are located in the `backend_fixed/tests/unit` directory and focus on testing specific functions and classes.

Example tests:
- `test_ccdm_service.py`: Tests the CCDM service methods
- `test_udl_client.py`: Tests the UDL client

### 2. Integration Tests

Integration tests verify that different components work together correctly. These tests are located in the `backend_fixed/tests/integration` directory.

Example tests:
- `test_ccdm_api.py`: Tests the CCDM API endpoints
- `test_satellites_api.py`: Tests the Satellites API endpoints

### 3. Frontend Component Tests

Frontend component tests verify that React components render and behave correctly. These tests are located in the `frontend/src/components/__tests__` directory.

Example tests:
- `StatusPanel.test.js`: Tests the StatusPanel component
- `ThreatAssessment.test.js`: Tests the ThreatAssessment component

## Test Fixtures

We use pytest fixtures to set up common test data:

- `conftest.py`: Contains common fixtures used across tests
- `test_db_engine`: Creates an in-memory SQLite database for testing
- `client`: Creates a test client for the FastAPI application
- `sample_data`: Creates sample test data in the database

## Running Tests

### Backend Tests

```bash
# Run all backend tests
./run_tests.sh --backend

# Run only unit tests
./run_tests.sh --backend --unit

# Run only integration tests
./run_tests.sh --backend --integration

# Run a simple test for the CCDM service
python3 test_simple_ccdm.py
```

### Frontend Tests

```bash
# Run all frontend tests
./run_tests.sh --frontend

# Run tests with coverage
cd frontend && npm run test:coverage
```

## Continuous Integration

We recommend setting up a CI/CD pipeline to run tests automatically on each pull request and push to the main branch. This can be done using GitHub Actions or another CI/CD tool.

Example GitHub Actions workflow:

```yaml
name: Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run backend tests
        run: ./run_tests.sh --backend
      - name: Set up Node.js
        uses: actions/setup-node@v2
        with:
          node-version: '16'
      - name: Install frontend dependencies
        run: cd frontend && npm install
      - name: Run frontend tests
        run: cd frontend && npm test
```

## Coverage Thresholds

We've set up coverage thresholds to ensure adequate test coverage:

- Backend: 70% for statements, branches, and functions
- Frontend: 70% for statements, branches, and functions (currently set to 10% during initial setup)

## Best Practices

1. **Use AAA pattern**: Arrange, Act, Assert
2. **Mock external dependencies**: Use `unittest.mock` for Python and Jest mocks for JavaScript
3. **Test edge cases**: Include tests for error conditions and edge cases
4. **Keep tests independent**: Each test should be able to run independently of other tests
5. **Test public interfaces**: Focus on testing the public interfaces of modules
6. **Keep tests fast**: Tests should run quickly to encourage frequent testing 