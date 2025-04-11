# AstroShield Platform

AstroShield is a comprehensive solution for Space Domain Awareness (SDA) that integrates sensors, analytics, and threat assessment to protect space assets. This platform follows a microservices architecture organized around core subsystems that handle different aspects of the SDA mission.

## Repository Structure

The repository is organized into the following main components:

- **src/**: Core AstroShield subsystem architecture and components
  - **asttroshield/**: Core library with reusable components
  - **api_client/**: API client libraries for external services
  - **kafka_client/**: Kafka producer and consumer implementations
  - **models/**: Data models and ML model interfaces
  - **tests/**: Unit and integration tests

- **backend/**: FastAPI backend services
  - **app/**: Main application code
  - **routers/**: API route definitions
  - **services/**: Business logic implementations
  - **models/**: Database models
  - **middleware/**: Auth and request middleware

- **frontend/**: Next.js frontend application
  - **app/**: Next.js pages and routes
  - **components/**: React components
  - **lib/**: Utility functions and API clients

- **ml/**: Machine learning models and training infrastructure
  - **models/**: ML model implementations
  - **training/**: Training scripts and utilities
  - **data_generation/**: Synthetic data generation

- **infrastructure/**: Infrastructure components
  - **circuit_breaker.py**: Circuit breaker pattern implementation
  - **rate_limiter.py**: Rate limiting functionality
  - **cache.py**: Caching implementations
  - **monitoring.py**: Monitoring and observability utilities

- **docs/**: Documentation for various subsystems
- **k8s/**: Kubernetes deployment configurations
- **config/**: Configuration files and templates

## Repository Management

This repository uses Git LFS (Large File Storage) to efficiently manage large files such as models and binary data. For details on working with Git LFS, see [GIT_LFS_GUIDE.md](GIT_LFS_GUIDE.md).

### Git LFS

Large files in this repository (models, checkpoints, etc.) are managed using Git LFS. When you clone this repository, you'll need to:

1. Install Git LFS: https://git-lfs.github.com/
2. Run `git lfs install` to set up Git LFS
3. Run `git lfs pull` to download the large files

## Core Architecture

AstroShield implements a subsystem-based architecture as defined in the SDA specifications. The system consists of the following subsystems:

### Subsystems

1. **Subsystem 0 (SS0) - Data Ingestion**: Handles raw data ingestion from sensors and external sources
2. **Subsystem 1 (SS1) - Target Modeling**: Processes ingested data to create and update target models
3. **Subsystem 2 (SS2) - State Estimation**: Tracks objects and estimates their current and future states
4. **Subsystem 3 (SS3) - Command & Control (C2)**: Manages mission control and decision support
5. **Subsystem 4 (SS4) - CCDM Detection**: Detects Camouflage, Concealment, Deception, and Maneuvering
6. **Subsystem 5 (SS5) - Hostility Monitoring**: Monitors for potentially hostile actions
7. **Subsystem 6 (SS6) - Threat Assessment**: Analyzes detected activities and provides threat assessments

## Message-Driven Architecture

The AstroShield platform is built on a message-driven architecture using Kafka as the messaging backbone. All subsystems communicate through standardized message formats that include:

- **Headers**: Metadata about the message, including trace information
- **Payloads**: The actual content of the message (schema-validated)

### Message Traceability

A key feature of the AstroShield platform is message traceability. Every message includes:

- `messageId`: A unique identifier for each message
- `traceId`: An identifier that remains consistent through an entire message chain
- `parentMessageIds`: References to messages that triggered the current message

This traceability provides complete visibility into message flow throughout the system, making it easier to debug, audit, and validate system behavior.

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+
- Kafka cluster (we use Confluent Cloud in production)
- PostgreSQL 14+ database
- Redis for caching (optional but recommended)

### Setting Up the Development Environment

1. Clone this repository
   ```bash
   git clone https://github.com/your-org/astroshield.git
   cd astroshield
   ```

2. Install Python dependencies
   ```bash
   pip install -e ".[dev]"  # Install with development dependencies
   ```

3. Install frontend dependencies
   ```bash
   cd frontend
   npm install
   ```

4. Configure environment variables
   - Copy `.env.example` to `.env` and fill in your configuration
   - Set database connection details and Kafka credentials

### Running the Backend

1. Start the FastAPI backend
   ```bash
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 3001 --reload
   ```

2. The API will be available at http://localhost:3001
   - API Documentation: http://localhost:3001/api/v1/documentation
   - Swagger UI: http://localhost:3001/api/v1/docs
   - ReDoc: http://localhost:3001/api/v1/redoc
   - OpenAPI Specification: http://localhost:3001/api/v1/openapi.json

For details on how to use the API documentation, see [Swagger Documentation Guide](backend/docs/swagger_guide.md).

### Running the Frontend

1. Start the Next.js development server
   ```bash
   cd frontend
   npm run dev
   ```

2. The frontend will be available at http://localhost:3000

### Running Tests

1. Run Python tests
   ```bash
   pytest
   ```

2. Run JavaScript tests
   ```bash
   cd frontend
   npm test
   ```

## Integration Package

For partners looking to integrate with AstroShield, refer to the `astroshield-integration-package` directory, which contains:

- Message schemas for all subsystems
- Example code for consuming and producing messages
- Documentation on integration patterns

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Architecture Overview](docs/architecture/README.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment.md)
- [Development Guidelines](docs/development.md)
- [Kafka Topics and Message Formats](docs/kafka_topics.md)
- [ML Model Specifications](docs/model_specifications.md)
- [Security Considerations](docs/security.md)

## License

This software is proprietary and confidential. Refer to the LICENSE file for details.

## Contact

For questions or support, contact the AstroShield development team at support@example.com.