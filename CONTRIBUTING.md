# Contributing to AstroShield

Thank you for your interest in contributing to AstroShield! Before contributing, please note that this is a proprietary software project owned by Stardrive Inc.

## Confidentiality and Ownership

By contributing to this project, you acknowledge and agree that:

1. All contributions become the exclusive property of Stardrive Inc.
2. You must have proper authorization to contribute
3. All work must be treated as confidential
4. You may be required to sign a Contributor License Agreement (CLA)

Please contact legal@stardrive.com before making any contributions.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a respectful and inclusive environment.

## Getting Started

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/your-username/astroshield.git
```
3. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

## Development Setup

1. Install dependencies:
```bash
# Python dependencies
pip install -e ".[dev]"  # Install with development extras

# Frontend
cd frontend
npm install
```

2. Set up environment:
```bash
cp .env.example .env
```

3. Start development servers:
```bash
# Backend
cd backend
uvicorn app.main:app --reload

# Frontend
cd frontend
npm run dev
```

## Repository Structure

The repository is organized into the following main directories:

- `src/`: Core AstroShield subsystem architecture and components
- `backend/`: FastAPI backend services
- `frontend/`: Next.js frontend application
- `ml/`: Machine learning models and training infrastructure
- `infrastructure/`: Infrastructure components
- `docs/`: Documentation
- `k8s/`: Kubernetes configuration

## Code Style

### Python (Backend)

- Follow PEP 8 guidelines
- Use type hints
- Format code with Black
- Sort imports with isort
- Lint with ruff

```bash
# Format code
black .
isort .

# Lint code
ruff check .
```

### TypeScript (Frontend)

- Follow ESLint configuration
- Use Prettier for formatting
- Follow React best practices

```bash
# Format code
npm run format

# Lint code
npm run lint
```

## Testing

### Backend Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest src/tests/test_specific.py
```

### Frontend Tests

```bash
# Run all tests
npm test

# Run with coverage
npm test -- --coverage

# Run specific test file
npm test -- src/components/__tests__/Component.test.tsx
```

## Pull Request Process

1. Update documentation
2. Add/update tests
3. Ensure CI passes
4. Request review
5. Address feedback

### PR Title Format

Use semantic commit messages:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `style:` Formatting
- `chore:` Maintenance

Example: `feat: Add spacecraft trajectory prediction`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Description of testing performed

## Screenshots
If applicable

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CI passes
- [ ] Code formatted
```

## Documentation

- Update README.md if needed
- Add inline code comments
- Update API documentation
- Update architecture docs if needed

## Continuous Integration

We use GitHub Actions for CI/CD. Every pull request triggers:

1. Unit tests
2. Integration tests
3. Linting checks
4. Type checking
5. Security scans

## Performance Considerations

- Consider the performance impact of your changes
- Profile code when making performance-critical changes
- Document performance characteristics when relevant

## Security Best Practices

- Never commit sensitive information (keys, passwords, etc.)
- Use proper authentication and authorization
- Validate all inputs
- Follow secure coding guidelines
- Report security issues confidentially

## Reporting Issues

### Bug Reports

Include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Screenshots if applicable
- Environment details

### Feature Requests

Include:
- Problem statement
- Proposed solution
- Alternative solutions
- Additional context

## Review Process

1. Code review by maintainers
2. Automated checks must pass
3. Documentation review
4. Final approval

## License and Legal

By contributing to this project, you agree that:

1. Your contributions become the exclusive property of Stardrive Inc.
2. You have the right to make the contribution
3. You will maintain the confidentiality of the codebase
4. Your contributions will be governed by our proprietary license

For any legal questions, please contact legal@stardrive.com