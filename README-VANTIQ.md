# AstroShield Vantiq Integration

This repository contains the integration between AstroShield and the Vantiq platform for real-time event processing, command and control, and data visualization.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/jackalkahwati/asttroshield_v0.git
cd asttroshield_v0
```

2. Set up the environment:
```bash
cp .env.vantiq .env
```

3. Edit the `.env` file with your Vantiq credentials.

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Start the application:
```bash
python backend/main.py
```

## Features

- **Real-time Trajectory Updates**: Send satellite trajectory data to Vantiq for processing and visualization.
- **Threat Detection**: Send threat detection events to Vantiq for alerting and response.
- **Command and Control**: Receive and execute commands from Vantiq.
- **Webhook Integration**: Receive real-time events from Vantiq.
- **Authentication**: Secure API communication with token-based authentication.

## Architecture

The integration uses a modular architecture:

- **API Layer**: FastAPI endpoints for receiving webhooks and commands from Vantiq.
- **Adapter Layer**: VantiqAdapter for communicating with the Vantiq API.
- **Configuration**: Environment-based configuration for flexible deployment.

## Documentation

For detailed documentation, see:

- [Vantiq Integration Guide](docs/vantiq-integration.md)
- [API Documentation](docs/api.md)
- [Configuration Guide](docs/configuration.md)

## Development

### Prerequisites

- Python 3.8+
- Vantiq account with API access
- FastAPI

### Testing

Run the tests:
```bash
pytest backend/tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact the AstroShield team at support@astroshield.com. 