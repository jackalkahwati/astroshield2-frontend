# Publishing AstroShield API to SwaggerHub

This document provides instructions for publishing the AstroShield API documentation to SwaggerHub.

## Available Tools

1. **Python Script** (`publish_to_swaggerhub.py`) - Interactive Python script for API publishing
2. **Shell Script** (`upload_to_swaggerhub.sh`) - Simple shell script for API publishing
3. **Documentation** (`backend/docs/publishing_to_swagger.md`) - Detailed guide with manual steps

## Using the Python Script (Recommended)

The Python script provides the most user-friendly way to publish your API:

```bash
# Install required packages
pip install requests

# Run the script with your SwaggerHub username
./publish_to_swaggerhub.py --username YOUR_USERNAME
```

Additional options:
```
--api-key      Your SwaggerHub API key (default: provided key)
--file         Path to OpenAPI spec (default: openapi.json)
--name         API name (default: AstroShield)
--version      API version (default: 1.0.0)
```

Example with all options:
```bash
./publish_to_swaggerhub.py --username YOUR_USERNAME --api-key YOUR_KEY --file ./path/to/openapi.json --name CustomName --version 2.0.0
```

## Using the Shell Script

A simpler shell script is also available:

```bash
./upload_to_swaggerhub.sh
```

This script attempts multiple API endpoint formats and provides feedback on the results.

## Manual Publishing Instructions

If the automated methods don't work due to API permissions, follow the manual instructions in the documentation:

```bash
# View the manual instructions
cat backend/docs/publishing_to_swagger.md
```

The manual instructions guide you through:
1. Exporting the OpenAPI specification
2. Logging into SwaggerHub
3. Creating a new API through the web interface
4. Uploading your specification

## Troubleshooting

- **405 Method Not Allowed**: This typically means your API key doesn't have proper permissions. Use the web UI instead.
- **Failed to open file**: Ensure the path to your OpenAPI specification is correct.
- **Authentication errors**: Verify you're using the correct API key.

## Need Help?

If you encounter issues, contact the AstroShield development team for assistance or consult the [SwaggerHub documentation](https://support.smartbear.com/swaggerhub/docs/apis/index.html). 