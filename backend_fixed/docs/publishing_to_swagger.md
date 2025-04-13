# Publishing AstroShield API to SwaggerHub

This guide explains how to publish the AstroShield API documentation to SwaggerHub.

## Prerequisites

- A SwaggerHub account (create one at [SwaggerHub](https://app.swaggerhub.com/signup))
- The AstroShield API key: `d8533cd1-a315-408f-bf4f-fcd898863daf` (for API access, not for SwaggerHub publishing)

## Important Note About the API Key

The API key provided (`d8533cd1-a315-408f-bf4f-fcd898863daf`) is meant for **accessing the AstroShield API** as a client, not for publishing to SwaggerHub. This key should be used:

- When making authenticated requests to the AstroShield API 
- In the `X-API-Key` header of your API requests
- For client applications integrating with AstroShield

For publishing to SwaggerHub, you'll need to use the manual web interface method described below.

## Steps to Publish to SwaggerHub

### 1. Export the OpenAPI Specification

First, ensure your local server is running to export the OpenAPI specification:

```bash
# Start the backend server if not already running
cd backend
python main.py
```

Then download the OpenAPI specification from your local server:

```bash
curl http://localhost:3001/api/v1/openapi.json -o openapi.json
```

If your server is not running, you can use the existing OpenAPI specification file in the root directory.

### 2. Log into SwaggerHub

1. Go to [app.swaggerhub.com](https://app.swaggerhub.com/login)
2. Log in with your credentials

### 3. Create a New API

1. Click the "Create New" button
2. Select "Create New API"
3. Choose "Import and Document API"
4. Select the `openapi.json` file you downloaded
5. Fill in the required fields:
   - API Name: AstroShield
   - Version: 1.0.0
   - Visibility: Public (or Private based on your preference)
6. Click "Create API"

## Viewing Your Published API

Once published, you can access your API documentation at:

```
https://app.swaggerhub.com/apis/YOUR_USERNAME/AstroShield/1.0.0
```

Replace `YOUR_USERNAME` with your SwaggerHub username.

## Testing the API with Your API Key

After publishing the documentation, you can test API endpoints directly from SwaggerHub:

1. Navigate to your published API documentation
2. Click on an endpoint that requires API key authentication
3. Click the "Try it out" button
4. In the authentication section, enter the API key: `d8533cd1-a315-408f-bf4f-fcd898863daf`
5. Fill in any required parameters
6. Click "Execute" to test the endpoint

## Troubleshooting

- **File Format Issues**: Ensure your OpenAPI specification is valid JSON
- **Import Errors**: Check the specification for validation issues
- **Authentication Problems**: Make sure you're logged into SwaggerHub with the correct account

## Updating the API

To update an existing API:

1. Make your changes to the OpenAPI specification
2. Follow the same steps as above, using the same API name and version
3. SwaggerHub will save the new version and maintain revision history

## Help and Support

If you encounter issues with SwaggerHub, consult their [documentation](https://support.smartbear.com/swaggerhub/docs/apis/index.html) or contact their support. 