# Manual Publishing to SwaggerHub

After multiple attempts to use the API key for automated publishing, it appears the API key provided (`d8533cd1-a315-408f-bf4f-fcd898863daf`) is likely meant for accessing the API rather than for publishing to SwaggerHub. The most reliable approach is to manually publish using the SwaggerHub web interface.

## Steps to Manually Publish to SwaggerHub

### 1. Ensure Your OpenAPI Specification is Ready

The OpenAPI specification (`openapi.json`) is already prepared and contains a complete definition of the AstroShield API, including:

- Authentication methods (Bearer Token and API Key)
- Endpoints for satellites, maneuvers, and authentication
- Detailed schema definitions
- Example requests and responses

### 2. Access SwaggerHub

1. Go to [SwaggerHub](https://app.swaggerhub.com/)
2. Sign in with your SwaggerHub account credentials
3. If you don't have an account, create one by clicking "Sign Up"

### 3. Create a New API

1. From your dashboard, click the "+ Create New" button
2. Select "Create New API"
3. Choose "Import and Document API"

### 4. Upload Your OpenAPI Specification

1. Select "Import from File" or "Import from URL"
2. For "Import from File":
   - Click "Browse" and navigate to the `openapi.json` file
   - Select the file and click "Open"
3. For "Import from URL" (if your API is hosted):
   - Enter the URL to your OpenAPI specification
   - For example: `http://localhost:3001/api/v1/openapi.json` (if your local server is running)

### 5. Configure API Settings

1. Fill in the required fields:
   - **API Name**: `AstroShield`
   - **Version**: `1.0.0`
   - **Visibility**: Choose "Public" or "Private" based on your requirements
   
2. Click "Create API"

### 6. Review and Enhance Your Documentation

Once imported, you can:

1. Review the documentation to ensure all endpoints and models are correctly displayed
2. Add additional descriptions or examples where needed
3. Test the API directly from the SwaggerHub interface
4. Share the documentation with others

### 7. Access Your Published API

Your published API will be available at:

```
https://app.swaggerhub.com/apis/YOUR_USERNAME/AstroShield/1.0.0
```

Replace `YOUR_USERNAME` with your SwaggerHub username.

## Using the API Key

The provided API key (`d8533cd1-a315-408f-bf4f-fcd898863daf`) can be used for:

1. **Accessing the API**: Use this key in the `X-API-Key` header when making requests to the AstroShield API
2. **Authentication**: Some endpoints in the AstroShield API support API Key authentication as shown in the OpenAPI specification
3. **Integration**: Include this key in client applications that need to access the AstroShield API

## Troubleshooting Manual Upload

- **File Format Issues**: Ensure your OpenAPI specification is valid JSON or YAML
- **Import Errors**: If you encounter errors during import, check the OpenAPI specification for validation issues
- **Authentication Problems**: Make sure you're logged into SwaggerHub with the correct account
- **Version Conflicts**: If the API version already exists, you might need to update the version number or use the "Update" option

For additional help, refer to the [SwaggerHub Documentation](https://support.smartbear.com/swaggerhub/docs/apis/index.html). 