# Manual Upload Instructions for SwaggerHub

It seems direct API publishing is giving us HTTP 405 errors. Let's use the web interface instead, which is more reliable.

## Step 1: Verify Your OpenAPI Specification

Your `openapi.json` file is already prepared and ready for upload. It's located in the root directory of your project.

## Step 2: Log in to SwaggerHub

1. Open your browser and go to [app.swaggerhub.com](https://app.swaggerhub.com/)
2. Log in with your credentials (username: stardrive)

## Step 3: Create a New API

1. From your dashboard, click the "+ Create New" button in the top-right corner
2. Select "Create New API"
3. Choose "Import and Document API"

## Step 4: Upload Your OpenAPI Specification

1. Select "Import from File"
2. Click "Browse" and navigate to your project directory
3. Select the `openapi.json` file from the root directory
4. Click "Open"

## Step 5: Configure API Settings

1. Complete the API details:
   - **Name**: AstroShield
   - **Version**: 1.0.0
   - **Visibility**: Public
   - **Auto Mock API**: Optional (enable if you want SwaggerHub to create a mock server)
   
2. Click "Create API"

## Step 6: Verify Your API Documentation

After creation:
1. Review your API documentation to ensure all endpoints, models, and schemas are correctly displayed
2. Test the interactive documentation by clicking on endpoints and trying them out
3. Your API will be available at: `https://app.swaggerhub.com/apis/stardrive/AstroShield/1.0.0`

## Step 7: Share Your API

Now that your API is published, you can:
1. Share the link with others
2. Export the documentation in various formats
3. Generate client code in different languages

## Troubleshooting

If you encounter any issues during manual upload:
- Ensure your OpenAPI specification is valid
- Check that your file is properly formatted JSON
- Try uploading a smaller portion of the specification if the size is an issue 