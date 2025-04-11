# Publishing to SwaggerHub Using the CLI

This guide shows how to publish your AstroShield API to SwaggerHub using the official SwaggerHub CLI.

## Prerequisites

- Node.js and npm installed
- Your OpenAPI specification file (`openapi.json`)
- A SwaggerHub account

## Important Note About SwaggerHub Usernames

The SwaggerHub CLI requires using your actual SwaggerHub username, **not your email address**. Your username:

- Can be found in your SwaggerHub profile
- Is visible in URLs when you view your APIs: `https://app.swaggerhub.com/apis/YOUR_USERNAME`
- Usually has no special characters (like `@`)
- Examples: `johnsmith`, `acme-corp`, `astro-shield`

## Step 1: Install the SwaggerHub CLI

```bash
# Install the SwaggerHub CLI globally
npm install -g swaggerhub-cli
```

## Step 2: Configure the CLI

```bash
# Configure your SwaggerHub credentials
swaggerhub configure

# You'll be prompted to enter:
# - SwaggerHub URL (default: https://api.swaggerhub.com)
# - API Key (create this in your SwaggerHub account settings)
```

## Step 3: Create or Update an API

### To create a new API:

```bash
# Create a new API
swaggerhub api:create USERNAME/AstroShield/1.0.0 --file openapi.json --visibility public
```

Replace `USERNAME` with your SwaggerHub username (not email).

### To update an existing API:

```bash
# Update an existing API
swaggerhub api:update USERNAME/AstroShield/1.0.0 --file openapi.json
```

## Step 4: Verify Publication

After publishing, you can verify that your API was published successfully:

```bash
# View API details
swaggerhub api:get USERNAME/AstroShield/1.0.0
```

Your API will be available at:
```
https://app.swaggerhub.com/apis/USERNAME/AstroShield/1.0.0
```

## Additional CLI Commands

```bash
# List all your APIs
swaggerhub api:list

# Download an API
swaggerhub api:get USERNAME/AstroShield/1.0.0 --output downloaded-spec.json

# Delete an API
swaggerhub api:delete USERNAME/AstroShield/1.0.0
```

## Complete Example

Here's a complete example workflow:

```bash
# Install CLI
npm install -g swaggerhub-cli

# Configure with your API key from SwaggerHub
swaggerhub configure

# Publish your API (using username 'astro-team')
swaggerhub api:create astro-team/AstroShield/1.0.0 --file openapi.json --visibility public

# Check that it was published
swaggerhub api:get astro-team/AstroShield/1.0.0
```

## Troubleshooting

- **Username Format Error**: Ensure you're using your SwaggerHub username, not your email address
- **Authentication Errors**: Verify you've configured the CLI with the correct API key from your SwaggerHub account
- **Not Found Errors**: Check that you're using the correct username and API path
- **Validation Errors**: Verify that your OpenAPI specification is valid

## How to Find Your SwaggerHub Username

1. Log in to SwaggerHub at https://app.swaggerhub.com
2. Look at the URL after you log in, it may show your username in the format: https://app.swaggerhub.com/apis/USERNAME
3. Alternatively, check your profile settings or look at the path of any existing APIs you have

For more information, see the [SwaggerHub CLI documentation](https://github.com/SmartBear/swaggerhub-cli). 