#!/bin/bash

# API Key for SwaggerHub
API_KEY="d8533cd1-a315-408f-bf4f-fcd898863daf"

# Check if openapi.json exists
if [ ! -f "openapi.json" ]; then
    echo "Error: openapi.json not found in the current directory"
    exit 1
fi

echo "Attempting to upload to SwaggerHub..."

# Try multiple endpoints to address potential API changes
echo "Trying first endpoint format..."
response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://api.swaggerhub.com/apis/AstroShield/AstroShield/1.0.0" \
     -H "Authorization: $API_KEY" \
     -H "Content-Type: application/json" \
     -d @openapi.json)

if [ "$response" -eq 200 ] || [ "$response" -eq 201 ]; then
    echo "Success! API uploaded successfully."
    exit 0
fi

echo "First attempt returned: $response"
echo "Trying second endpoint format..."

response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://api.swaggerhub.com/specs" \
     -H "Authorization: $API_KEY" \
     -H "Content-Type: application/json" \
     -d "{\"name\":\"AstroShield\",\"version\":\"1.0.0\",\"specification\":$(cat openapi.json)}")

if [ "$response" -eq 200 ] || [ "$response" -eq 201 ]; then
    echo "Success! API uploaded successfully."
    exit 0
fi

echo "Second attempt returned: $response"
echo "Trying third endpoint format with PUT..."

response=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "https://api.swaggerhub.com/apis/AstroShield/AstroShield/1.0.0?isPrivate=false" \
     -H "Authorization: $API_KEY" \
     -H "Content-Type: application/json" \
     -d @openapi.json)

if [ "$response" -eq 200 ] || [ "$response" -eq 201 ]; then
    echo "Success! API uploaded successfully."
    exit 0
fi

echo "Third attempt returned: $response"
echo "All attempts failed. Please check your API key and SwaggerHub account."
echo "Trying to verify API key access..."

response=$(curl -s -X GET "https://api.swaggerhub.com/apis?limit=1" -H "Authorization: $API_KEY")
echo "API key check result: $response"

echo "Upload failed. Please check SwaggerHub documentation for the latest API endpoints." 