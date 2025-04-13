from fastapi import Request, Response
from typing import Optional

async def dynamic_cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin")
    
    # Call the next middleware/route handler
    response = await call_next(request)
    
    # Add CORS headers for all requests - use "*" for development
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    
    # Handle preflight requests
    if request.method == "OPTIONS":
        response = Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    return response 