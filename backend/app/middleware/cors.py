from fastapi import Request, Response
from typing import Optional

async def dynamic_cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin")
    
    # Default to a forbidden response
    response = Response("Not allowed", status_code=403)
    
    # Check if origin is allowed (vercel.app or localhost)
    if origin and (
        ".vercel.app" in origin or 
        "localhost" in origin or 
        "railway.app" in origin or  # Allow Railway domains
        "astroshield.com" in origin  # Allow production domain
    ):
        # Call the next middleware/route handler
        response = await call_next(request)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                }
            )
    else:
        # For non-allowed origins, try the next middleware
        response = await call_next(request)
    
    return response 