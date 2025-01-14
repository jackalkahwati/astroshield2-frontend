from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to AstroShield API v1.0"}

# Create handler for AWS Lambda
handler = Mangum(app)

# Only import and include router after basic setup works
# from endpoints import router
# app.include_router(router, prefix="/api") 