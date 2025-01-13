from fastapi import FastAPI
from mangum import Mangum
from endpoints import router

app = FastAPI()

# Include the router
app.include_router(router)

# Create handler for AWS Lambda
handler = Mangum(app)

@app.get("/")
async def root():
    return {"message": "Welcome to AstroShield API"} 