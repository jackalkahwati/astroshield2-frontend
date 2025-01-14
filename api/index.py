from fastapi import FastAPI, HTTPException, Header
from mangum import Mangum

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to AstroShield API"}

@app.get("/cron")
def cron_job(authorization: str = Header(None)):
    expected_auth = "Bearer 1234"
    if not authorization or authorization != expected_auth:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"message": "Hello Cron!"}

handler = Mangum(app) 