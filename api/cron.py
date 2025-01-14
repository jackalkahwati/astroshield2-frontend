from fastapi import FastAPI, Header, HTTPException
from typing import Optional

app = FastAPI()

@app.get("/cron")
async def cron_job(authorization: Optional[str] = Header(None)):
    if not authorization or authorization != f"Bearer {process.env.CRON_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"message": "Hello Cron!"} 