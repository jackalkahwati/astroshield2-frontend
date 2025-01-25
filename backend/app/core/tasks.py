from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
from app.core.security import key_store

def setup_periodic_tasks(app: FastAPI):
    scheduler = AsyncIOScheduler()
    
    # Schedule key rotation every 24 hours
    @scheduler.scheduled_job("interval", hours=24)
    def rotate_keys_job():
        print(f"[{datetime.utcnow()}] Rotating encryption keys...")
        key_store.rotate_keys()
        print(f"[{datetime.utcnow()}] Key rotation complete")
    
    # Schedule key cleanup every week
    @scheduler.scheduled_job("interval", days=7)
    def cleanup_old_keys_job():
        print(f"[{datetime.utcnow()}] Cleaning up old encryption keys...")
        key_store.cleanup_old_keys(max_age_days=30)
        print(f"[{datetime.utcnow()}] Key cleanup complete")
    
    scheduler.start()
    app.state.scheduler = scheduler 