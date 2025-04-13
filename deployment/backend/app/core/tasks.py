from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
from app.core.security import key_store

def setup_periodic_tasks(app: FastAPI):
    """Setup periodic tasks for the application"""
    scheduler = AsyncIOScheduler()
    
    # Add periodic tasks
    scheduler.add_job(rotate_keys, 'interval', hours=24, args=[app])
    
    # Start the scheduler
    scheduler.start()
    
    # Store the scheduler instance
    app.state.scheduler = scheduler

async def rotate_keys(app: FastAPI):
    """Rotate encryption keys periodically"""
    key_store.rotate_keys()
    print(f"Rotated encryption keys at {datetime.utcnow().isoformat()}")

# Schedule key cleanup every week
@scheduler.scheduled_job("interval", days=7)
def cleanup_old_keys_job():
    print(f"[{datetime.utcnow()}] Cleaning up old encryption keys...")
    key_store.cleanup_old_keys(max_age_days=30)
    print(f"[{datetime.utcnow()}] Key cleanup complete") 