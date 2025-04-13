#!/usr/bin/env python3
"""
Simple database setup for the AstroShield platform.
Creates a SQLite database with basic tables.
"""
import os
import sys
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("DATABASE_URL", "sqlite:///./astroshield.db")
if DB_PATH.startswith("sqlite:///"):
    DB_PATH = DB_PATH[len("sqlite:///"):]

def setup_database():
    """Create a new SQLite database with basic schema"""
    print(f"Setting up database at {DB_PATH}")
    
    # Check if database directory exists
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS satellites (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        status TEXT DEFAULT 'active',
        last_update TEXT,
        created_at TEXT,
        description TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS maneuvers (
        id TEXT PRIMARY KEY,
        satellite_id TEXT,
        status TEXT DEFAULT 'planned',
        type TEXT,
        start_time TEXT,
        end_time TEXT,
        created_at TEXT,
        description TEXT,
        FOREIGN KEY (satellite_id) REFERENCES satellites (id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        type TEXT,
        description TEXT,
        created_at TEXT,
        severity TEXT,
        satellite_id TEXT,
        FOREIGN KEY (satellite_id) REFERENCES satellites (id)
    )
    ''')
    
    # Add some sample data
    now = datetime.utcnow().isoformat()
    
    # Sample satellites
    satellites = [
        ("sat-001", "AstroShield Demo Sat 1", "active", now, now, "Demo satellite for testing"),
        ("sat-002", "AstroShield Demo Sat 2", "active", now, now, "Secondary demo satellite"),
        ("sat-003", "Test Satellite Alpha", "inactive", now, now, "Inactive test satellite")
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO satellites (id, name, status, last_update, created_at, description) VALUES (?, ?, ?, ?, ?, ?)",
        satellites
    )
    
    # Sample maneuvers
    maneuvers = [
        ("mnv-001", "sat-001", "completed", "collision_avoidance", 
         (datetime.utcnow().replace(hour=datetime.utcnow().hour-2)).isoformat(),
         (datetime.utcnow().replace(hour=datetime.utcnow().hour-1)).isoformat(),
         now, "Collision avoidance maneuver"),
        ("mnv-002", "sat-001", "planned", "station_keeping",
         (datetime.utcnow().replace(hour=datetime.utcnow().hour+5)).isoformat(),
         None, now, "Scheduled station keeping")
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO maneuvers (id, satellite_id, status, type, start_time, end_time, created_at, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        maneuvers
    )
    
    # Sample events
    events = [
        ("evt-001", "warning", "Potential conjunction detected", now, "warning", "sat-001"),
        ("evt-002", "info", "Telemetry update received", now, "info", "sat-001"),
        ("evt-003", "error", "Communication disruption", now, "critical", "sat-002")
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO events (id, type, description, created_at, severity, satellite_id) VALUES (?, ?, ?, ?, ?, ?)",
        events
    )
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database setup complete")
    return True

if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)
