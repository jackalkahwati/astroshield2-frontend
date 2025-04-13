#!/usr/bin/env python3
"""
Database setup for the AstroShield platform.
Supports both SQLite (development) and PostgreSQL (production) databases.
"""
import os
import sys
import sqlite3
import psycopg2
from datetime import datetime
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get database URL from environment
DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./astroshield.db")

def is_postgres_url(url):
    """Check if the database URL is for PostgreSQL"""
    return url.startswith('postgresql://') or url.startswith('postgres://')

def setup_sqlite_database(db_path):
    """Set up SQLite database with basic schema"""
    logger.info(f"Setting up SQLite database at {db_path}")
    
    # Check if database directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
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
    
    # Add sample data for development
    insert_sample_data(conn, cursor, "sqlite")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    logger.info("SQLite database setup complete")
    return True

def setup_postgres_database(db_url):
    """Set up PostgreSQL database with basic schema"""
    logger.info("Setting up PostgreSQL database")
    
    try:
        # Parse the database URL
        result = urlparse(db_url)
        username = result.username
        password = result.password
        database = result.path[1:]
        hostname = result.hostname
        port = result.port or 5432
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            dbname=database,
            user=username,
            password=password,
            host=hostname,
            port=port
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS satellites (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            last_update TIMESTAMP,
            created_at TIMESTAMP,
            description TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS maneuvers (
            id TEXT PRIMARY KEY,
            satellite_id TEXT,
            status TEXT DEFAULT 'planned',
            type TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            created_at TIMESTAMP,
            description TEXT,
            FOREIGN KEY (satellite_id) REFERENCES satellites (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            type TEXT,
            description TEXT,
            created_at TIMESTAMP,
            severity TEXT,
            satellite_id TEXT,
            FOREIGN KEY (satellite_id) REFERENCES satellites (id)
        )
        ''')
        
        # Only insert sample data if this is not a production environment
        if os.environ.get("ENVIRONMENT", "development").lower() != "production":
            insert_sample_data(conn, cursor, "postgres")
        
        # Close connection
        cursor.close()
        conn.close()
        
        logger.info("PostgreSQL database setup complete")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up PostgreSQL database: {e}")
        return False

def insert_sample_data(conn, cursor, db_type):
    """Insert sample data into the database for development/testing"""
    now = datetime.utcnow().isoformat()
    
    # Sample satellites
    satellites = [
        ("sat-001", "AstroShield Demo Sat 1", "active", now, now, "Demo satellite for testing"),
        ("sat-002", "AstroShield Demo Sat 2", "active", now, now, "Secondary demo satellite"),
        ("sat-003", "Test Satellite Alpha", "inactive", now, now, "Inactive test satellite")
    ]
    
    if db_type == "sqlite":
        cursor.executemany(
            "INSERT OR REPLACE INTO satellites (id, name, status, last_update, created_at, description) VALUES (?, ?, ?, ?, ?, ?)",
            satellites
        )
    else:
        # PostgreSQL uses different parameter style
        for sat in satellites:
            cursor.execute(
                "INSERT INTO satellites (id, name, status, last_update, created_at, description) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
                sat
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
    
    if db_type == "sqlite":
        cursor.executemany(
            "INSERT OR REPLACE INTO maneuvers (id, satellite_id, status, type, start_time, end_time, created_at, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            maneuvers
        )
    else:
        for mnv in maneuvers:
            cursor.execute(
                "INSERT INTO maneuvers (id, satellite_id, status, type, start_time, end_time, created_at, description) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status",
                mnv
            )
    
    # Sample events
    events = [
        ("evt-001", "warning", "Potential conjunction detected", now, "warning", "sat-001"),
        ("evt-002", "info", "Telemetry update received", now, "info", "sat-001"),
        ("evt-003", "error", "Communication disruption", now, "critical", "sat-002")
    ]
    
    if db_type == "sqlite":
        cursor.executemany(
            "INSERT OR REPLACE INTO events (id, type, description, created_at, severity, satellite_id) VALUES (?, ?, ?, ?, ?, ?)",
            events
        )
    else:
        for evt in events:
            cursor.execute(
                "INSERT INTO events (id, type, description, created_at, severity, satellite_id) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET type = EXCLUDED.type",
                evt
            )

def setup_database():
    """Set up the database based on the configured URL"""
    try:
        if is_postgres_url(DB_URL):
            return setup_postgres_database(DB_URL)
        else:
            # For SQLite, extract the path
            if DB_URL.startswith("sqlite:///"):
                db_path = DB_URL[len("sqlite:///"):]
            else:
                db_path = DB_URL
            return setup_sqlite_database(db_path)
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False

if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)
