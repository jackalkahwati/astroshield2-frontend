#!/usr/bin/env python3

import sqlite3
import bcrypt
import hashlib

def hash_password(password: str) -> str:
    """Create password hash using bcrypt"""
    # Simple bcrypt-style hash for demo
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def init_database():
    """Initialize database and create demo users"""
    
    conn = sqlite3.connect('/home/stardrive/astroshield/astroshield.db')
    cursor = conn.cursor()
    
    # Create users table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT,
            hashed_password TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            is_superuser BOOLEAN DEFAULT 0,
            roles TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
    ''')
    
    # Demo users
    demo_users = [
        ('admin@astroshield.com', 'AstroShield Administrator', 'admin123', True, True, '["admin", "operator", "analyst"]'),
        ('operator@astroshield.com', 'System Operator', 'operator123', True, False, '["operator"]'),
        ('analyst@astroshield.com', 'Data Analyst', 'analyst123', True, False, '["analyst"]'),
        ('demo@astroshield.com', 'Demo User', 'demo123', True, False, '["user"]')
    ]
    
    for email, full_name, password, is_active, is_superuser, roles in demo_users:
        # Check if user exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            print(f'✅ User {email} already exists')
            continue
            
        # Create user
        hashed_password = hash_password(password)
        cursor.execute('''
            INSERT INTO users (email, full_name, hashed_password, is_active, is_superuser, roles)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (email, full_name, hashed_password, is_active, is_superuser, roles))
        print(f'🎉 Created user: {email} / {password}')
    
    conn.commit()
    conn.close()
    
    print('\n🎯 Demo users ready!')
    print('Use these credentials to login:')
    print('  Admin: admin@astroshield.com / admin123')
    print('  Operator: operator@astroshield.com / operator123')  
    print('  Analyst: analyst@astroshield.com / analyst123')
    print('  Demo: demo@astroshield.com / demo123')

if __name__ == '__main__':
    init_database() 