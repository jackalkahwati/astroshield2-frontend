#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/stardrive/astroshield')
sys.path.append('/home/stardrive/astroshield/backend_fixed')

from sqlalchemy.orm import Session
from backend_fixed.app.db.session import SessionLocal, engine
from backend_fixed.app.models.user import User
from backend_fixed.app.core.security import get_password_hash
from backend_fixed.app.db.base_class import Base

def create_admin_user():
    """Create default admin user for AstroShield"""
    
    # Create tables
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Create session
    db: Session = SessionLocal()
    
    try:
        # Check if admin user already exists
        existing_user = db.query(User).filter(User.email == "admin@astroshield.com").first()
        
        if existing_user:
            print("✅ Admin user already exists!")
            print(f"   Email: {existing_user.email}")
            print(f"   Active: {existing_user.is_active}")
            print(f"   Superuser: {existing_user.is_superuser}")
            return existing_user
        
        # Create admin user
        admin_user = User(
            email="admin@astroshield.com",
            full_name="AstroShield Administrator",
            hashed_password=get_password_hash("admin123"),
            is_active=True,
            is_superuser=True,
            roles=["admin", "operator", "analyst"]
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        print("🎉 Admin user created successfully!")
        print(f"   Email: {admin_user.email}")
        print(f"   Password: admin123")
        print(f"   Roles: {admin_user.roles}")
        
        # Create additional demo users
        demo_users = [
            {
                "email": "operator@astroshield.com",
                "full_name": "System Operator",
                "password": "operator123",
                "roles": ["operator"]
            },
            {
                "email": "analyst@astroshield.com", 
                "full_name": "Data Analyst",
                "password": "analyst123",
                "roles": ["analyst"]
            }
        ]
        
        for user_data in demo_users:
            existing = db.query(User).filter(User.email == user_data["email"]).first()
            if not existing:
                demo_user = User(
                    email=user_data["email"],
                    full_name=user_data["full_name"],
                    hashed_password=get_password_hash(user_data["password"]),
                    is_active=True,
                    is_superuser=False,
                    roles=user_data["roles"]
                )
                db.add(demo_user)
        
        db.commit()
        print("✅ Demo users created!")
        
        return admin_user
        
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        db.rollback()
        return None
    finally:
        db.close()

def test_authentication():
    """Test the authentication system"""
    
    print("\n=== Testing Authentication ===")
    
    # Test imports
    try:
        from backend_fixed.app.core.security import authenticate_user, create_access_token
        print("✅ Security modules imported successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test database connection
    try:
        db: Session = SessionLocal()
        user_count = db.query(User).count()
        print(f"✅ Database connected - {user_count} users found")
        db.close()
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Initializing AstroShield Authentication System...")
    
    if test_authentication():
        admin_user = create_admin_user()
        if admin_user:
            print("\n🎯 Authentication Setup Complete!")
            print("\nDefault Credentials:")
            print("  Admin: admin@astroshield.com / admin123")
            print("  Operator: operator@astroshield.com / operator123") 
            print("  Analyst: analyst@astroshield.com / analyst123")
            print("\nTo test: POST to http://localhost:5002/api/v1/auth/token")
        else:
            print("❌ Failed to create admin user")
            sys.exit(1)
    else:
        print("❌ Authentication system test failed")
        sys.exit(1) 