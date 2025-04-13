from datetime import datetime, timedelta
from typing import Any, Union, Optional, List, Dict, Tuple
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from app.models.user import User, UserBase
import os
import logging

logger = logging.getLogger(__name__)

# Security configuration - in production, these should be in environment variables
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-for-development-only")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: datetime

class TokenData(BaseModel):
    email: Optional[str] = None
    is_superuser: bool = False

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Get password hash"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT access token"""
    to_encode = data.copy()
    
    # Default expiration to 30 minutes if not specified
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[User]:
    """
    Get the current user from the token.
    Returns None if no token or invalid token provided.
    """
    if not token:
        return None
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
            
        token_data = TokenData(email=email, is_superuser=payload.get("is_superuser", False))
    except JWTError as e:
        logger.warning(f"JWT token validation error: {str(e)}")
        return None
        
    # Mock user for development - replace with database lookup in production
    user = User(
        id=1,
        email=token_data.email,
        is_active=True,
        is_superuser=token_data.is_superuser,
        full_name="Test User",
        created_at=datetime.utcnow() - timedelta(days=30)
    )
    return user

def check_roles(required_roles: Optional[List[str]] = None):
    """
    Dependency to check if the current user has the required roles.
    If no roles are required, just checks that the user is authenticated.
    """
    async def role_checker(user: Optional[User] = Depends(get_current_user)) -> User:
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user"
            )
            
        if required_roles is None:
            return user
            
        if "admin" in required_roles and not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
            
        # In a real implementation, you would check for specific roles here
        # For now, we just check for "active" as a sample role
        if "active" in required_roles and not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Active role required"
            )
            
        return user
    return role_checker

# Auth handler functions
async def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user by username/email and password.
    In a real implementation, you would look up the user in the database.
    """
    # Mock user for development - replace with database lookup in production
    if username == "test@example.com" and password == "password":
        return User(
            id=1,
            email=username,
            is_active=True,
            is_superuser=False,
            full_name="Test User",
            created_at=datetime.utcnow() - timedelta(days=30)
        )
    elif username == "admin@example.com" and password == "admin":
        return User(
            id=2,
            email=username,
            is_active=True,
            is_superuser=True,
            full_name="Admin User",
            created_at=datetime.utcnow() - timedelta(days=30)
        )
        
    return None