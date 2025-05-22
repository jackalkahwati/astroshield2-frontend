from datetime import datetime, timedelta
from typing import Any, Union, Optional, List, Dict, Tuple
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from app.models.user import User
from app.db.session import get_db
from sqlalchemy.orm import Session
import os
import logging
from app.core.roles import Roles

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

async def get_current_user(db: Session = Depends(get_db), token: Optional[str] = Depends(oauth2_scheme)) -> Optional[User]:
    """
    Get the current user from the token by fetching from DB.
    Returns None if no token or invalid token or user not found.
    """
    if not token:
        return None
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub") # Assuming 'sub' in JWT is the user's email
        if email is None:
            raise credentials_exception
        # TokenData might still be useful for validating payload structure if complex
        # token_data = TokenData(email=email, is_superuser=payload.get("is_superuser", False))
    except JWTError as e:
        logger.warning(f"JWT token validation error: {str(e)}")
        raise credentials_exception # Raise exception, don't return None directly here for required auth
        
    user = db.query(User).filter(User.email == email).first()
    
    if user is None:
        logger.warning(f"User from token not found in DB: {email}")
        # Depending on policy, you might raise credentials_exception here too,
        # or allow check_roles to handle it if user is None.
        # For required authentication, failing to find the user should be an auth error.
        raise credentials_exception
        
    # The user object fetched from DB should have is_active, is_superuser, and roles populated.
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
            
        if required_roles:
            # Ensure user.roles exists; this would come from your actual user model
            user_roles = getattr(user, 'roles', []) 
            if not isinstance(user_roles, list):
                logger.error(f"User {user.email} 'roles' attribute is not a list.")
                user_roles = [] # Default to empty list to prevent further errors

            for role in required_roles:
                role_value = role.value if isinstance(role, Roles) else role
                if role_value == "admin":
                    if not user.is_superuser:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Admin privileges required. User does not have 'admin' (superuser) status."
                        )
                elif role_value == "viewer":
                    # viewer role just needs to be authenticated & active which we've already confirmed
                    pass
                elif role_value == "active": # backward compatibility; treat as viewer
                    pass # Already checked by user.is_active
                elif role_value not in user_roles:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Insufficient permissions. Missing role: {role_value}"
                    )
        return user
    return role_checker

# Auth handler functions
async def authenticate_user(email: str, password: str, db: Session = Depends(get_db)) -> Optional[User]:
    """
    Authenticate a user by email and password.
    Fetches user from the database.
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        logger.warning(f"Authentication attempt for non-existent user: {email}")
        return None
    if not user.is_active: # Optionally check if user is active before verifying password
        logger.warning(f"Authentication attempt for inactive user: {email}")
        return None
    if not pwd_context.verify(password, user.hashed_password): # Use pwd_context defined above
        logger.warning(f"Invalid password for user: {email}")
        return None
    logger.info(f"User successfully authenticated: {email}")
    return user