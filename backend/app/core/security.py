from datetime import datetime, timedelta
from typing import Any, Union, Optional, List, Dict, Tuple
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from app.models.user import User, UserBase
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import json
import os

# Security configuration
SECRET_KEY = "your-secret-key"  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    is_superuser: bool = False

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    # Mock user for development - replace with database lookup in production
    user = User(
        id=1,
        email=email,
        is_active=True,
        is_superuser=True,
        full_name="Test User"
    )
    return user

def check_roles(required_roles: Optional[List[str]] = None):
    def role_checker(user: User = Depends(get_current_user)) -> bool:
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user"
            )
            
        if required_roles is None:
            return True
            
        if "admin" in required_roles and not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
            
        return True
    return role_checker

# Key management classes
class KeyVersion(BaseModel):
    key_id: str
    key: str
    created_at: datetime
    expires_at: datetime
    is_active: bool

class KeyRotation:
    def __init__(self):
        self.keys: List[KeyVersion] = []
        self.rotation_interval = timedelta(days=30)
        self._initialize_keys()

    def _initialize_keys(self):
        current_time = datetime.utcnow()
        self.add_key(current_time)

    def add_key(self, current_time: datetime):
        key_id = f"key_{len(self.keys) + 1}"
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        key = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        
        new_key = KeyVersion(
            key_id=key_id,
            key=key,
            created_at=current_time,
            expires_at=current_time + self.rotation_interval,
            is_active=True
        )
        self.keys.append(new_key)
        return new_key

    def get_active_key(self) -> KeyVersion:
        current_time = datetime.utcnow()
        active_key = next(
            (key for key in self.keys if key.is_active and key.expires_at > current_time),
            None
        )
        if not active_key:
            active_key = self.add_key(current_time)
        return active_key

    def rotate_keys(self):
        current_time = datetime.utcnow()
        for key in self.keys:
            if key.expires_at <= current_time:
                key.is_active = False
        self.add_key(current_time)

key_rotation = KeyRotation()

class KeyStore:
    def __init__(self):
        self.current_version = 1
        self._keys = {}
        self.rotate_keys()

    def rotate_keys(self):
        self.current_version += 1
        self._keys[self.current_version] = jwt.encode(
            {"version": self.current_version, "created": datetime.utcnow().isoformat()},
            SECRET_KEY,
            algorithm=ALGORITHM
        )

    def get_current_key(self):
        return self._keys.get(self.current_version)

# Global key store instance
key_store = KeyStore() 