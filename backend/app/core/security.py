from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from app.core.config import settings
from app.models.user import User
from app.db.session import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    roles: List[str] = []

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
        new_key = KeyVersion(
            key_id=key_id,
            key=settings.generate_secret_key(),
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

class RoleChecker:
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Security(oauth2_scheme)):
        if not any(role in self.allowed_roles for role in user.roles):
            raise HTTPException(
                status_code=403,
                detail="You don't have enough permissions"
            )
        return user

key_rotation = KeyRotation()

def create_access_token(data: dict):
    key_version = key_rotation.get_active_key()
    to_encode = data.copy()
    to_encode.update({
        "exp": datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        "kid": key_version.key_id
    })
    return jwt.encode(to_encode, key_version.key, algorithm=settings.ALGORITHM)

def verify_token(token: str) -> TokenData:
    try:
        unverified_headers = jwt.get_unverified_headers(token)
        key_id = unverified_headers.get("kid")
        if not key_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        key_version = next((k for k in key_rotation.keys if k.key_id == key_id), None)
        if not key_version:
            raise HTTPException(status_code=401, detail="Invalid key version")
        
        payload = jwt.decode(token, key_version.key, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        roles: List[str] = payload.get("roles", [])
        
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return TokenData(username=username, roles=roles)
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db = Depends(get_db)
) -> User:
    token_data = verify_token(token)
    user = await db.get_user(username=token_data.username)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user 