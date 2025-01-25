from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from fastapi import Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from app.core.config import settings
from app.models.user import User
from app.db.session import get_db
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import json
import os

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

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a new access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire, "key_version": key_store.current_version})
    
    current_key = key_store.get_current_key()
    if not current_key:
        raise HTTPException(
            status_code=500,
            detail="No active signing key available"
        )
    
    encoded_jwt = jwt.encode(
        to_encode, 
        current_key["private"], 
        algorithm="RS256"
    )
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify and decode a JWT token"""
    try:
        # First try to decode without verification to get key version
        unverified_payload = jwt.decode(
            token,
            options={"verify_signature": False}
        )
        key_version = unverified_payload.get("key_version")
        
        if not key_version:
            raise HTTPException(
                status_code=401,
                detail="Invalid token format"
            )
        
        # Get the key for this version
        key_data = key_store.get_key_by_version(key_version)
        if not key_data:
            raise HTTPException(
                status_code=401,
                detail="Token signed with unknown key version"
            )
        
        # Verify the token with the correct public key
        payload = jwt.decode(
            token,
            key_data["public"],
            algorithms=["RS256"]
        )
        return payload
        
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials"
        )

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db = Depends(get_db)
) -> User:
    token_data = verify_token(token)
    user = await db.get_user(username=token_data.get("sub"))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

class KeyStore:
    def __init__(self):
        self.keys: Dict[int, Dict] = {}
        self.current_version = 0
        self.load_keys()

    def load_keys(self):
        """Load keys from secure storage"""
        # In production, load from secure storage (e.g., AWS KMS, HashiCorp Vault)
        if os.path.exists("keys.json"):
            with open("keys.json", "r") as f:
                data = json.load(f)
                self.keys = {int(k): v for k, v in data["keys"].items()}
                self.current_version = data["current_version"]
        else:
            self.rotate_keys()

    def save_keys(self):
        """Save keys to secure storage"""
        # In production, save to secure storage
        with open("keys.json", "w") as f:
            json.dump({
                "keys": self.keys,
                "current_version": self.current_version
            }, f)

    def generate_key_pair(self) -> Tuple[str, str]:
        """Generate a new RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        return public_pem, private_pem

    def rotate_keys(self):
        """Generate new keys and rotate the old ones"""
        new_version = self.current_version + 1
        public_key, private_key = self.generate_key_pair()
        
        # Deactivate old keys
        for version in self.keys:
            self.keys[version]["active"] = False
        
        # Add new keys
        self.keys[new_version] = {
            "public": public_key,
            "private": private_key,
            "created_at": datetime.utcnow().isoformat(),
            "active": True
        }
        
        self.current_version = new_version
        self.save_keys()

    def get_current_key(self) -> Optional[Dict]:
        """Get the current active key"""
        return self.keys.get(self.current_version)

    def get_key_by_version(self, version: int) -> Optional[Dict]:
        """Get a specific key by version"""
        return self.keys.get(version)

    def cleanup_old_keys(self, max_age_days: int = 30):
        """Remove keys older than max_age_days"""
        now = datetime.utcnow()
        versions_to_remove = []
        
        for version, key_data in self.keys.items():
            created_at = datetime.fromisoformat(key_data["created_at"])
            age = now - created_at
            
            if age > timedelta(days=max_age_days) and not key_data["active"]:
                versions_to_remove.append(version)
        
        for version in versions_to_remove:
            del self.keys[version]
        
        if versions_to_remove:
            self.save_keys()

# Global key store instance
key_store = KeyStore() 