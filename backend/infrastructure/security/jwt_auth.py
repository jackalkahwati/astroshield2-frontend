"""JWT Authentication and Authorization module."""

import jwt
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Callable
import os
from functools import wraps

from fastapi import HTTPException, Depends, Request, WebSocket, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Secret key for JWT signing - in production, this should be in environment variables
JWT_SECRET = os.getenv("JWT_SECRET", "astroshield_development_secret_key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 60  # 1 hour token lifetime

# Security scheme for Swagger UI
security_scheme = HTTPBearer()

class JWTHandler:
    """Handler for JWT operations."""
    
    @staticmethod
    def create_token(user_id: str, username: str, roles: List[str] = None) -> str:
        """
        Create a JWT token for a user.
        
        Args:
            user_id: User ID
            username: Username
            roles: List of roles the user has
            
        Returns:
            JWT token string
        """
        payload = {
            "sub": user_id,
            "username": username,
            "roles": roles or [],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
        }
        
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def validate_token(token: str) -> Dict[str, Any]:
        """
        Validate a JWT token and return the payload.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            jwt.PyJWTError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )

# Role-based permissions
ROLE_PERMISSIONS = {
    "admin": {
        "ccdm_read", 
        "ccdm_write", 
        "ccdm_admin", 
        "ccdm_realtime",
        "user_management",
        "system_config"
    },
    "analyst": {
        "ccdm_read", 
        "ccdm_write", 
        "ccdm_realtime"
    },
    "viewer": {
        "ccdm_read"
    }
}

def get_permissions_for_roles(roles: List[str]) -> Set[str]:
    """
    Get all permissions for a list of roles.
    
    Args:
        roles: List of role names
        
    Returns:
        Set of permission strings
    """
    permissions = set()
    for role in roles:
        if role in ROLE_PERMISSIONS:
            permissions.update(ROLE_PERMISSIONS[role])
    return permissions

# FastAPI dependency for getting current user from token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> Dict[str, Any]:
    """
    FastAPI dependency that extracts and validates the JWT token.
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        Dict with user information
        
    Raises:
        HTTPException: If token is missing or invalid
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    payload = JWTHandler.validate_token(token)
    
    return {
        "user_id": payload["sub"],
        "username": payload["username"],
        "roles": payload.get("roles", []),
        "permissions": get_permissions_for_roles(payload.get("roles", []))
    }

# Permission checker for FastAPI endpoints
def require_permission(permission: str):
    """
    Decorator factory to check if user has specific permission.
    
    Args:
        permission: Required permission string
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("current_user")
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="User dependency missing, add current_user parameter"
                )
            
            if permission not in user["permissions"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: requires {permission}"
                )
                
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Helper for WebSocket connections
async def verify_websocket_token(websocket: WebSocket, required_permission: str = None) -> Dict[str, Any]:
    """
    Verify JWT token from WebSocket initial message.
    
    Args:
        websocket: WebSocket connection
        required_permission: Optional permission to check
        
    Returns:
        Dict with user information
        
    Raises:
        WebSocketDisconnect: If token is invalid or insufficient permissions
    """
    try:
        # Expect token as first message
        token = await websocket.receive_text()
        
        # Validate token
        payload = JWTHandler.validate_token(token)
        
        # Create user info
        user = {
            "user_id": payload["sub"],
            "username": payload["username"],
            "roles": payload.get("roles", []),
            "permissions": get_permissions_for_roles(payload.get("roles", []))
        }
        
        # Check permission if required
        if required_permission and required_permission not in user["permissions"]:
            logger.warning(f"Permission denied: {required_permission} for user {user['username']}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: requires {required_permission}"
            )
            
        return user
    except Exception as e:
        logger.error(f"WebSocket authentication error: {str(e)}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="WebSocket authentication failed"
        )

class SecurityException(Exception):
    """Exception raised for security-related issues."""
    pass

def check_permission(user: Dict[str, Any], permission: str) -> bool:
    """
    Check if a user has a specific permission.
    
    Args:
        user: User dict returned from get_current_user or verify_websocket_token
        permission: Permission to check
        
    Returns:
        True if user has permission, False otherwise
    """
    return permission in user.get("permissions", set()) 