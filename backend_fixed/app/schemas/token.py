from pydantic import BaseModel
from typing import Optional
from datetime import datetime # For expires_at

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: datetime # To inform client about token expiry

class TokenPayload(BaseModel):
    sub: Optional[str] = None # Subject (usually user email or ID)
    # You can add other fields you expect in the token payload here
    # For example, roles: Optional[list[str]] = None
    # is_superuser: Optional[bool] = False 