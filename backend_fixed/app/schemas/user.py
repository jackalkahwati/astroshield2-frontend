from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime # Keep if your UserInDB includes timestamps

# Shared properties
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False
    roles: Optional[List[str]] = []

# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str

# Properties to receive via API on update
class UserUpdate(UserBase):
    password: Optional[str] = None

# Properties stored in DB
class UserInDBBase(UserBase):
    id: int
    # Add created_at, updated_at if they are in your SQLAlchemy model and you want to expose them
    # created_at: datetime 
    # updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True # Replaces orm_mode = True

# Additional properties to return via API
class User(UserInDBBase):
    pass

# Additional properties stored in DB
class UserInDB(UserInDBBase):
    hashed_password: str 