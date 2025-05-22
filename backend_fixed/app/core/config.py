from pydantic_settings import BaseSettings
from typing import Optional
import secrets
from pydantic import EmailStr

class Settings(BaseSettings):
    PROJECT_NAME: str = "AstroShield"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment and Server
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 5002
    LOG_LEVEL: str = "info"
    
    # UDL Configuration
    UDL_BASE_URL: Optional[str] = None
    UDL_USERNAME: Optional[str] = None
    UDL_PASSWORD: Optional[str] = None
    # Add UDL_API_KEY if you might use it later
    # UDL_API_KEY: Optional[str] = None
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"
    
    # Database
    DATABASE_URL: Optional[str] = "sqlite:///./astroshield.db" # Default to local sqlite if not set
    
    # Default User Credentials (For initial setup scripts only; change/remove in production)
    DEFAULT_ADMIN_EMAIL: EmailStr = "admin@example.com"
    DEFAULT_ADMIN_USER_PASSWORD: str = "change_this_password"
    DEFAULT_TEST_USER_PASSWORD: str = "change_this_password"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 