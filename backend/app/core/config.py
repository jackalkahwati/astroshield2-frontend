from pydantic_settings import BaseSettings
from typing import Optional
import secrets

class Settings(BaseSettings):
    PROJECT_NAME: str = "AstroShield"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "https://localhost:3000",
        "http://localhost",
        "http://localhost:8080",
        "https://asttroshieldv0-lb7e5ekcn-jackalkahwatis-projects.vercel.app"
    ]
    
    # Database
    DATABASE_URL: Optional[str] = None
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 