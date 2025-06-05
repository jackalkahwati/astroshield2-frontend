from pydantic_settings import BaseSettings
from typing import Optional
import secrets
from pydantic import EmailStr, Field, validator
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "AstroShield"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment and Server
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=5002, env="PORT")
    LOG_LEVEL: str = Field(default="info", env="LOG_LEVEL")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # UDL Configuration - Production Ready
    UDL_BASE_URL: str = Field(default="https://udl.bluestack.mil/api/v1", env="UDL_BASE_URL")
    UDL_USERNAME: Optional[str] = Field(default=None, env="UDL_USERNAME")
    UDL_PASSWORD: Optional[str] = Field(default=None, env="UDL_PASSWORD") 
    UDL_API_KEY: str = Field(default="", env="UDL_API_KEY")
    UDL_TIMEOUT: int = Field(default=30, env="UDL_TIMEOUT")
    
    # Space-Track.org Configuration
    SPACE_TRACK_USERNAME: Optional[str] = Field(default=None, env="SPACE_TRACK_USERNAME")
    SPACE_TRACK_PASSWORD: Optional[str] = Field(default=None, env="SPACE_TRACK_PASSWORD")
    SPACE_TRACK_BASE_URL: str = Field(default="https://www.space-track.org/basicspacedata/query", env="SPACE_TRACK_BASE_URL")
    
    # Security - Production Ready
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=1440, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # 24 hours
    ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    CORS_ORIGINS: str = Field(default="https://astroshield.sdataplab.com,https://app.astroshield.sdataplab.com", env="CORS_ORIGINS")
    
    @validator('CORS_ORIGINS')
    def validate_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    # Database - Production Configuration
    DATABASE_URL: str = Field(default="postgresql://astroshield:AstroShield2024SecurePass!@db:5432/astroshield", env="DATABASE_URL")
    DB_POOL_SIZE: int = Field(default=20, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=30, env="DB_MAX_OVERFLOW")
    DB_TIMEOUT: int = Field(default=30, env="DB_TIMEOUT")
    
    # Redis Configuration
    REDIS_URL: str = Field(default="redis://redis:6379/0", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    CACHE_TTL_SECONDS: int = Field(default=300, env="CACHE_TTL_SECONDS")
    
    # Production Admin Credentials - Should be changed immediately
    DEFAULT_ADMIN_EMAIL: EmailStr = Field(default="admin@astroshield.sdataplab.com", env="DEFAULT_ADMIN_EMAIL")
    DEFAULT_ADMIN_USER_PASSWORD: str = Field(default="AstroShield2024Admin!", env="DEFAULT_ADMIN_PASSWORD")
    DEFAULT_TEST_USER_PASSWORD: str = Field(default="AstroShield2024Test!", env="DEFAULT_TEST_PASSWORD")
    
    # Email configuration - Production SMTP
    SMTP_TLS: bool = Field(default=True, env="SMTP_TLS")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_HOST: str = Field(default="smtp.astroshield.sdataplab.com", env="SMTP_HOST")
    SMTP_USER: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    EMAILS_FROM_EMAIL: EmailStr = Field(default="noreply@astroshield.sdataplab.com", env="EMAILS_FROM_EMAIL")
    EMAILS_FROM_NAME: str = Field(default="AstroShield Production", env="EMAILS_FROM_NAME")
    EMAILS_ENABLED: bool = Field(default=True, env="EMAILS_ENABLED")
    
    # SDA Welders Arc Integration - Production Ready
    KAFKA_BOOTSTRAP_SERVERS: str = Field(
        default="b-1-public.apollocluster.isivjt.c3.kafka.us-gov-west-1.amazonaws.com:9196,b-2-public.apollocluster.isivjt.c3.kafka.us-gov-west-1.amazonaws.com:9196", 
        env="KAFKA_BOOTSTRAP_SERVERS"
    )
    KAFKA_SECURITY_PROTOCOL: str = Field(default="SASL_SSL", env="KAFKA_SECURITY_PROTOCOL")
    KAFKA_SASL_MECHANISM: str = Field(default="SCRAM-SHA-512", env="KAFKA_SASL_MECHANISM")
    KAFKA_USERNAME: Optional[str] = Field(default=None, env="KAFKA_USERNAME")
    KAFKA_PASSWORD: Optional[str] = Field(default=None, env="KAFKA_PASSWORD")
    
    NODE_RED_URL: str = Field(default="https://node-red.astroshield.sdataplab.com", env="NODE_RED_URL")
    NODE_RED_USER: Optional[str] = Field(default=None, env="NODE_RED_USER")
    NODE_RED_PASSWORD: Optional[str] = Field(default=None, env="NODE_RED_PASSWORD")
    
    # VANTIQ Integration - Production Ready
    VANTIQ_API_URL: str = Field(default="https://astroshield.vantiq.com/api/v1", env="VANTIQ_API_URL")
    VANTIQ_API_TOKEN: str = Field(default="", env="VANTIQ_API_TOKEN")
    VANTIQ_NAMESPACE: str = Field(default="astroshield_production", env="VANTIQ_NAMESPACE")
    VANTIQ_USERNAME: Optional[str] = Field(default=None, env="VANTIQ_USERNAME")
    VANTIQ_PASSWORD: Optional[str] = Field(default=None, env="VANTIQ_PASSWORD")
    
    # Supabase Configuration
    SUPABASE_URL: Optional[str] = Field(default=None, env="SUPABASE_URL")
    SUPABASE_ANON_KEY: Optional[str] = Field(default=None, env="SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = Field(default=None, env="SUPABASE_SERVICE_ROLE_KEY")
    
    # Monitoring and Alerting
    MONITORING_ENABLED: bool = Field(default=True, env="MONITORING_ENABLED")
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_BURST: int = Field(default=200, env="RATE_LIMIT_BURST")
    
    # Feature Flags
    ML_MODELS_ENABLED: bool = Field(default=True, env="ML_MODELS_ENABLED")
    REAL_TIME_PROCESSING_ENABLED: bool = Field(default=True, env="REAL_TIME_PROCESSING_ENABLED")
    ADVANCED_ANALYTICS_ENABLED: bool = Field(default=True, env="ADVANCED_ANALYTICS_ENABLED")
    
    # SSL Configuration
    SSL_ENABLED: bool = Field(default=True, env="SSL_ENABLED")
    SSL_CERT_PATH: str = Field(default="/etc/ssl/certs/astroshield.crt", env="SSL_CERT_PATH")
    SSL_KEY_PATH: str = Field(default="/etc/ssl/private/astroshield.key", env="SSL_KEY_PATH")
    
    @validator('SECRET_KEY', pre=True)
    def validate_secret_key(cls, v):
        if v == "change-this-in-production" or len(v) < 32:
            return secrets.token_urlsafe(32)
        return v
    
    @validator('DATABASE_URL')
    def validate_database_url(cls, v):
        if "sqlite" in v and cls.ENVIRONMENT == "production":
            raise ValueError("SQLite not allowed in production environment")
        return v
    
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"
    
    def get_cors_origins(self) -> list:
        if isinstance(self.CORS_ORIGINS, str):
            return [origin.strip() for origin in self.CORS_ORIGINS.split(',')]
        return self.CORS_ORIGINS
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Validate production settings
if settings.is_production():
    if settings.DEFAULT_ADMIN_USER_PASSWORD == "change_this_password":
        raise ValueError("Production admin password must be changed from default")
    if settings.SECRET_KEY == "change-this-in-production":
        raise ValueError("Production secret key must be changed from default")
    if not settings.UDL_API_KEY:
        raise ValueError("UDL API key is required in production") 