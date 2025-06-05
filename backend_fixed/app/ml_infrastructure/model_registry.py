"""
Real ML Model Registry Implementation for AstroShield.

This module provides comprehensive model management capabilities including
versioning, metadata storage, performance tracking, and deployment management.
"""

import os
import json
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pickle
import joblib
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, DateTime, Float, Text, Boolean, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)
router = APIRouter()

# Database models
Base = declarative_base()

class MLModelDB(Base):
    """Database model for ML model metadata."""
    __tablename__ = "ml_models"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    framework = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, nullable=False)
    
    # Model artifacts
    model_path = Column(String, nullable=False)
    model_size_bytes = Column(Integer)
    model_hash = Column(String)
    
    # Model configuration
    hyperparameters = Column(JSON)
    training_config = Column(JSON)
    preprocessing_config = Column(JSON)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    custom_metrics = Column(JSON)
    
    # Deployment status
    is_active = Column(Boolean, default=False)
    is_deployed = Column(Boolean, default=False)
    deployment_env = Column(String)
    endpoint_url = Column(String)
    
    # Validation
    is_validated = Column(Boolean, default=False)
    validation_date = Column(DateTime)
    validation_metrics = Column(JSON)

# Pydantic models
class ModelMetadata(BaseModel):
    """Enhanced model metadata with comprehensive information."""
    id: str
    name: str
    version: str
    model_type: str
    framework: str
    description: Optional[str] = None
    created_at: str
    updated_at: str
    created_by: str
    status: str
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    custom_metrics: Dict[str, float] = {}
    
    # Model configuration
    hyperparameters: Dict[str, Any] = {}
    training_config: Dict[str, Any] = {}
    preprocessing_config: Dict[str, Any] = {}
    
    # Deployment info
    is_active: bool = False
    is_deployed: bool = False
    deployment_env: Optional[str] = None
    endpoint_url: Optional[str] = None
    
    # Model artifacts
    model_size_bytes: Optional[int] = None
    model_hash: Optional[str] = None

class ModelRegistration(BaseModel):
    """Enhanced model registration with comprehensive metadata."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Type of model (classifier, regressor, etc.)")
    framework: str = Field(..., description="ML framework (scikit-learn, pytorch, etc.)")
    description: Optional[str] = Field(None, description="Model description")
    created_by: str = Field(..., description="Creator of the model")
    
    # Model configuration
    hyperparameters: Dict[str, Any] = Field({}, description="Model hyperparameters")
    training_config: Dict[str, Any] = Field({}, description="Training configuration")
    preprocessing_config: Dict[str, Any] = Field({}, description="Preprocessing configuration")
    
    # Performance metrics
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Model accuracy")
    precision: Optional[float] = Field(None, ge=0, le=1, description="Model precision")
    recall: Optional[float] = Field(None, ge=0, le=1, description="Model recall")
    f1_score: Optional[float] = Field(None, ge=0, le=1, description="Model F1 score")
    custom_metrics: Dict[str, float] = Field({}, description="Custom metrics")

class ModelUpdate(BaseModel):
    """Enhanced model update with validation."""
    description: Optional[str] = None
    accuracy: Optional[float] = Field(None, ge=0, le=1)
    precision: Optional[float] = Field(None, ge=0, le=1)
    recall: Optional[float] = Field(None, ge=0, le=1)
    f1_score: Optional[float] = Field(None, ge=0, le=1)
    custom_metrics: Optional[Dict[str, float]] = None
    is_active: Optional[bool] = None
    is_deployed: Optional[bool] = None
    deployment_env: Optional[str] = None
    endpoint_url: Optional[str] = None

class ModelPerformanceMetrics(BaseModel):
    """Model performance tracking."""
    model_id: str
    timestamp: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    prediction_count: Optional[int] = None
    error_count: Optional[int] = None
    environment: str = "production"

class ModelRegistryService:
    """Comprehensive ML Model Registry Service."""
    
    def __init__(self, database_url: str = None, storage_path: str = None):
        """Initialize the model registry service."""
        self.database_url = database_url or "sqlite:///./astroshield_models.db"
        self.storage_path = Path(storage_path or "./models")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.engine = create_engine(self.database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Model cache
        self._model_cache = {}
        
        logger.info("Model registry service initialized")

    def get_db(self):
        """Get database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def list_models(self, name: Optional[str] = None, model_type: Optional[str] = None,
                   is_active: Optional[bool] = None, limit: int = 100) -> List[ModelMetadata]:
        """List ML models with optional filtering."""
        try:
            db = next(self.get_db())
            query = db.query(MLModelDB)
            
            if name:
                query = query.filter(MLModelDB.name == name)
            if model_type:
                query = query.filter(MLModelDB.model_type == model_type)
            if is_active is not None:
                query = query.filter(MLModelDB.is_active == is_active)
            
            models = query.order_by(MLModelDB.updated_at.desc()).limit(limit).all()
            
            return [self._db_to_metadata(model) for model in models]
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get a specific model by ID."""
        try:
            # Check cache first
            if model_id in self._model_cache:
                return self._model_cache[model_id]
            
            db = next(self.get_db())
            model = db.query(MLModelDB).filter(MLModelDB.id == model_id).first()
            
            if model:
                metadata = self._db_to_metadata(model)
                self._model_cache[model_id] = metadata
                return metadata
                
        except Exception as e:
            logger.error(f"Error getting model {model_id}: {str(e)}")
            
        return None

    def register_model(self, model_request: ModelRegistration, model_file: Optional[bytes] = None) -> ModelMetadata:
        """Register a new ML model."""
        try:
            # Generate model ID
            model_id = str(uuid.uuid4())
            
            # Handle model file storage
            model_path = None
            model_size = None
            model_hash = None
            
            if model_file:
                model_path, model_size, model_hash = self._store_model_file(model_id, model_file)
            
            # Determine status
            status = "active" if model_request.accuracy and model_request.accuracy > 0.8 else "registered"
            
            # Create database entry
            db_model = MLModelDB(
                id=model_id,
                name=model_request.name,
                version=model_request.version,
                model_type=model_request.model_type,
                framework=model_request.framework,
                description=model_request.description,
                created_by=model_request.created_by,
                model_path=model_path or "",
                model_size_bytes=model_size,
                model_hash=model_hash,
                hyperparameters=model_request.hyperparameters,
                training_config=model_request.training_config,
                preprocessing_config=model_request.preprocessing_config,
                accuracy=model_request.accuracy,
                precision=model_request.precision,
                recall=model_request.recall,
                f1_score=model_request.f1_score,
                custom_metrics=model_request.custom_metrics,
                is_active=(status == "active")
            )
            
            db = next(self.get_db())
            db.add(db_model)
            db.commit()
            db.refresh(db_model)
            
            registered_model = self._db_to_metadata(db_model)
            
            # Clear cache
            self._model_cache.clear()
            
            logger.info(f"Model {model_request.name} v{model_request.version} registered with ID {model_id}")
            return registered_model
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error registering model: {str(e)}")

    def update_model(self, model_id: str, update_request: ModelUpdate) -> Optional[ModelMetadata]:
        """Update model metadata."""
        try:
            db = next(self.get_db())
            model = db.query(MLModelDB).filter(MLModelDB.id == model_id).first()
            
            if not model:
                return None
            
            # Update fields
            update_data = update_request.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(model, field):
                    setattr(model, field, value)
            
            model.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(model)
            
            updated_model = self._db_to_metadata(model)
            
            # Clear cache
            if model_id in self._model_cache:
                del self._model_cache[model_id]
            
            logger.info(f"Model {model_id} updated")
            return updated_model
            
        except Exception as e:
            logger.error(f"Error updating model {model_id}: {str(e)}")
            return None

    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        try:
            db = next(self.get_db())
            model = db.query(MLModelDB).filter(MLModelDB.id == model_id).first()
            
            if not model:
                return False
            
            # Delete model file
            if model.model_path and os.path.exists(model.model_path):
                os.remove(model.model_path)
            
            # Delete database entry
            db.delete(model)
            db.commit()
            
            # Clear cache
            if model_id in self._model_cache:
                del self._model_cache[model_id]
            
            logger.info(f"Model {model_id} deleted")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {str(e)}")
            return False

    def _store_model_file(self, model_id: str, model_file: bytes) -> tuple:
        """Store model file and return path, size, and hash."""
        file_path = self.storage_path / f"{model_id}.pkl"
        with open(file_path, 'wb') as f:
            f.write(model_file)
        
        model_hash = hashlib.sha256(model_file).hexdigest()
        return str(file_path), len(model_file), model_hash

    def _db_to_metadata(self, db_model: MLModelDB) -> ModelMetadata:
        """Convert database model to metadata."""
        # Determine status based on model state
        if db_model.is_deployed:
            status = "deployed"
        elif db_model.is_active:
            status = "active"
        elif db_model.is_validated:
            status = "validated"
        else:
            status = "registered"
        
        return ModelMetadata(
            id=db_model.id,
            name=db_model.name,
            version=db_model.version,
            model_type=db_model.model_type,
            framework=db_model.framework,
            description=db_model.description,
            created_at=db_model.created_at.isoformat(),
            updated_at=db_model.updated_at.isoformat(),
            created_by=db_model.created_by,
            status=status,
            accuracy=db_model.accuracy,
            precision=db_model.precision,
            recall=db_model.recall,
            f1_score=db_model.f1_score,
            custom_metrics=db_model.custom_metrics or {},
            hyperparameters=db_model.hyperparameters or {},
            training_config=db_model.training_config or {},
            preprocessing_config=db_model.preprocessing_config or {},
            is_active=db_model.is_active,
            is_deployed=db_model.is_deployed,
            deployment_env=db_model.deployment_env,
            endpoint_url=db_model.endpoint_url,
            model_size_bytes=db_model.model_size_bytes,
            model_hash=db_model.model_hash
        )

# Global registry instance
_registry_service = ModelRegistryService()

# API Endpoints
@router.get("/models", response_model=List[ModelMetadata])
async def list_models(
    name: Optional[str] = None,
    model_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    limit: int = 100
):
    """List all available models with optional filtering."""
    return _registry_service.list_models(name, model_type, is_active, limit)

@router.get("/models/{model_id}", response_model=ModelMetadata)
async def get_model(model_id: str):
    """Get a specific model by ID."""
    model = _registry_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.post("/models", response_model=ModelMetadata)
async def register_model(
    model: ModelRegistration,
    model_file: Optional[UploadFile] = File(None)
):
    """Register a new model with optional file upload."""
    model_file_bytes = None
    if model_file:
        model_file_bytes = await model_file.read()
    
    return _registry_service.register_model(model, model_file_bytes)

@router.patch("/models/{model_id}", response_model=ModelMetadata)
async def update_model(model_id: str, update: ModelUpdate):
    """Update model metadata."""
    updated_model = _registry_service.update_model(model_id, update)
    if not updated_model:
        raise HTTPException(status_code=404, detail="Model not found")
    return updated_model

@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model."""
    success = _registry_service.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"message": "Model deleted successfully"}

@router.get("/models/{model_id}/download")
async def download_model(model_id: str):
    """Download model file."""
    model = _registry_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Implementation for file download would go here
    return {"message": "Model download endpoint - implementation needed"}

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for model registry service."""
    try:
        # Test database connection
        models = _registry_service.list_models(limit=1)
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "models_count": len(models)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        } 