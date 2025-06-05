"""
Updated model serving with TLE explainer integration
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import random
import math
import os
import logging
from datetime import datetime

from .tle_explainer import TLEExplainerInput, TLEExplanation

logger = logging.getLogger(__name__)

router = APIRouter()

# Conditional import based on environment
USE_HUGGINGFACE_MODEL = os.getenv("USE_HUGGINGFACE_MODEL", "false").lower() == "true"

if USE_HUGGINGFACE_MODEL:
    try:
        from .tle_explainer_hf import TLEExplainerServiceHF
        tle_explainer_service = TLEExplainerServiceHF()
        logger.info("Using Hugging Face TLE explainer model")
    except Exception as e:
        logger.warning(f"Failed to load HF model, falling back to mock: {e}")
        from .tle_explainer import MockTLEExplainerService
        tle_explainer_service = MockTLEExplainerService()
else:
    from .tle_explainer import MockTLEExplainerService
    tle_explainer_service = MockTLEExplainerService()

class PredictionInput(BaseModel):
    model_name: str
    input_data: Dict[str, Any]

class PredictionOutput(BaseModel):
    model: str
    result: Dict[str, Any]

# Mock ML models with realistic space domain predictions
ML_MODELS = {
    "collision_risk_predictor": {
        "description": "Predicts collision probability between two objects",
        "version": "v2.1.0",
        "accuracy": 0.94
    },
    "trajectory_predictor": {
        "description": "Predicts future satellite trajectory",
        "version": "v1.5.2",
        "accuracy": 0.91
    },
    "tle_orbit_explainer": {
        "description": "Translates TLEs into natural language explanations with decay risk and anomaly detection",
        "version": "v1.0.0",
        "accuracy": 0.92,
        "model_type": "LoRA",
        "base_model": "Qwen/Qwen1.5-7B",
        "status": "active" if USE_HUGGINGFACE_MODEL else "mock"
    }
}

@router.get("/models/", response_model=Dict[str, Any])
async def list_models():
    """List all available ML models"""
    return {
        "models": ML_MODELS,
        "total_count": len(ML_MODELS),
        "status": "operational"
    }

@router.post("/models/tle-explainer/predict", response_model=PredictionOutput)
async def explain_tle(tle_input: TLEExplainerInput):
    """Convert TLE to natural language explanation"""
    try:
        explanation = await tle_explainer_service.explain_tle(tle_input)
        
        return PredictionOutput(
            model="tle_orbit_explainer",
            result={
                "explanation": explanation.dict(),
                "model_version": ML_MODELS["tle_orbit_explainer"]["version"],
                "model_accuracy": ML_MODELS["tle_orbit_explainer"]["accuracy"],
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": 150.0  # Mock processing time
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TLE explanation failed: {str(e)}")

@router.get("/models/tle-explainer/status")
async def get_tle_explainer_status():
    """Get status of TLE explainer model"""
    status = {
        "model_loaded": False,
        "using_huggingface": USE_HUGGINGFACE_MODEL,
        "device": "unknown",
        "cache_size": 0
    }
    
    if USE_HUGGINGFACE_MODEL and hasattr(tle_explainer_service, 'hf_model'):
        hf_model = tle_explainer_service.hf_model
        if hf_model:
            status.update({
                "model_loaded": hf_model._model_loaded,
                "device": hf_model.device if hasattr(hf_model, 'device') else "unknown",
                "cache_size": len(tle_explainer_service._explanation_cache)
            })
        
    return status

@router.post("/models/tle-explainer/warmup")
async def warmup_tle_explainer():
    """Warmup the TLE explainer model"""
    if hasattr(tle_explainer_service, 'warmup'):
        await tle_explainer_service.warmup()
        return {"status": "warmed up"}
    else:
        return {"status": "warmup not needed for mock implementation"} 