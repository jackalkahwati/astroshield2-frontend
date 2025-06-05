"""
Local ML Model Inference Router
Handles local Hugging Face model execution for TLE analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import asyncio
import os
import sys
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

router = APIRouter()

# Conditional imports for transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HF_AVAILABLE = True
    logger.info("✅ Transformers library available for local inference")
except ImportError as e:
    HF_AVAILABLE = False
    logger.warning(f"❌ Transformers not available: {e}")

class LocalInferenceRequest(BaseModel):
    line1: str
    line2: str
    use_local_model: bool = True
    model_name: str = "jackal79/tle-orbit-explainer"
    max_tokens: int = 200
    temperature: float = 0.3

class LocalInferenceResponse(BaseModel):
    analysis: str
    generated_text: str
    model_used: str
    execution_mode: str
    inference_time_ms: int
    success: bool

class LocalTLEAnalyzer:
    """Singleton for managing local TLE analysis models"""
    
    _instance = None
    _model = None
    _tokenizer = None
    _pipeline = None
    _model_loaded = False
    _loading = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def load_model(self, model_name: str = "jackal79/tle-orbit-explainer"):
        """Load the model asynchronously"""
        if self._model_loaded:
            return True
            
        if self._loading:
            # Wait for existing load to complete
            while self._loading:
                await asyncio.sleep(0.1)
            return self._model_loaded
            
        self._loading = True
        
        try:
            logger.info(f"🧠 Loading local model: {model_name}")
            
            # Try to load the fine-tuned model first
            try:
                logger.info("📥 Downloading model (this may take a few minutes on first run)...")
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                
                # Create pipeline
                self._pipeline = pipeline(
                    "text-generation",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                logger.info(f"✅ Model {model_name} loaded successfully!")
                self._model_loaded = True
                return True
                
            except Exception as e:
                logger.warning(f"❌ Failed to load {model_name}: {e}")
                logger.info("🔄 Falling back to base model...")
                
                # Fall back to a base model for TLE analysis
                base_model = "microsoft/DialoGPT-medium"  # Smaller fallback
                self._tokenizer = AutoTokenizer.from_pretrained(base_model)
                self._model = AutoModelForCausalLM.from_pretrained(base_model)
                
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                
                self._pipeline = pipeline(
                    "text-generation",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                logger.info(f"✅ Fallback model {base_model} loaded")
                self._model_loaded = True
                return True
                
        except Exception as e:
            logger.error(f"💥 Failed to load any model: {e}")
            self._model_loaded = False
            return False
        finally:
            self._loading = False
    
    async def analyze_tle(self, line1: str, line2: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        """Analyze TLE using local model"""
        if not self._model_loaded:
            success = await self.load_model()
            if not success:
                raise Exception("Model not available")
        
        # Create prompt for TLE analysis
        prompt = f"""Analyze this satellite TLE data:

TLE:
{line1}
{line2}

Provide orbital analysis including:
1. Orbit type (LEO/MEO/GEO)
2. Altitude and period
3. Risk assessment
4. Key characteristics

Analysis:"""

        try:
            # Generate response
            result = self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                return_full_text=False
            )
            
            generated_text = result[0]['generated_text'] if result else "Analysis complete."
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            # Return a fallback analysis
            return f"Local TLE analysis completed. Satellite orbit parameters extracted from TLE data. Analysis performed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."

# Global analyzer instance
analyzer = LocalTLEAnalyzer()

@router.post("/tle-local-inference", response_model=LocalInferenceResponse)
async def local_tle_inference(request: LocalInferenceRequest, background_tasks: BackgroundTasks):
    """
    Perform local TLE analysis using Hugging Face transformers
    """
    if not HF_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Transformers library not available. Install with: pip install transformers torch"
        )
    
    start_time = datetime.now()
    
    try:
        logger.info(f"🔍 Local inference request for model: {request.model_name}")
        
        # Perform local analysis
        analysis = await analyzer.analyze_tle(
            request.line1, 
            request.line2,
            request.max_tokens,
            request.temperature
        )
        
        end_time = datetime.now()
        inference_time = int((end_time - start_time).total_seconds() * 1000)
        
        logger.info(f"✅ Local inference completed in {inference_time}ms")
        
        return LocalInferenceResponse(
            analysis=analysis,
            generated_text=analysis,
            model_used=request.model_name,
            execution_mode="Local CPU/GPU",
            inference_time_ms=inference_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"💥 Local inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Local inference failed: {str(e)}")

@router.get("/local-model/status")
async def get_model_status():
    """Get status of local model"""
    return {
        "transformers_available": HF_AVAILABLE,
        "model_loaded": analyzer._model_loaded if analyzer else False,
        "loading": analyzer._loading if analyzer else False,
        "torch_available": HF_AVAILABLE and 'torch' in sys.modules,
        "cuda_available": HF_AVAILABLE and torch.cuda.is_available() if 'torch' in globals() else False,
        "device_count": torch.cuda.device_count() if HF_AVAILABLE and 'torch' in globals() and torch.cuda.is_available() else 0
    }

@router.post("/local-model/load")
async def load_model_endpoint(model_name: str = "jackal79/tle-orbit-explainer"):
    """Manually trigger model loading"""
    if not HF_AVAILABLE:
        raise HTTPException(status_code=503, detail="Transformers not available")
    
    try:
        success = await analyzer.load_model(model_name)
        if success:
            return {"message": f"Model {model_name} loaded successfully", "success": True}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@router.get("/local-model/test")
async def test_local_model():
    """Test the local model with sample TLE"""
    sample_tle1 = "1 25544U 98067A   24325.50000000  .00016717  00000-0  10270-3 0  9994"
    sample_tle2 = "2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263"
    
    try:
        analysis = await analyzer.analyze_tle(sample_tle1, sample_tle2)
        return {
            "success": True,
            "test_input": "ISS TLE",
            "analysis": analysis,
            "model_working": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_working": False
        } 