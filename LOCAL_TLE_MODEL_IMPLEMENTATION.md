# 🏠 Local TLE Model Implementation

## Overview

The AstroShield TLE Chat feature now supports **local Hugging Face model execution**! This means you can run AI-powered TLE analysis directly on your machine without needing internet access or API keys.

## 🎯 What's New

### **Priority Model Selection**
1. **🏠 Local Models** (First Priority) - Runs on your hardware
2. **🌐 Hugging Face API** (Fallback) - Cloud-based inference
3. **🧠 Advanced AI Simulation** (Backup) - Sophisticated local calculations
4. **🔌 Offline Mode** (Final Fallback) - Basic TLE parsing

### **Local Model Benefits**
- ✅ **No Internet Required** - Works completely offline
- ✅ **No API Costs** - Free unlimited usage
- ✅ **Privacy** - Your data never leaves your machine
- ✅ **Speed** - Direct hardware execution
- ✅ **Customizable** - Load any compatible Hugging Face model

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Already installed if you followed previous setup
pip install transformers torch
```

### 2. Start Backend with Local Models
```bash
cd backend_fixed
python -m uvicorn app.main:app --reload
```

### 3. Start Frontend
```bash
cd astroshield-production/frontend
npm run dev
```

### 4. Use TLE Chat
1. Open http://localhost:3000/tle-chat
2. Paste any TLE (Two-Line Element set)
3. See local AI analysis! 🧠

## 🔧 Technical Implementation

### Backend Architecture

#### **New Router: `local_ml_inference.py`**
- **Path**: `/api/v1/ml/tle-local-inference`
- **Purpose**: Handle local model execution
- **Model Loading**: Automatic lazy loading on first request
- **Fallback**: Uses smaller models if target model unavailable

```python
# Example API call
POST /api/v1/ml/tle-local-inference
{
  "line1": "1 25544U 98067A...",
  "line2": "2 25544  51.6400...",
  "model_name": "jackal79/tle-orbit-explainer"
}
```

#### **LocalTLEAnalyzer Class**
- **Singleton Pattern**: One model instance across requests
- **Async Loading**: Non-blocking model initialization
- **Smart Fallbacks**: Multiple model options if primary fails
- **Memory Management**: Efficient model caching

### Frontend Integration

#### **Updated API Route: `/api/tle-explanations/explain`**
- **Priority Order**: Local → API → Simulation → Offline
- **Smart Detection**: Automatically detects which mode was used
- **Response Flags**: Shows execution mode in UI

```typescript
// New execution flow
1. Try local model first
2. Fall back to Hugging Face API if local unavailable
3. Use advanced simulation if both fail
4. Provide offline analysis as final fallback
```

## 📊 Model Information

### **Primary Model: `jackal79/tle-orbit-explainer`**
- **Base**: Fine-tuned Qwen-1.5-7B
- **Size**: ~15GB download (first time only)
- **Purpose**: Specialized TLE analysis
- **Performance**: High-quality orbital explanations

### **Fallback Model: `microsoft/DialoGPT-medium`**
- **Size**: ~1.4GB
- **Purpose**: General text generation for TLE analysis
- **Performance**: Basic but functional

### **Hardware Requirements**
- **Minimum**: 8GB RAM, CPU-only execution
- **Recommended**: 16GB RAM + GPU for faster inference
- **Storage**: 20GB free space for models

## 🧪 Testing

### **Test Script: `test_local_tle_models.py`**
```bash
# Test complete integration
python test_local_tle_models.py

# Test with custom URLs
python test_local_tle_models.py --backend http://localhost:8000 --frontend http://localhost:3000
```

### **Manual Testing**
```bash
# Test model status
curl http://localhost:8000/api/v1/ml/local-model/status

# Test inference
curl -X POST http://localhost:8000/api/v1/ml/tle-local-inference \
  -H "Content-Type: application/json" \
  -d '{
    "line1": "1 25544U 98067A   24325.50000000  .00016717  00000-0  10270-3 0  9994",
    "line2": "2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263"
  }'
```

## 🎨 UI Indicators

### **Chat Interface Shows:**
- 🏠 **Local AI Model Analysis** - When local model is used
- 🌐 **Hugging Face API Analysis** - When API is used
- 🧠 **AI Simulation** - When advanced simulation is used
- 🔌 **Offline Mode** - When basic parsing is used

### **Anomaly Flags:**
- `LOCAL_AI_MODEL` - Local model execution
- `TRANSFORMERS` - Local transformers library used
- `OFFLINE_CAPABLE` - Can work without internet

## 📖 API Reference

### **Endpoints**

#### `POST /api/v1/ml/tle-local-inference`
**Description**: Run local TLE analysis
**Request Body**:
```json
{
  "line1": "string",
  "line2": "string", 
  "use_local_model": true,
  "model_name": "jackal79/tle-orbit-explainer",
  "max_tokens": 200,
  "temperature": 0.3
}
```

#### `GET /api/v1/ml/local-model/status`
**Description**: Check model loading status
**Response**:
```json
{
  "transformers_available": true,
  "model_loaded": false,
  "loading": false,
  "torch_available": true,
  "cuda_available": false,
  "device_count": 0
}
```

#### `POST /api/v1/ml/local-model/load`
**Description**: Manually trigger model loading
**Parameters**: `model_name` (optional)

#### `GET /api/v1/ml/local-model/test`
**Description**: Test model with sample TLE

## 🔧 Configuration

### **Environment Variables**
```bash
# Optional: Prefer local models over API
export USE_LOCAL_MODELS=true

# Optional: Model cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Optional: CUDA device selection
export CUDA_VISIBLE_DEVICES=0
```

### **Model Selection Priority**
1. `jackal79/tle-orbit-explainer` (if available)
2. `microsoft/DialoGPT-medium` (fallback)
3. Any compatible text generation model

## 🚨 Troubleshooting

### **Common Issues**

#### **Model Download Fails**
```bash
# Check internet connection during first run
# Models are cached after first download
# Try smaller fallback model if primary fails
```

#### **Out of Memory**
```bash
# Reduce model size or use CPU-only mode
# Close other applications
# Consider using quantized models
```

#### **Slow Performance**
```bash
# Use GPU if available
# Reduce max_tokens parameter
# Consider model quantization
```

#### **Backend Connection Failed**
```bash
# Ensure backend is running on correct port
# Check firewall settings
# Verify transformers library is installed
```

### **Debug Commands**
```bash
# Check model status
curl http://localhost:8000/api/v1/ml/local-model/status

# Test model loading
curl -X POST http://localhost:8000/api/v1/ml/local-model/load

# Run test inference
curl -X GET http://localhost:8000/api/v1/ml/local-model/test
```

## 🎯 Usage Examples

### **Chat Interface**
```
User: Paste ISS TLE
1 25544U 98067A   24325.50000000  .00016717  00000-0  10270-3 0  9994
2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263

AstroShield: 🏠 **Local AI Model Analysis**: The International Space Station (ISS) 
operates in a Low Earth Orbit at approximately 408 km altitude. The 51.64° 
inclination provides optimal coverage for crew transportation and Earth observation...

[LOCAL_AI_MODEL] [TRANSFORMERS] [OFFLINE_CAPABLE]
```

### **API Usage**
```python
import requests

# Test local model
response = requests.post('http://localhost:8000/api/v1/ml/tle-local-inference', json={
    'line1': '1 25544U 98067A   24325.50000000  .00016717  00000-0  10270-3 0  9994',
    'line2': '2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263'
})

result = response.json()
print(f"Analysis: {result['analysis']}")
print(f"Model: {result['model_used']}")
print(f"Time: {result['inference_time_ms']}ms")
```

## 🔮 Advanced Features

### **Model Customization**
- Load custom fine-tuned models
- Adjust inference parameters
- Use quantized models for speed
- Multi-GPU support

### **Performance Optimization**
- Model caching and reuse
- Batch processing capabilities
- Async inference handling
- Memory-efficient loading

### **Integration Options**
- REST API for external applications
- WebSocket support for real-time analysis
- Batch file processing
- Custom model endpoints

## 🎉 Benefits Summary

### **For Users**
- 🚀 **Instant Analysis** - No waiting for API responses
- 🔒 **Privacy** - Data stays on your machine
- 💰 **Cost-Free** - No API usage fees
- 🌐 **Offline** - Works without internet

### **For Developers** 
- 🛠️ **Customizable** - Load any compatible model
- 📈 **Scalable** - Add more models easily
- 🔧 **Debuggable** - Full control over inference
- 🎯 **Reliable** - No external dependencies

---

**🛰️ Ready to experience local AI-powered TLE analysis in AstroShield!**

Start your servers and open http://localhost:3000/tle-chat to see the magic ✨ 