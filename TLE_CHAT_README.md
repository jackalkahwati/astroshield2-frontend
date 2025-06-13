# TLE Chat AI Integration

## Overview

The AstroShield TLE Chat interface provides AI-powered analysis of Two-Line Element (TLE) data using the fine-tuned `jackal79/tle-orbit-explainer` model from Hugging Face. The interface now includes a fully conversational mode for asking questions about orbital mechanics and TLE interpretation.

## Features

### 🧠 AI-Powered Analysis
- **Model**: `jackal79/tle-orbit-explainer` (Qwen-1.5-7B fine-tuned)
- **Developer**: Jack Al-Kahwati / Stardrive
- **Platform**: Hugging Face Inference API
- **Specialization**: TLE analysis and orbital mechanics

### 🛰️ Capabilities
- Natural language explanations of orbital elements
- Decay risk assessment with confidence scores
- Anomaly detection and status reporting
- Educational orbital mechanics information
- Interactive chat interface for questions
- **NEW**: Conversational Q&A about orbital mechanics

### 📊 Analysis Modes
1. **🧠 AI Mode**: Uses Hugging Face `jackal79/tle-orbit-explainer` model
2. **🖥️ Backend Mode**: Uses internal AstroShield TLE processing service
3. **🔌 Offline Mode**: Basic orbital calculations when services unavailable

### 💬 Conversational Interface
- Ask questions about TLEs and orbital mechanics
- Get expert-level answers about space operations
- Learn about satellite orbits and decay risk assessment
- Explore educational content about space domain awareness

## Setup

### Environment Variables
Create a `.env.local` file in the frontend directory:

```env
# Hugging Face API Token for AI-powered TLE analysis
# Get your token from: https://huggingface.co/settings/tokens
HUGGINGFACE_API_TOKEN=your_huggingface_token_here

# Backend API URL (optional)
BACKEND_URL=http://localhost:8000
```

### Getting a Hugging Face Token
1. Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token to your `.env.local` file

## Usage

### Basic TLE Analysis
1. Navigate to `/tle-chat` in the application
2. Paste a TLE (Two-Line Element set) in the chat input
3. The AI will provide detailed orbital analysis

### Conversational Q&A
1. Navigate to `/tle-chat` in the application
2. Type a question about TLEs or orbital mechanics
3. The AI will provide an expert answer
4. Continue the conversation with follow-up questions

### Example TLE Format
```
1 25544U 98067A   24325.50000000  .00016717  00000-0  10270-3 0  9994
2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263
```

### Example Questions
- "What is a TLE?"
- "How do you assess orbital decay risk from a TLE?"
- "How can I detect if a satellite has performed a maneuver?"
- "How accurate are reentry predictions from TLE data?"
- "What are the different types of orbits?"

## API Integration

The TLE Chat uses a multi-tier approach:

1. **Primary**: Hugging Face `jackal79/tle-orbit-explainer` API
2. **Fallback**: Backend TLE processing service
3. **Final Fallback**: Offline orbital calculations

### API Endpoints

#### TLE Analysis Endpoint
- **Path**: `/api/tle-explanations/explain`
- **Method**: POST
- **Body**: 
  ```json
  {
    "line1": "TLE line 1",
    "line2": "TLE line 2",
    "include_risk_assessment": true,
    "include_anomaly_detection": true
  }
  ```

#### Conversation Mode Endpoint
- **Path**: `/api/tle-explanations/explain`
- **Method**: POST
- **Body**: 
  ```json
  {
    "query": "What is a TLE?",
    "conversation_mode": true,
    "previous_messages": [
      {"role": "user", "content": "Previous question"},
      {"role": "assistant", "content": "Previous answer"}
    ]
  }
  ```

### Response Format
```json
{
  "norad_id": "25544",
  "satellite_name": "International Space Station (ISS)",
  "orbit_description": "AI analysis of orbital characteristics...",
  "orbit_type": "LEO",
  "altitude_description": "400-500 km altitude",
  "period_minutes": 90.5,
  "inclination_degrees": 51.64,
  "eccentricity": 0.000777,
  "decay_risk_score": 0.3,
  "decay_risk_level": "MEDIUM",
  "anomaly_flags": ["HUGGINGFACE_AI"],
  "predicted_lifetime_days": 365,
  "confidence_score": 0.85,
  "technical_details": {
    "ai_analysis": "Full AI model response...",
    "model_used": "jackal79/tle-orbit-explainer",
    "note": "Analysis powered by fine-tuned Qwen-1.5-7B model"
  }
}
```

## Model Information

### jackal79/tle-orbit-explainer
- **Type**: LoRA adapter for Qwen-1.5-7B
- **Purpose**: TLE analysis and orbital mechanics
- **License**: TLE-Orbit-NonCommercial v1.0
- **Repository**: [Hugging Face Model](https://huggingface.co/jackal79/tle-orbit-explainer)
- **Paper**: [Medium Article](https://medium.com/@jack_16944/enhancing-space-awareness-with-fine-tuned-transformer-models-introducing-tle-orbit-explainer-67ae40653ed5)

### Model Capabilities
- Quick summarization of satellite orbital states
- Plain-language TLE explanations for educational purposes
- Offline dataset labeling (orbital classifications)
- Risk assessment and anomaly detection
- **NEW**: Conversational Q&A about orbital mechanics

## File Structure

```
astroshield-production/frontend/
├── app/
│   ├── tle-chat/
│   │   └── page.tsx                    # Main TLE chat interface
│   └── api/
│       └── tle-explanations/
│           └── explain/
│               └── route.ts            # API route with HF integration
├── components/
│   ├── tle-chat/
│   │   ├── tle-chat-interface.tsx      # Reusable chat component
│   │   └── README.md                   # Component documentation
│   └── ui/                            # Reusable UI components
└── TLE_CHAT_README.md                 # This file
```

## Development

### Running Locally
```bash
cd astroshield-production/frontend
npm install
npm run dev
```

### Testing the Integration
1. Start the frontend: `npm run dev`
2. Navigate to `http://localhost:3000/tle-chat`
3. Try pasting an example TLE from the sidebar
4. Try asking a question about orbital mechanics
5. Watch for AI analysis indicators in the response

### Troubleshooting

#### No AI Analysis
- Check that `HUGGINGFACE_API_TOKEN` is set in `.env.local`
- Verify the token has correct permissions
- Check browser console for API errors

#### Falling Back to Offline Mode
- Normal behavior when Hugging Face API is unavailable
- Backend service can also provide analysis if configured
- Offline mode provides basic orbital calculations

## Contributing

When contributing to the TLE Chat feature:

1. Test all three analysis modes (AI, Backend, Offline)
2. Ensure proper error handling and fallbacks
3. Maintain the conversational chat interface
4. Update this README for any new features

## License

This integration respects the `jackal79/tle-orbit-explainer` license terms:
- ✅ Free for non-commercial use, research, and internal evaluation
- 🚫 Commercial use requires separate licensing

For commercial licensing, contact: jack@thestardrive.com 