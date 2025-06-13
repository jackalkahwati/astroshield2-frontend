# TLE Chat Component

## Overview

The TLE Chat component provides a conversational interface for analyzing Two-Line Element (TLE) sets and answering questions about orbital mechanics. It leverages the `jackal79/tle-orbit-explainer` fine-tuned model from Hugging Face for natural language explanations of orbital parameters.

## Features

- **TLE Analysis**: Parse and interpret TLE data with detailed orbital parameters
- **Conversational Interface**: Ask questions about orbital mechanics and get expert answers
- **Multi-tier Processing**:
  - Primary: Hugging Face `jackal79/tle-orbit-explainer` model
  - Secondary: AstroShield's internal TLE processing service
  - Fallback: Offline orbital calculations with enhanced analysis
- **Educational Content**: Learn about orbital mechanics, decay risk, and satellite operations
- **Example Library**: Pre-loaded TLEs and common questions for quick reference

## Components

### TLEChatInterface

The main component that provides the chat interface with the following features:

- Message history with proper styling for user and assistant messages
- Input area with support for multi-line input
- TLE validation and parsing
- Examples sidebar with categorized content
- Export and clear chat functionality

### API Integration

The component integrates with the `/api/tle-explanations/explain` endpoint which supports:

1. **TLE Analysis Mode**: Provide `line1` and `line2` parameters to analyze a TLE
2. **Conversation Mode**: Provide `query` and `conversation_mode: true` parameters for Q&A

## Usage

```tsx
import TLEChatInterface from "@/components/tle-chat/tle-chat-interface"

export default function MyPage() {
  return (
    <TLEChatInterface 
      height="600px"
      title="TLE Orbit Analyzer"
      description="Ask questions or paste TLE data for analysis"
      showExamples={true}
    />
  )
}
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `initialMessages` | `ChatMessage[]` | `[]` | Initial messages to display |
| `onSendMessage` | `(message: string) => Promise<any>` | `undefined` | Optional custom handler for sending messages |
| `showExamples` | `boolean` | `true` | Whether to show the examples sidebar |
| `height` | `string` | `"600px"` | Height of the chat interface |
| `title` | `string` | `"TLE Orbit Analyzer"` | Title displayed in the header |
| `description` | `string` | `"Ask questions or paste TLE data for AI-powered orbital analysis"` | Description displayed in the header |

## Environment Setup

To enable the AI-powered analysis, you need to set up the Hugging Face API token:

1. Create a `.env.local` file in the project root
2. Add your Hugging Face API token:
```
HUGGINGFACE_API_TOKEN=your_token_here
```

## Example TLEs

The component includes example TLEs for:
- International Space Station
- Starlink satellites
- Reentry objects
- GPS satellites

## Example Questions

Users can ask various questions about:
- TLE interpretation
- Orbital decay assessment
- Maneuver detection
- Reentry prediction
- Orbit types and characteristics

## Implementation Notes

- The component uses a responsive design that works well on both desktop and mobile
- Message history is stored in component state and can be exported as JSON
- TLE validation ensures proper format before sending to the API
- The API has fallback mechanisms for when the Hugging Face model is unavailable 