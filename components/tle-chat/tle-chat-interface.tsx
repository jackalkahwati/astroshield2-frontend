import React, { useState, useRef, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  Loader2, Rocket, Star, Target, Satellite, 
  Info, Activity, AlertTriangle, Eye, Shield, FileText, Brain,
  Navigation, Radar, Clipboard, Network, Cpu, BarChart2, Zap, Copy,
  Download, RefreshCw, Send, Trash2
} from "lucide-react"
import { toast } from "sonner"

interface TLEExplanation {
  norad_id: string
  satellite_name: string
  orbit_description: string
  orbit_type: string
  altitude_description: string
  period_minutes: number
  inclination_degrees: number
  eccentricity: number
  decay_risk_score: number
  decay_risk_level: string
  anomaly_flags: string[]
  predicted_lifetime_days?: number
  last_updated?: string
  confidence_score: number
  technical_details?: Record<string, any>
}

interface ChatMessage {
  id: string
  type: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  tleData?: TLEExplanation
}

interface TLEChatInterfaceProps {
  initialMessages?: ChatMessage[]
  onSendMessage?: (message: string) => Promise<any>
  showExamples?: boolean
  height?: string
  title?: string
  description?: string
}

// Example TLEs for the interface
const EXAMPLE_TLES = [
  {
    name: "International Space Station",
    description: "Low Earth Orbit • Active human spaceflight",
    icon: <Rocket className="h-4 w-4" />,
    content: `1 25544U 98067A   25008.50000000  .00001234  00000+0  23456-4 0  9999
2 25544  51.6400 250.0000 0005000  90.0000 270.0000 15.48000000123456`
  },
  {
    name: "Starlink Satellite",
    description: "LEO constellation • Communications",
    icon: <Satellite className="h-4 w-4" />,
    content: `1 43013U 17083A   25008.50000000  .00002000  00000+0  12345-4 0  9999
2 43013  53.0000 180.0000 0001000  45.0000 315.0000 15.20000000100000`
  },
  {
    name: "Reentry Object",
    description: "Low altitude • High decay risk",
    icon: <AlertTriangle className="h-4 w-4" />,
    content: `1 88888U 24001A   25008.50000000  .00050000  00000+0  12345-3 0  9999
2 88888  28.5000 100.0000 0010000 180.0000 180.0000 16.50000000100000`
  },
  {
    name: "GPS Satellite",
    description: "MEO • Navigation constellation",
    icon: <Navigation className="h-4 w-4" />,
    content: `1 32260U 07047A   25008.50000000  .00000010  00000+0  00000-0 0  9999
2 32260  55.0000 120.0000 0000100 180.0000 180.0000 2.00000000100000`
  }
]

// Example questions about TLEs and orbital mechanics
const EXAMPLE_QUESTIONS = [
  {
    name: "What is a TLE?",
    description: "Basic explanation",
    icon: <Info className="h-4 w-4" />,
    content: "What is a TLE and how do I interpret it?"
  },
  {
    name: "Orbital Decay",
    description: "Risk assessment",
    icon: <Activity className="h-4 w-4" />,
    content: "How do you assess orbital decay risk from a TLE?"
  },
  {
    name: "Maneuver Detection",
    description: "Change analysis",
    icon: <Target className="h-4 w-4" />,
    content: "How can I detect if a satellite has performed a maneuver by looking at TLEs?"
  },
  {
    name: "Reentry Prediction",
    description: "End-of-life analysis",
    icon: <Shield className="h-4 w-4" />,
    content: "How accurate are reentry predictions from TLE data?"
  }
]

export default function TLEChatInterface({
  initialMessages = [],
  onSendMessage,
  showExamples = true,
  height = "600px",
  title = "TLE Orbit Analyzer",
  description = "Ask questions or paste TLE data for AI-powered orbital analysis"
}: TLEChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages.length > 0 
    ? initialMessages 
    : [{
        id: '1',
        type: 'system',
        content: '🛰️ Welcome to the TLE Orbit Analyzer! I can analyze TLE data and answer questions about orbital mechanics. Try pasting a TLE or asking a question.',
        timestamp: new Date()
      }]
  )
  
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    scrollToBottom()
  }, [messages])
  
  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight
    }
  }
  
  const validateTLE = (text: string): { isValid: boolean; error?: string } => {
    const lines = text.trim().split('\n')
    
    if (lines.length !== 2) {
      return { isValid: false, error: 'TLE must contain exactly 2 lines' }
    }

    const line1 = lines[0].trim()
    const line2 = lines[1].trim()

    if (line1.length !== 69 || line2.length !== 69) {
      return { isValid: false, error: 'Each TLE line must be exactly 69 characters' }
    }

    if (!line1.startsWith('1 ') || !line2.startsWith('2 ')) {
      return { isValid: false, error: 'Line 1 must start with "1 " and Line 2 must start with "2 "' }
    }

    return { isValid: true }
  }
  
  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      // Check if input looks like TLE data
      const validation = validateTLE(inputValue)
      
      if (validation.isValid) {
        // Handle TLE analysis
        const lines = inputValue.trim().split('\n')
        
        const response = await fetch('/api/tle-explanations/explain', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            line1: lines[0].trim(),
            line2: lines[1].trim(),
            include_risk_assessment: true,
            include_anomaly_detection: true
          })
        })

        if (!response.ok) {
          throw new Error(`API returned ${response.status}`)
        }

        const tleData: TLEExplanation = await response.json()
        
        const assistantMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: formatTLEResponse(tleData),
          timestamp: new Date(),
          tleData
        }

        setMessages(prev => [...prev, assistantMessage])
      } else {
        // Handle conversational queries
        let response
        
        if (onSendMessage) {
          // Use provided callback if available
          response = await onSendMessage(inputValue)
        } else {
          // Default implementation using the explain API with conversation mode
          response = await fetch('/api/tle-explanations/explain', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              query: inputValue,
              conversation_mode: true,
              previous_messages: messages.slice(-5).map(m => ({
                role: m.type === 'user' ? 'user' : 'assistant',
                content: m.content
              }))
            })
          })
          
          if (!response.ok) {
            throw new Error(`API returned ${response.status}`)
          }
          
          response = await response.json()
        }

        const assistantMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: typeof response === 'string' ? response : response.response || response.content || 'I processed your request, but I\'m not sure how to respond.',
          timestamp: new Date(),
          tleData: response.tleData
        }

        setMessages(prev => [...prev, assistantMessage])
      }
    } catch (error) {
      console.error('Error processing message:', error)
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: '⚠️ Error processing your request. Please try again or rephrase your question.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const formatTLEResponse = (data: TLEExplanation): string => {
    const riskEmoji = data.decay_risk_level === 'HIGH' ? '🔴' : 
                     data.decay_risk_level === 'MEDIUM' ? '🟡' : '🟢'
    
    return `🛰️ **${data.satellite_name || 'Satellite'}** (ID: ${data.norad_id})

**📊 Orbital Elements:**
• **Orbit Type:** ${data.orbit_type}
• **Altitude:** ${data.altitude_description}
• **Period:** ${data.period_minutes.toFixed(1)} minutes
• **Inclination:** ${data.inclination_degrees.toFixed(2)}°
• **Eccentricity:** ${data.eccentricity.toFixed(6)}

${riskEmoji} **Decay Risk:** ${data.decay_risk_level} (${(data.decay_risk_score * 100).toFixed(1)}%)
${data.predicted_lifetime_days ? `⏱️ **Estimated Lifetime:** ${Math.round(data.predicted_lifetime_days)} days` : ''}

🎯 **Confidence:** ${(data.confidence_score * 100).toFixed(1)}%`
  }

  // Simple markdown processor for bold text
  const processMarkdown = (text: string): string => {
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
  }

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      toast.success('Copied to clipboard!')
    } catch (error) {
      toast.error('Failed to copy to clipboard')
    }
  }

  const exportChat = () => {
    const chatData = {
      timestamp: new Date().toISOString(),
      messages: messages.map(msg => ({
        type: msg.type,
        content: msg.content,
        timestamp: msg.timestamp.toISOString(),
        tleData: msg.tleData
      }))
    }
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `tle-chat-${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    
    toast.success('Chat exported successfully!')
  }

  const clearChat = () => {
    setMessages([{
      id: '1',
      type: 'system',
      content: '🛰️ Chat cleared! I can analyze TLE data and answer questions about orbital mechanics. Try pasting a TLE or asking a question.',
      timestamp: new Date()
    }])
    toast.success('Chat cleared!')
  }

  const insertExample = (text: string) => {
    setInputValue(text)
  }

  return (
    <div className="flex h-full" style={{ height }}>
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col overflow-hidden border rounded-lg bg-card">
        {/* Header */}
        <CardHeader className="border-b bg-card pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Satellite className="h-5 w-5" />
                {title}
              </CardTitle>
              {description && (
                <CardDescription>{description}</CardDescription>
              )}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={clearChat}
              className="h-8"
            >
              <Trash2 className="h-4 w-4 mr-1" />
              Clear
            </Button>
          </div>
        </CardHeader>
        
        {/* Messages Area */}
        <div 
          ref={scrollAreaRef}
          className="flex-1 overflow-y-auto p-4 space-y-4"
        >
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[85%] p-4 rounded-lg border shadow-sm ${
                  message.type === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : message.type === 'system'
                    ? 'bg-muted text-muted-foreground'
                    : 'bg-card text-card-foreground'
                }`}
              >
                <div 
                  className="text-sm leading-relaxed"
                  dangerouslySetInnerHTML={{ __html: processMarkdown(message.content) }}
                />
                
                {message.tleData && (
                  <div className="mt-3 pt-3 border-t border-border/50">
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="outline" className="text-xs">
                        {message.tleData.orbit_type}
                      </Badge>
                      <Badge 
                        variant={
                          message.tleData.decay_risk_level === 'HIGH' ? 'destructive' :
                          message.tleData.decay_risk_level === 'MEDIUM' ? 'default' : 'secondary'
                        }
                        className="text-xs"
                      >
                        {message.tleData.decay_risk_level} Risk
                      </Badge>
                    </div>
                  </div>
                )}
                
                <div className="flex items-center justify-between mt-3 pt-2 border-t border-border/50 text-xs opacity-70">
                  <span>{message.timestamp.toLocaleTimeString()}</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(message.content)}
                    className="h-6 px-2 hover:bg-muted"
                  >
                    <Copy className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-card shadow-sm p-4 rounded-lg border flex items-center gap-3">
                <Satellite className="h-4 w-4" />
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">Analyzing data...</span>
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t p-4 bg-card">
          <div className="flex gap-3">
            <Textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  sendMessage()
                }
              }}
              placeholder="Enter TLE data or ask a question about orbital mechanics..."
              className="min-h-[60px] resize-none"
              disabled={isLoading}
            />
            
            <div className="flex flex-col gap-2">
              <Button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="h-12 w-12"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={exportChat}
                className="h-8 w-8"
                title="Export Chat"
              >
                <Download className="h-3 w-3" />
              </Button>
            </div>
          </div>
          
          <div className="mt-2 text-xs text-muted-foreground">
            Press Enter to send • Shift+Enter for new line
          </div>
        </div>
      </div>

      {/* Examples Sidebar */}
      {showExamples && (
        <div className="w-72 ml-4 border rounded-lg bg-card overflow-hidden">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Examples</CardTitle>
          </CardHeader>
          
          <div className="px-4 pb-4">
            <Tabs defaultValue="tle" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="tle" className="text-xs">TLEs</TabsTrigger>
                <TabsTrigger value="questions" className="text-xs">Questions</TabsTrigger>
              </TabsList>
              
              <TabsContent value="tle" className="mt-2 space-y-2">
                {EXAMPLE_TLES.map((example, index) => (
                  <Card 
                    key={index}
                    className="cursor-pointer transition-all hover:shadow-md border hover:border-primary/50"
                    onClick={() => insertExample(example.content)}
                  >
                    <CardContent className="p-3">
                      <div className="flex items-start gap-2">
                        <div className="text-primary mt-0.5">
                          {example.icon}
                        </div>
                        <div>
                          <h4 className="font-medium text-xs">{example.name}</h4>
                          <p className="text-xs text-muted-foreground">{example.description}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>
              
              <TabsContent value="questions" className="mt-2 space-y-2">
                {EXAMPLE_QUESTIONS.map((example, index) => (
                  <Card 
                    key={index}
                    className="cursor-pointer transition-all hover:shadow-md border hover:border-primary/50"
                    onClick={() => insertExample(example.content)}
                  >
                    <CardContent className="p-3">
                      <div className="flex items-start gap-2">
                        <div className="text-primary mt-0.5">
                          {example.icon}
                        </div>
                        <div>
                          <h4 className="font-medium text-xs">{example.name}</h4>
                          <p className="text-xs text-muted-foreground">{example.description}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>
            </Tabs>
          </div>
        </div>
      )}
    </div>
  )
} 