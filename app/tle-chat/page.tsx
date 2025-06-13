"use client"

import React, { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import TLEChatInterface from "@/components/tle-chat/tle-chat-interface"
import { Satellite, Brain, Activity, Shield } from "lucide-react"

export default function TLEChatPage() {
  const [activeTab, setActiveTab] = useState('chat')

  return (
    <div className="flex-1 p-6 space-y-6">
        <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
          <Satellite className="h-6 w-6" />
          <h2 className="text-3xl font-bold">TLE Chat</h2>
        </div>
      </div>

      <Tabs defaultValue="chat" value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList>
          <TabsTrigger value="chat">
            <Brain className="h-4 w-4 mr-2" />
            Chat Interface
          </TabsTrigger>
          <TabsTrigger value="about">
            <Activity className="h-4 w-4 mr-2" />
            About
          </TabsTrigger>
          <TabsTrigger value="capabilities">
            <Shield className="h-4 w-4 mr-2" />
            Capabilities
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="chat" className="space-y-4">
          <TLEChatInterface 
            height="calc(100vh - 200px)"
            title="TLE Orbit Analyzer"
            description="Ask questions or paste TLE data for AI-powered analysis"
            showExamples={true}
          />
        </TabsContent>
        
        <TabsContent value="about" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>About TLE Chat</CardTitle>
              <CardDescription>
                Powered by the fine-tuned jackal79/tle-orbit-explainer model
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="text-lg font-medium mb-2">What is TLE Chat?</h3>
                <p className="text-muted-foreground">
                  TLE Chat is an AI-powered interface for analyzing Two-Line Element (TLE) sets and answering 
                  questions about orbital mechanics. It combines a specialized fine-tuned model with AstroShield's 
                  orbital analysis capabilities to provide insights into satellite orbits, decay risks, and 
                  space operations.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">How It Works</h3>
                <p className="text-muted-foreground">
                  The system uses a multi-tier approach:
                </p>
                <ol className="list-decimal list-inside space-y-1 mt-2 text-muted-foreground">
                  <li>Primary: Hugging Face <code>jackal79/tle-orbit-explainer</code> API</li>
                  <li>Secondary: AstroShield's internal TLE processing service</li>
                  <li>Fallback: Offline orbital calculations with enhanced analysis</li>
                </ol>
          </div>

              <div>
                <h3 className="text-lg font-medium mb-2">Model Information</h3>
                <p className="text-muted-foreground">
                  The <code>jackal79/tle-orbit-explainer</code> model is a LoRA adapter for Qwen-1.5-7B,
                  specifically fine-tuned for TLE analysis and orbital mechanics. It can provide natural language
                  explanations of orbital elements, decay risk assessments, and educational content about
                  space operations.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="capabilities" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>TLE Chat Capabilities</CardTitle>
              <CardDescription>
                What you can do with the TLE Chat interface
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-2 flex items-center">
                    <Satellite className="h-5 w-5 mr-2" />
                    TLE Analysis
                  </h3>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>Parse and interpret TLE data</li>
                    <li>Identify satellite orbit types</li>
                    <li>Calculate orbital parameters</li>
                    <li>Assess decay risk levels</li>
                    <li>Estimate orbital lifetime</li>
                  </ul>
                </div>
                
                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-2 flex items-center">
                    <Brain className="h-5 w-5 mr-2" />
                    Educational Content
                  </h3>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>Explain orbital mechanics concepts</li>
                    <li>Describe different orbit types</li>
                    <li>Teach TLE interpretation</li>
                    <li>Clarify space operations terminology</li>
                    <li>Answer questions about satellites</li>
                  </ul>
            </div>
            
                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-2 flex items-center">
                    <Activity className="h-5 w-5 mr-2" />
                    Advanced Features
                  </h3>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>Detect potential maneuvers</li>
                    <li>Identify anomalies in orbital data</li>
                    <li>Compare multiple TLEs</li>
                    <li>Provide confidence scoring</li>
                    <li>Export analysis results</li>
                  </ul>
        </div>

                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-2 flex items-center">
                    <Shield className="h-5 w-5 mr-2" />
                    Operational Support
                  </h3>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>Assess reentry predictions</li>
                    <li>Evaluate collision risks</li>
                    <li>Analyze orbital stability</li>
                    <li>Support mission planning</li>
                    <li>Provide situational awareness</li>
                  </ul>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
    </div>
  )
} 