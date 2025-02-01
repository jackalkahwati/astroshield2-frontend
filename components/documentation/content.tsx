"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

const documentationSections = [
  {
    id: "getting-started",
    title: "Getting Started",
    content:
      "Welcome to AstroShield! This guide will help you get started with our satellite monitoring and control system...",
  },
  {
    id: "dashboard",
    title: "Dashboard Overview",
    content: "The AstroShield dashboard provides a comprehensive view of your satellite network...",
  },
  {
    id: "analytics",
    title: "Using Analytics",
    content: "Our advanced analytics features allow you to gain deeper insights into your satellite operations...",
  },
  {
    id: "reporting",
    title: "Reporting and Exports",
    content: "Learn how to generate custom reports and export data from AstroShield...",
  },
]

export function DocumentationContent() {
  const [activeTab, setActiveTab] = useState("getting-started")

  return (
    <Card>
      <CardHeader>
        <CardTitle>User Guide</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            {documentationSections.map((section) => (
              <TabsTrigger key={section.id} value={section.id}>
                {section.title}
              </TabsTrigger>
            ))}
          </TabsList>
          {documentationSections.map((section) => (
            <TabsContent key={section.id} value={section.id}>
              <h3 className="text-lg font-semibold mb-2">{section.title}</h3>
              <p>{section.content}</p>
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  )
}

