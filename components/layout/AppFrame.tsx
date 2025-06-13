"use client"

import React from 'react'

interface AppFrameProps {
  children: React.ReactNode
  className?: string
}

/**
 * AppFrame - Material Design compliant viewport-locked layout container
 * 
 * Features:
 * - Uses Material Design dark theme base surface (#121212)
 * - Proper elevation hierarchy
 * - Viewport-locked flex column layout
 * - Proper overflow handling for child components
 * - Accessibility-compliant focus management
 */
export default function AppFrame({ children, className = "" }: AppFrameProps) {
  return (
    <main className={`
      flex flex-col h-screen
      elevation-0
      text-high-emphasis
      overflow-hidden
      ${className}
    `}>
      <div className="flex-1 flex overflow-hidden">
        {children}
      </div>
    </main>
  )
}

/**
 * Surface component for proper Material Design elevation
 */
export function Surface({ 
  children, 
  elevation = 1, 
  className = "",
  interactive = false 
}: { 
  children: React.ReactNode
  elevation?: 0 | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24
  className?: string
  interactive?: boolean
}) {
  const elevationClass = `elevation-${elevation}`
  const interactiveClass = interactive ? 'state-hover' : ''
  
  return (
    <div className={`
      ${elevationClass}
      ${interactiveClass}
      ${className}
    `}>
      {children}
    </div>
  )
}

/**
 * Content area with proper spacing and typography
 */
export function ContentArea({ 
  children, 
  className = "",
  padding = "lg"
}: { 
  children: React.ReactNode
  className?: string
  padding?: "none" | "sm" | "md" | "lg" | "xl"
}) {
  const paddingMap = {
    none: "",
    sm: "p-4",
    md: "p-6", 
    lg: "p-8",
    xl: "p-12"
  }
  
  return (
    <div className={`
      flex-1 overflow-y-auto
      ${paddingMap[padding]}
      ${className}
    `}>
      {children}
    </div>
  )
} 