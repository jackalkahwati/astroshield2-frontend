"use client"

import * as React from "react"
import { createContext, useContext, useEffect, useState } from "react"

type Theme = "dark" | "light" | "system"

type ThemeProviderProps = {
  children: React.ReactNode
  defaultTheme?: Theme
  storageKey?: string
}

type ThemeProviderState = {
  theme: Theme
  setTheme: (theme: Theme) => void
}

const initialState: ThemeProviderState = {
  theme: "dark", // Always default to dark for space operations
  setTheme: () => null,
}

const ThemeProviderContext = createContext<ThemeProviderState>(initialState)

export function ThemeProvider({
  children,
  defaultTheme = "dark", // Always default to dark for Material Design
  storageKey = "astroshield-ui-theme",
  ...props
}: ThemeProviderProps) {
  const [theme, setTheme] = useState<Theme>(defaultTheme)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    
    // Check localStorage for saved theme preference
    const storedTheme = localStorage.getItem(storageKey) as Theme
    if (storedTheme && (storedTheme === "dark" || storedTheme === "light" || storedTheme === "system")) {
      setTheme(storedTheme)
    } else {
      // Force dark theme for space operations
      setTheme("dark")
      localStorage.setItem(storageKey, "dark")
    }
  }, [storageKey])

  useEffect(() => {
    if (!mounted) return

    const root = window.document.documentElement
    const body = window.document.body

    // Remove all theme classes
    root.classList.remove("light", "dark")
    body.classList.remove("light", "dark")

    let activeTheme = theme

    if (theme === "system") {
      const systemTheme = window.matchMedia("(prefers-color-scheme: dark)")
        .matches
        ? "dark"
        : "light"
      activeTheme = systemTheme
    }

    // Apply theme classes to both html and body
    root.classList.add(activeTheme)
    body.classList.add(activeTheme)
    
    // Set CSS custom property for theme
    root.style.setProperty('--theme', activeTheme)
    
    // Force Material Design dark theme styling
    if (activeTheme === "dark") {
      root.style.setProperty('--background', '#121212')
      root.style.setProperty('--foreground', 'rgba(255, 255, 255, 0.87)')
    } else {
      root.style.setProperty('--background', '#ffffff')
      root.style.setProperty('--foreground', 'rgba(0, 0, 0, 0.87)')
    }
  }, [theme, mounted])

  const value = {
    theme,
    setTheme: (theme: Theme) => {
      localStorage.setItem(storageKey, theme)
      setTheme(theme)
    },
  }

  // Prevent hydration mismatch by not rendering until mounted
  if (!mounted) {
    return (
      <ThemeProviderContext.Provider value={initialState}>
        <div className="loading-theme" style={{ 
          backgroundColor: '#121212', 
          color: 'rgba(255, 255, 255, 0.87)',
          minHeight: '100vh' 
        }}>
          {children}
        </div>
      </ThemeProviderContext.Provider>
    )
  }

  return (
    <ThemeProviderContext.Provider {...props} value={value}>
      {children}
    </ThemeProviderContext.Provider>
  )
}

export const useTheme = () => {
  const context = useContext(ThemeProviderContext)

  if (context === undefined)
    throw new Error("useTheme must be used within a ThemeProvider")

  return context
} 