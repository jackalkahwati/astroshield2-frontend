"use client"

import * as React from "react"
import { ThemeProvider as NextThemesProvider } from "next-themes"

export interface ThemeProviderProps {
  children: React.ReactNode
  storageKey?: string
  forcedTheme?: string
  enableSystem?: boolean
  disableTransitionOnChange?: boolean
  themes?: string[]
}

export function ThemeProvider({
  children,
  storageKey = "ui-theme",
  forcedTheme,
  enableSystem = true,
  disableTransitionOnChange = false,
  themes = ["light", "dark"]
}: ThemeProviderProps) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="dark"
      value={{
        light: "light",
        dark: "dark",
        system: "system"
      }}
      forcedTheme={forcedTheme}
      enableSystem={enableSystem}
      disableTransitionOnChange={disableTransitionOnChange}
      themes={themes}
      storageKey={storageKey}
    >
      {children}
    </NextThemesProvider>
  )
} 