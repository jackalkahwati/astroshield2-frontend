"use client"

import * as React from "react"
import { ThemeProvider as NextThemesProvider } from "next-themes"
import type { ThemeProviderProps as NextThemesProviderProps } from "next-themes"

export interface ThemeProviderProps {
  children: React.ReactNode
  attribute?: NextThemesProviderProps['attribute']
  defaultTheme?: string
}

export function ThemeProvider({
  children,
  attribute = "class",
  defaultTheme = "system"
}: ThemeProviderProps) {
  return (
    <NextThemesProvider
      attribute={attribute}
      defaultTheme={defaultTheme}
      enableSystem
    >
      {children}
    </NextThemesProvider>
  )
} 