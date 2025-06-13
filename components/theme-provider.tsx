"use client"

import * as React from "react"
import { ThemeProvider as NextThemesProvider } from "next-themes"

// Define our own ThemeProviderProps
interface ThemeProviderProps {
  children: React.ReactNode
  [key: string]: any
}

const colors = {
  dark: {
    background: "hsl(240 10% 3.9%)",
    foreground: "hsl(0 0% 98%)",
    primary: "hsl(217.2 91.2% 59.8%)",
    secondary: "hsl(217.2 32.6% 17.5%)",
    accent: "hsl(142.1 70.6% 45.3%)",
    muted: "hsl(217.2 32.6% 17.5%)",
    border: "hsl(217.2 32.6% 17.5%)",
  }
}

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  const [mounted, setMounted] = React.useState(false)

  // Ensure we're not rendering theme styles on the server
  React.useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    // Return children to avoid layout shift, but without theme context
    return <>{children}</>
  }

  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem
      disableTransitionOnChange
      storageKey="astroshield-theme-preference"
      {...props}
    >
      {children}
    </NextThemesProvider>
  )
}

