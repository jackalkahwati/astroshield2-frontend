"use client"

import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useEffect, useState } from "react"

export function ModeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    const fetchInitialTheme = async () => {
      try {
        const response = await fetch('http://localhost:8001/api/settings')
        const settings = await response.json()
        if (settings.theme) {
          setTheme(settings.theme)
        }
      } catch (error) {
        console.error('Error fetching theme:', error)
        // Set a default theme if fetch fails
        setTheme('dark')
      }
      setMounted(true)
    }
    
    fetchInitialTheme()
  }, [setTheme])

  const updateTheme = async (newTheme: string) => {
    try {
      // Update theme in the UI first for immediate feedback
      setTheme(newTheme)

      // Then sync with backend
      const response = await fetch('http://localhost:8001/api/settings', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ theme: newTheme }),
      })

      if (!response.ok) {
        throw new Error('Failed to update theme in backend')
      }
    } catch (error) {
      console.error('Error updating theme:', error)
      // Revert theme if backend update fails
      setTheme(theme || 'dark')
    }
  }

  if (!mounted) {
    return null
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="icon">
          <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          <span className="sr-only">Toggle theme</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => updateTheme("light")}>
          Light
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => updateTheme("dark")}>
          Dark
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
} 