"use client"

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/components/providers/theme-provider'
import { useSidebar } from '@/components/providers/sidebar-provider'
import { 
  Search, 
  Bell, 
  Sun, 
  Moon, 
  User,
  Menu,
  ChevronDown
} from 'lucide-react'

/**
 * TopBar - Material Design App Bar
 * 
 * Features:
 * - 8dp elevation with proper surface color
 * - Material Design app bar patterns
 * - Proper spacing and typography
 * - Accessibility-compliant controls
 * - Responsive design considerations
 * - Hamburger menu for sidebar toggle
 */
export function TopBar() {
  const [searchExpanded, setSearchExpanded] = useState(false)
  const { theme, setTheme } = useTheme()
  const { expanded, setExpanded } = useSidebar()

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark")
  }

  const toggleSidebar = () => {
    setExpanded(!expanded)
  }

  return (
    <header className="elevation-8 border-b border-subtle backdrop-blur supports-[backdrop-filter]:bg-surface-8/60">
      <div className="flex h-16 items-center px-4">
        {/* Hamburger Menu - Left Side */}
        <div className="flex items-center">
          <Button
            variant="ghost"
            size="sm"
            className="md-button-text h-9 w-9 mr-3"
            type="button"
            onClick={toggleSidebar}
            aria-label="Toggle sidebar"
          >
            <Menu className="h-5 w-5" />
          </Button>
        </div>

        {/* Right Side - Search and Controls */}
        <div className="ml-auto flex items-center gap-4">
          <div className="relative">
            <form className="flex items-center">
              <div className={`
                overflow-hidden transition-all duration-300 ease-in-out
                ${searchExpanded ? 'w-64' : 'w-0'}
              `}>
                <input
                  type="text"
                  className="md-input h-9 w-full rounded-l-md border border-r-0"
                  placeholder="Search..."
                  style={{ opacity: searchExpanded ? 1 : 0 }}
                />
              </div>
              <Button
                variant="outline"
                size="sm"
                className="md-button-outlined h-9 w-9 rounded-md transition-all duration-300 ease-in-out"
                type="button"
                onClick={() => setSearchExpanded(!searchExpanded)}
              >
                <Search className="h-4 w-4" />
              </Button>
            </form>
          </div>

          {/* Notifications */}
          <Button
            variant="outline"
            size="sm"
            className="md-button-outlined h-9 w-9 relative"
            type="button"
            aria-label="Notifications"
          >
            <Bell className="h-4 w-4" />
            {/* Notification badge */}
            <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-error text-[10px] font-bold text-white flex items-center justify-center">
              3
            </span>
          </Button>

          {/* Theme Toggle */}
          <Button
            variant="outline"
            size="sm"
            className="md-button-outlined h-9 w-9"
            type="button"
            onClick={toggleTheme}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? (
              <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all" />
            ) : (
              <Moon className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all" />
            )}
            <span className="sr-only">Toggle theme</span>
          </Button>

          {/* User Menu */}
          <Button
            variant="ghost"
            className="md-button-text h-9 w-9 relative"
            type="button"
            aria-label="User menu"
          >
            <User className="h-5 w-5" />
            <span className="sr-only">User menu</span>
          </Button>
        </div>
      </div>
    </header>
  )
}

