"use client"

import React, { createContext, useContext, useState, useEffect } from "react"

type SidebarContextType = {
  expanded: boolean
  setExpanded: (expanded: boolean) => void
}

const SidebarContext = createContext<SidebarContextType>({
  expanded: false,
  setExpanded: () => {},
})

export function SidebarProvider({
  children,
}: {
  children: React.ReactNode
}) {
  // Check for saved preference in localStorage, default to true on large screens, false on mobile
  const [expanded, setExpanded] = useState<boolean>(true)
  
  // Initialize state based on screen size and saved preference
  useEffect(() => {
    // Get saved preference or default to expanded on desktop, collapsed on mobile
    const savedState = localStorage.getItem('astroshield-sidebar-expanded')
    const isMobile = window.innerWidth < 768
    
    if (savedState !== null) {
      setExpanded(savedState === 'true')
    } else {
      setExpanded(!isMobile)
    }
    
    // Update on window resize for responsive behavior
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setExpanded(false)
      }
    }
    
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])
  
  // Save preference to localStorage
  const handleSetExpanded = (value: boolean) => {
    setExpanded(value)
    localStorage.setItem('astroshield-sidebar-expanded', value.toString())
  }

  return (
    <SidebarContext.Provider
      value={{
        expanded,
        setExpanded: handleSetExpanded,
      }}
    >
      {children}
    </SidebarContext.Provider>
  )
}

export const useSidebar = () => useContext(SidebarContext) 