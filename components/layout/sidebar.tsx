"use client"

import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { useSidebar } from '@/components/providers/sidebar-provider'
import {
  BarChart3,
  Satellite,
  MessageSquare,
  Eye,
  Navigation,
  Target,
  Zap,
  BarChart2,
  GitBranch,
  Radar,
  Lightbulb,
  Activity,
  AlertTriangle,
  Settings,
  Shield,
  Menu,
  X
} from 'lucide-react'

const navigationItems = [
  { name: 'Dashboard', href: '/', icon: BarChart3 },
  { name: 'Satellite Tracking', href: '/tracking', icon: Satellite },
  { name: 'Orbital Intelligence', href: '/tle-chat', icon: MessageSquare },
  { name: 'CCDM Analysis', href: '/ccdm', icon: Eye },
  { name: 'Trajectory Analysis', href: '/trajectory-analysis', icon: Navigation },
  { name: 'Stability Analysis', href: '/stability', icon: Target },
  { name: 'Maneuvers', href: '/maneuvers', icon: Zap },
  { name: 'Analytics', href: '/analytics', icon: BarChart2 },
  { name: 'Event Correlation', href: '/event-correlation', icon: GitBranch },
  { name: 'Proximity Operations', href: '/proximity-operations', icon: Radar },
  { name: 'Decision Support', href: '/decision-support', icon: Lightbulb },
  { name: 'Kafka Monitor', href: '/kafka-monitor', icon: Activity },
  { name: 'Protection', href: '/protection', icon: AlertTriangle },
  { name: 'Settings', href: '/settings', icon: Settings },
]

/**
 * Sidebar - Material Design Navigation Drawer
 * 
 * Features:
 * - 2dp elevation with proper surface color
 * - Material Design navigation patterns
 * - State overlays for interactive elements
 * - Proper accessibility support
 * - Responsive design with mobile overlay
 */
export function Sidebar() {
  const pathname = usePathname()
  const { expanded, setExpanded } = useSidebar()

  return (
    <>
      {/* Mobile overlay */}
      {expanded && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden" 
          onClick={() => setExpanded(false)}
        />
      )}
      
      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 flex h-full flex-col
        elevation-2
        transition-all duration-300 ease-in-out
        ${expanded ? 'w-64' : 'w-0 md:w-16'}
        md:relative md:z-0
        overflow-hidden
      `}>
        <div className="flex h-16 items-center justify-between border-b border-subtle px-4">
          <div className={`flex items-center justify-between w-full ${!expanded && 'md:justify-center'}`}>
            {/* Expanded view - Full logo */}
            <div className={`flex items-center ${!expanded && 'hidden md:hidden'}`}>
              <Shield className="h-6 w-6 text-primary" />
              <span className="ml-2 font-semibold text-high-emphasis">AstroShield</span>
            </div>
            
            {/* Collapsed view - Just icon (desktop only) */}
            <div className={`items-center ${expanded ? 'hidden' : 'hidden md:flex'}`}>
              <Shield className="h-6 w-6 text-primary" />
            </div>

            {/* Mobile close button */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setExpanded(false)}
              className="h-9 w-9 md:hidden md-button-text"
              aria-label="Close sidebar"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>
        </div>

        {/* Navigation Items */}
        <div className="flex-1 overflow-auto py-2">
          <nav className="flex-1 py-4 px-4">
            <div className="space-y-2">
              {navigationItems.map((item) => {
                const isActive = pathname === item.href
                const Icon = item.icon
                
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={`
                      flex items-center rounded-lg text-sm font-medium transition-colors
                      ${expanded ? 'space-x-3 px-3 py-2' : 'justify-center p-2 md:p-3'}
                      relative overflow-hidden
                      md-nav-item
                      ${isActive 
                        ? 'active elevation-1' 
                        : 'text-medium-emphasis hover:text-high-emphasis'
                      }
                    `}
                    title={!expanded ? item.name : undefined}
                  >
                    <Icon className="h-4 w-4 flex-shrink-0" />
                    {expanded && <span className="truncate">{item.name}</span>}
                    
                    {/* State overlay for hover effects */}
                    {!isActive && (
                      <div className="absolute inset-0 bg-primary opacity-0 transition-opacity duration-150 hover:opacity-[0.04] pointer-events-none" />
                    )}
                  </Link>
                )
              })}
            </div>
          </nav>
        </div>
      </div>
    </>
  )
}

