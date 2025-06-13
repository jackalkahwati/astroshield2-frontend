'use client'

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { List, BarChart3 } from 'lucide-react'
import { cn } from '@/lib/utils'

export type ViewMode = 'list' | 'graph'

interface ViewToggleState {
  viewMode: ViewMode
  setViewMode: (mode: ViewMode) => void
  isListView: boolean
  isGraphView: boolean
}

// Custom hook for managing view toggle state
export function useViewToggle(defaultMode: ViewMode = 'graph'): ViewToggleState {
  const [viewMode, setViewMode] = useState<ViewMode>(defaultMode)

  const isListView = viewMode === 'list'
  const isGraphView = viewMode === 'graph'

  return {
    viewMode,
    setViewMode,
    isListView,
    isGraphView
  }
}

interface ViewToggleProps {
  currentView: ViewMode
  onViewChange: (mode: ViewMode) => void
  className?: string
}

// ViewToggle component for switching between list and graph views
export function ViewToggle({ currentView, onViewChange, className }: ViewToggleProps) {
  return (
    <div className={cn("flex items-center rounded-lg border p-1", className)}>
      <Button
        variant={currentView === 'list' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => onViewChange('list')}
        className="h-8 px-3"
      >
        <List className="h-4 w-4 mr-1" />
        List
      </Button>
      <Button
        variant={currentView === 'graph' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => onViewChange('graph')}
        className="h-8 px-3"
      >
        <BarChart3 className="h-4 w-4 mr-1" />
        Graph
      </Button>
    </div>
  )
}

export default ViewToggle 