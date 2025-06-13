"use client"

import { useState, useCallback } from "react"

export interface Toast {
  id?: string
  title?: string
  description?: string
  variant?: "default" | "destructive"
}

const toasts: Toast[] = []

export function useToast() {
  const [, setUpdateTrigger] = useState(0)

  const toast = useCallback((toast: Toast) => {
    const id = Math.random().toString(36).substr(2, 9)
    const newToast = { ...toast, id }
    toasts.push(newToast)
    setUpdateTrigger(prev => prev + 1)
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      const index = toasts.findIndex(t => t.id === id)
      if (index > -1) {
        toasts.splice(index, 1)
        setUpdateTrigger(prev => prev + 1)
      }
    }, 5000)
    
    return {
      id,
      dismiss: () => {
        const index = toasts.findIndex(t => t.id === id)
        if (index > -1) {
          toasts.splice(index, 1)
          setUpdateTrigger(prev => prev + 1)
        }
      }
    }
  }, [])

  return {
    toast,
    toasts,
    dismiss: (toastId: string) => {
      const index = toasts.findIndex(t => t.id === toastId)
      if (index > -1) {
        toasts.splice(index, 1)
        setUpdateTrigger(prev => prev + 1)
      }
    }
  }
}

export const toast = (toastData: Toast) => {
  // Simple console implementation for development
  console.log('Toast:', toastData)
}
