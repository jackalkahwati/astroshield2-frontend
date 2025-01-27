"use client"

import { ReactNode } from 'react'
import { Navbar } from './navbar'
import { Sidebar } from './sidebar'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-background">
      <Navbar className="border-b border-border/40" />
      <div className="flex h-[calc(100vh-4rem)]">
        <Sidebar className="border-r border-border/40" />
        <main className="flex-1 overflow-y-auto p-6 lg:p-8">
          <div className="mx-auto max-w-[1600px] space-y-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
} 