import React, { useEffect } from 'react'
import { useTheme } from 'next-themes'

export default function Home() {
  const { theme, setTheme } = useTheme()

  useEffect(() => {
    // Set initial theme
    setTheme('dark')
  }, [])

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <h1 className="text-4xl font-bold">AstroShield Dashboard</h1>
        <p className="mt-4">Welcome to the AstroShield monitoring system.</p>
      </div>
    </main>
  )
} 