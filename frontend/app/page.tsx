"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"

export default function HomePage() {
  const router = useRouter()

  useEffect(() => {
    router.push("/dashboard")
  }, [router])

  return (
    <div className="flex h-screen w-full items-center justify-center bg-background">
      <div className="flex flex-col items-center space-y-6">
        <div className="h-10 w-10 animate-spin rounded-full border-b-2 border-t-2 border-primary"></div>
        <div className="text-center">
          <p className="text-xl font-semibold text-foreground">Welcome to AstroShield v2.3</p>
          <p className="text-sm text-muted-foreground">Initializing next-generation satellite protection systems...</p>
          <p className="text-xs text-muted-foreground mt-2">Please wait while we connect to secure systems</p>
        </div>
      </div>
    </div>
  )
}

