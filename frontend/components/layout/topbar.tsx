"use client"

import React from "react"
import { ThemeToggle } from "@/components/theme-toggle"
import { Settings, User } from "lucide-react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { UserNav } from "@/components/user-nav"

export function TopBar() {
  return (
    <header className="flex h-14 items-center gap-4 border-b bg-background px-6">
      <div className="flex flex-1 items-center justify-end space-x-4">
        <div className="flex items-center space-x-2">
          <ThemeToggle />
          <Link href="/settings">
            <Button variant="ghost" size="icon">
              <Settings className="h-5 w-5" />
            </Button>
          </Link>
          <UserNav />
        </div>
      </div>
    </header>
  )
} 