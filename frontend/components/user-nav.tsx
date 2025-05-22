"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { User, LogOut, Settings, UserIcon } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useToast } from "@/components/ui/use-toast"

export function UserNav() {
  const router = useRouter()
  const { toast } = useToast()

  const handleLogout = () => {
    try {
      // Clear authentication tokens
      localStorage.removeItem('astroshield_token')
      localStorage.removeItem('astroshield_token_type')
      localStorage.removeItem('token') // Legacy token key
      
      // Clear any other user data
      localStorage.removeItem('user')
      localStorage.removeItem('user_preferences')
      
      toast({
        title: "Logged Out",
        description: "You have been successfully logged out.",
      })
      
      // Redirect to login page
      router.push('/login')
    } catch (error) {
      console.error('Error during logout:', error)
      toast({
        title: "Logout Error", 
        description: "There was an issue logging out. Please try again.",
        variant: "destructive",
      })
    }
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="relative">
          <User className="h-5 w-5" />
          <span className="sr-only">User menu</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuItem asChild>
          <Link href="/profile" className="w-full cursor-pointer flex items-center">
            <UserIcon className="mr-2 h-4 w-4" />
            Profile
          </Link>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <Link href="/settings" className="w-full cursor-pointer flex items-center">
            <Settings className="mr-2 h-4 w-4" />
            Settings
          </Link>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <Link href="/login" className="w-full cursor-pointer flex items-center">
            <UserIcon className="mr-2 h-4 w-4" />
            Login
          </Link>
        </DropdownMenuItem>
        <DropdownMenuItem>
          <button 
            className="w-full text-left flex items-center"
            onClick={handleLogout}
          >
            <LogOut className="mr-2 h-4 w-4" />
            Logout
          </button>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

