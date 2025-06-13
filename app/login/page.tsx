"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Shield } from "lucide-react"

export default function LoginPage() {
  const router = useRouter()
  const [credentials, setCredentials] = useState({
    username: "",
    password: ""
  })

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault()
    // Simple authentication - in production this would be proper auth
    if (credentials.username && credentials.password) {
      router.push("/dashboard")
    } else {
      alert("Please enter both username and password")
    }
  }

  return (
    <div className="min-h-screen bg-[#0A0E1A] text-white flex items-center justify-center p-6">
      <div className="w-full max-w-md space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex justify-center">
            <Shield className="h-12 w-12 text-blue-400" />
          </div>
          <h1 className="text-3xl font-bold">AstroShield</h1>
          <p className="text-gray-400">Space Operations Center</p>
        </div>

        {/* Login Form */}
        <Card className="bg-[#1A1F2E] border-gray-800">
          <CardHeader>
            <CardTitle className="text-white text-center">Sign In</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="username" className="text-white">Username</Label>
                <Input
                  id="username"
                  type="text"
                  value={credentials.username}
                  onChange={(e) => setCredentials({ ...credentials, username: e.target.value })}
                  placeholder="Enter your username"
                  className="bg-[#0A0E1A] border-gray-700 text-white"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password" className="text-white">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={credentials.password}
                  onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
                  placeholder="Enter your password"
                  className="bg-[#0A0E1A] border-gray-700 text-white"
                />
              </div>
              <Button type="submit" className="w-full">
                Sign In
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Demo Credentials Info */}
        <Card className="bg-[#1A1F2E] border-gray-800">
          <CardHeader>
            <CardTitle className="text-white text-sm">Demo Access</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-400 text-sm">
              Enter any username and password to access the demo
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 