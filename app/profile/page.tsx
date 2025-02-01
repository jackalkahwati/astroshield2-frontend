"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { User, Mail, Settings } from "lucide-react"

export default function ProfilePage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Profile</h1>
        <Button variant="outline" size="sm">
          <Settings className="mr-2 h-4 w-4" />
          Edit Profile
        </Button>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center space-x-4">
            <Avatar className="h-16 w-16">
              <AvatarImage src="https://avatar.vercel.sh/astroshield.png" />
              <AvatarFallback>AS</AvatarFallback>
            </Avatar>
            <div>
              <CardTitle>Admin User</CardTitle>
              <CardDescription>System Administrator</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4">
            <div className="flex items-center space-x-4">
              <User className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium">Username</p>
                <p className="text-sm text-muted-foreground">admin</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Mail className="h-5 w-5 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium">Email</p>
                <p className="text-sm text-muted-foreground">admin@astroshield.com</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Your recent actions in the system</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="text-sm">
                <p className="font-medium">Updated system settings</p>
                <p className="text-muted-foreground">2 hours ago</p>
              </div>
              <div className="text-sm">
                <p className="font-medium">Reviewed satellite data</p>
                <p className="text-muted-foreground">5 hours ago</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>System Access</CardTitle>
            <CardDescription>Your system permissions and access levels</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="text-sm">
                <p className="font-medium">Role</p>
                <p className="text-muted-foreground">Administrator</p>
              </div>
              <div className="text-sm">
                <p className="font-medium">Access Level</p>
                <p className="text-muted-foreground">Full System Access</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 