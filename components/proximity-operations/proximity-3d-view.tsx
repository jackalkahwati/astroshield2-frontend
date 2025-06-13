'use client'

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function Proximity3DView() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>3D Proximity View</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-96 flex items-center justify-center bg-muted rounded">
          <p className="text-muted-foreground">3D Proximity Visualization - Coming Soon</p>
        </div>
      </CardContent>
    </Card>
  )
} 