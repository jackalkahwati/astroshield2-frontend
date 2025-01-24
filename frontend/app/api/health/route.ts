import { NextResponse } from 'next/server'

export async function GET() {
  try {
    // In a real app, we would check the backend health here
    return NextResponse.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services: {
        database: 'connected',
        api: 'operational',
        telemetry: 'active'
      }
    })
  } catch (error) {
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    )
  }
} 