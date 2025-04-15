"use client";

import React from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Rocket, Shield, Satellite, BarChart3, Navigation, Settings, Gauge } from 'lucide-react';
import Link from 'next/link';

export default function Home() {
  // Feature cards for the dashboard
  const features = [
    {
      title: "Satellite Tracking",
      description: "Monitor the position and status of satellites in real-time",
      icon: Satellite,
      href: "/tracking"
    },
    {
      title: "Stability Analysis",
      description: "Assess satellite stability and predict potential issues",
      icon: Gauge,
      href: "/stability"
    },
    {
      title: "Trajectory Analysis",
      description: "Analyze spacecraft trajectories and predict re-entry paths",
      icon: Navigation,
      href: "/trajectory"
    },
    {
      title: "Maneuvers",
      description: "Plan and execute satellite maneuvers for collision avoidance",
      icon: Rocket,
      href: "/maneuvers"
    },
    {
      title: "Analytics",
      description: "Access comprehensive analytics and reports on satellite fleet",
      icon: BarChart3,
      href: "/analytics"
    },
    {
      title: "Settings",
      description: "Configure system preferences and user settings",
      icon: Settings,
      href: "/settings"
    }
  ];

  return (
    <div className="container mx-auto p-6">
      <div className="mb-8 text-center">
        <div className="flex justify-center mb-4">
          <Shield className="h-16 w-16 text-primary" />
        </div>
        <h1 className="text-4xl font-bold mb-2">AstroShield Dashboard</h1>
        <p className="text-muted-foreground max-w-xl mx-auto">
          Comprehensive satellite protection system for monitoring, analyzing, and safeguarding your space assets
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-8">
        {features.map((feature, index) => (
          <Card key={index} className="overflow-hidden transition-all hover:shadow-lg">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <feature.icon className="h-5 w-5 text-primary" />
                <CardTitle>{feature.title}</CardTitle>
              </div>
              <CardDescription>{feature.description}</CardDescription>
            </CardHeader>
            <CardFooter className="pt-3 flex justify-end">
              <Link href={feature.href} passHref>
                <Button variant="ghost" size="sm">Access Feature</Button>
              </Link>
            </CardFooter>
          </Card>
        ))}
      </div>

      <Card className="mt-10">
        <CardHeader>
          <CardTitle>System Status</CardTitle>
          <CardDescription>Current status of AstroShield systems and components</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="font-medium">Backend API</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Operational</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="font-medium">Data Processing</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Operational</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="font-medium">Trajectory Analysis</span>
              <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs">Partial</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="font-medium">UDL Integration</span>
              <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs">Limited</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 