import { Inter } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import { SidebarProvider } from "@/components/providers/sidebar-provider"
import { Toaster } from "@/components/ui/toaster"
import { cn } from "@/lib/utils"
import { Layout } from "@/components/layout/Layout"

import "@/styles/globals.css"

const inter = Inter({ subsets: ["latin"] })

export const metadata = {
  title: "AstroShield - Advanced Satellite Protection System",
  description: "A comprehensive dashboard for real-time satellite monitoring, threat detection, and orbital maneuver planning. Featuring advanced analytics, protection metrics, and intelligent alert systems.",
  keywords: "satellite protection, orbital monitoring, space debris, collision avoidance, maneuver planning",
  authors: [{ name: "AstroShield Team" }],
  viewport: "width=device-width, initial-scale=1",
  icons: {
    icon: "/favicon.ico",
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body className={cn(inter.className, "min-h-screen bg-background font-sans antialiased")}>
        <ThemeProvider>
          <SidebarProvider>
            <Layout>
              {children}
            </Layout>
          </SidebarProvider>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  )
}