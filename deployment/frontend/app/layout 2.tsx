import type { Metadata } from "next"
import "./globals.css"
import { SidebarProvider } from "@/components/providers/sidebar-provider"
import { Sidebar } from "@/components/layout/sidebar"
import { TopBar } from "@/components/layout/top-bar"
import { Toaster } from "@/components/ui/toaster"

export const metadata: Metadata = {
  title: "AstroShield",
  description: "Space Situational Awareness Platform",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </head>
      <body className="min-h-screen font-sans antialiased bg-background text-foreground">
        <SidebarProvider>
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <div className="flex flex-col flex-1 overflow-hidden">
              <TopBar />
              <main className="flex-1 overflow-y-auto">
                <div className="container mx-auto p-6">
                  {children}
                </div>
              </main>
            </div>
          </div>
        </SidebarProvider>
        <Toaster />
      </body>
    </html>
  )
}