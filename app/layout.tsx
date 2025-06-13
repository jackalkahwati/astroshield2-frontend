import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { ThemeProvider } from '@/components/providers/theme-provider'
import { SidebarProvider } from '@/components/providers/sidebar-provider'
import { Sidebar } from '@/components/layout/sidebar'
import { TopBar } from '@/components/layout/top-bar'
import { Toaster } from '@/components/ui/toaster'

const inter = Inter({ 
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter'
})

export const metadata: Metadata = {
  title: 'AstroShield',
  description: 'Space Situational Awareness Platform',
  viewport: 'width=device-width, initial-scale=1.0',
  themeColor: '#121212', // Material Design dark theme base
}

/**
 * Root Layout - Material Design Dark Theme
 * 
 * Features:
 * - Material Design #121212 base surface color
 * - Proper elevation hierarchy
 * - Material Design typography scale
 * - Accessibility-compliant color contrasts
 * - Responsive design breakpoints
 */
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning className={inter.variable}>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </head>
      <body className="min-h-screen font-sans antialiased elevation-0 text-high-emphasis">
        <ThemeProvider defaultTheme="dark" storageKey="astroshield-ui-theme">
          <SidebarProvider>
            {/* Material Design Layout Structure */}
            <div className="flex h-screen overflow-hidden">
              {/* Navigation Drawer - 2dp elevation */}
              <Sidebar />
              
              {/* Main Content Area */}
              <div className="flex flex-col flex-1 overflow-hidden">
                {/* Top App Bar - 8dp elevation */}
                <TopBar />
                
                {/* Primary Content - Base elevation */}
                <main className="flex-1 overflow-y-auto elevation-0">
                  <div className="container mx-auto p-6">
                    {children}
                  </div>
                </main>
              </div>
            </div>
          </SidebarProvider>
          
          {/* Toast Notifications */}
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  )
}