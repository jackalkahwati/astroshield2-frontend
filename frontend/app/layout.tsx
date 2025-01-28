import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { Sidebar } from "@/components/layout/sidebar"
import { TopBar } from "@/components/layout/topbar"
import { ClientWrapper } from "@/components/layout/client-wrapper"
import { SidebarProvider } from "@/components/providers/sidebar-provider"

const inter = Inter({ subsets: ["latin"] })

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
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ClientWrapper>
          <SidebarProvider>
            <div className="flex min-h-screen flex-col">
              <TopBar />
              <div className="flex flex-1">
                <Sidebar className="w-64 fixed h-[calc(100vh-3.5rem)]" />
                <main className="flex-1 ml-64 p-6 overflow-auto mt-14 transition-all duration-300">
                  {children}
                </main>
              </div>
            </div>
          </SidebarProvider>
        </ClientWrapper>
      </body>
    </html>
  )
}