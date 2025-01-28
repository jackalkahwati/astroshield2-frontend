import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { Sidebar } from "@/components/layout/sidebar"
import { ClientWrapper } from "@/components/layout/client-wrapper"

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
          <div className="flex min-h-screen">
            <Sidebar className="w-64 fixed h-full" />
            <main className="flex-1 ml-64 p-6 overflow-auto">
              {children}
            </main>
          </div>
        </ClientWrapper>
      </body>
    </html>
  )
}