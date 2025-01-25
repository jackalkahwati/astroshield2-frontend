import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/components/providers/theme-provider"
import { SidebarProvider } from "@/components/providers/sidebar-provider"
import Layout from "@/components/layout/Layout"

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
      <head />
      <body className={inter.className}>
        <ThemeProvider>
          <SidebarProvider>
            <Layout>{children}</Layout>
          </SidebarProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}