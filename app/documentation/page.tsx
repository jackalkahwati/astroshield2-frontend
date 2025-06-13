import type { Metadata } from "next"
import { DocumentationContent } from "@/components/documentation/content"

export const metadata: Metadata = {
  title: "Documentation | AstroShield",
  description: "User guides and documentation",
}

export default function DocumentationPage() {
  return (
    <div className="flex-1 space-y-6">
      <h2 className="text-3xl font-bold tracking-tight">Documentation</h2>
      <DocumentationContent />
    </div>
  )
}

