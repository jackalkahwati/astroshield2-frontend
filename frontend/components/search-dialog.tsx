"use client"

import { useState, useRef, useEffect } from "react"
import { Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

// Mock searchable terms (in a real app, this would be dynamically generated or fetched)
const searchableTerms = [
  "Satellite",
  "Orbit",
  "Mission",
  "Launch",
  "Tracking",
  "Telemetry",
  "Debris",
  "Collision",
  "Maneuver",
  "Communication",
  "Solar Panel",
  "Propulsion",
  "Ground Station",
  "Payload",
  "Attitude Control",
]

export function SearchDialog() {
  const [isExpanded, setIsExpanded] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [filteredTerms, setFilteredTerms] = useState<string[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    // Implement search logic here
    console.log("Searching for:", searchQuery)
  }

  const toggleExpand = () => {
    setIsExpanded(!isExpanded)
    if (!isExpanded) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setSearchQuery(value)
    if (value) {
      const filtered = searchableTerms.filter((term) => term.toLowerCase().includes(value.toLowerCase()))
      setFilteredTerms(filtered)
    } else {
      setFilteredTerms([])
    }
  }

  const handleTermClick = (term: string) => {
    setSearchQuery(term)
    setFilteredTerms([])
    inputRef.current?.focus()
  }

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        isExpanded &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node) &&
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsExpanded(false)
        setFilteredTerms([])
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [isExpanded])

  return (
    <div className="relative">
      <form onSubmit={handleSearch} className="flex items-center">
        <div className={cn("overflow-hidden transition-all duration-300 ease-in-out", isExpanded ? "w-64" : "w-0")}>
          <Input
            ref={inputRef}
            type="text"
            placeholder="Search..."
            value={searchQuery}
            onChange={handleInputChange}
            className={cn("h-9 w-full rounded-l-md border border-r-0", !isExpanded && "opacity-0")}
          />
        </div>
        <Button
          type={isExpanded ? "submit" : "button"}
          variant="outline"
          size="icon"
          className={cn("h-9 w-9 rounded-md transition-all duration-300 ease-in-out", isExpanded && "rounded-l-none")}
          onClick={toggleExpand}
        >
          <Search className="h-4 w-4" />
        </Button>
      </form>
      {isExpanded && filteredTerms.length > 0 && (
        <div
          ref={dropdownRef}
          className="absolute z-10 mt-1 w-64 rounded-md bg-background/80 backdrop-blur-sm shadow-lg ring-1 ring-black ring-opacity-5 max-h-60 overflow-auto"
        >
          <ul className="py-1">
            {filteredTerms.map((term, index) => (
              <li
                key={index}
                className="px-4 py-2 text-sm text-foreground hover:bg-accent hover:text-accent-foreground cursor-pointer"
                onClick={() => handleTermClick(term)}
              >
                {term}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

