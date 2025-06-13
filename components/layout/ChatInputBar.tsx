"use client"

import React, { KeyboardEvent } from 'react'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { 
  Send, 
  Download, 
  Trash2, 
  Copy,
  Loader2,
  Zap,
  Sparkles
} from "lucide-react"
import { cn } from "@/lib/utils"

interface ChatInputBarProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onExport?: () => void
  onClear?: () => void
  onCopy?: () => void
  isLoading?: boolean
  placeholder?: string
  selectedModel?: string
  modelDescription?: string
  disabled?: boolean
  className?: string
  showActions?: {
    export?: boolean
    clear?: boolean
    copy?: boolean
  }
}

export function ChatInputBar({
  value,
  onChange,
  onSend,
  onExport,
  onClear,
  onCopy,
  isLoading = false,
  placeholder = "Type your message...",
  selectedModel,
  modelDescription,
  disabled = false,
  className,
  showActions = { export: true, clear: true, copy: true }
}: ChatInputBarProps) {
  
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter') {
      if (e.shiftKey) {
        // Allow new line on Shift+Enter
        return
      } else {
        // Send on Enter
        e.preventDefault()
        if (!isLoading && value.trim()) {
          onSend()
        }
      }
    }
  }

  const canSend = value.trim() && !isLoading && !disabled

  return (
    <footer className={cn("chat-input-wrapper", className)}>
      {/* Model indicator */}
      {selectedModel && (
        <div className="flex items-center gap-2 mb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <Badge variant="outline" className="text-xs font-medium">
              {selectedModel}
            </Badge>
            {isLoading && (
              <Badge variant="secondary" className="text-xs">
                <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                Processing...
              </Badge>
            )}
          </div>
          {modelDescription && (
            <span className="text-xs text-muted hidden md:block truncate">
              {modelDescription}
            </span>
          )}
        </div>
      )}

      {/* Input and actions */}
      <div className="chat-input-form">
        {/* Main input area */}
        <div className="flex-1 relative">
          <Textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            className={cn(
              "chat-textarea min-h-[44px] max-h-[120px] pr-12",
              "border-border-subtle bg-surface-2 text-text-primary",
              "placeholder:text-text-muted resize-none",
              "focus:border-primary focus:ring-1 focus:ring-primary/20"
            )}
            rows={1}
          />
          
          {/* Send button inside textarea */}
          <Button
            onClick={onSend}
            disabled={!canSend}
            size="sm"
            className={cn(
              "absolute right-2 bottom-2 h-8 w-8 p-0",
              "btn-primary shadow-sm"
            )}
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Action buttons */}
        <div className="flex gap-1">
          {showActions.copy && onCopy && (
            <Button
              onClick={onCopy}
              variant="ghost"
              size="sm"
              className="btn-ghost h-10 w-10 p-0"
              title="Copy last response"
            >
              <Copy className="h-4 w-4" />
            </Button>
          )}
          
          {showActions.export && onExport && (
            <Button
              onClick={onExport}
              variant="ghost"
              size="sm"
              className="btn-ghost h-10 w-10 p-0"
              title="Export conversation"
            >
              <Download className="h-4 w-4" />
            </Button>
          )}
          
          {showActions.clear && onClear && (
            <Button
              onClick={onClear}
              variant="ghost"
              size="sm"
              className="btn-ghost h-10 w-10 p-0"
              title="Clear conversation"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Helper text */}
      <div className="flex justify-between items-center mt-2 text-xs text-text-muted">
        <span>Press Enter to send, Shift+Enter for new line</span>
        <span className="hidden sm:block">
          {value.length > 0 && `${value.length} characters`}
        </span>
      </div>
    </footer>
  )
}

export default ChatInputBar 