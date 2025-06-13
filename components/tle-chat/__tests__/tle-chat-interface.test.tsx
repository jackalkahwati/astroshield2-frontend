import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import TLEChatInterface from '../tle-chat-interface'

describe('TLEChatInterface', () => {
  it('renders the chat interface and allows user input', async () => {
    render(<TLEChatInterface showExamples={false} />)
    
    // Check for welcome message
    expect(screen.getByText(/Welcome to the TLE Orbit Analyzer/i)).toBeInTheDocument()
    
    // Type a message
    const textarea = screen.getByPlaceholderText(/Enter TLE data or ask a question/i)
    fireEvent.change(textarea, { target: { value: 'What is a TLE?' } })
    expect(textarea).toHaveValue('What is a TLE?')
    
    // Send the message
    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter', charCode: 13 })
    
    // Wait for the assistant response
    await waitFor(() => {
      expect(screen.getByText(/AI assistant specializing in orbital mechanics/i)).toBeInTheDocument()
    })
  })
}) 