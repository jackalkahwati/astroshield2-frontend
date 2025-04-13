/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen } from '@testing-library/react';

// A simple React component that uses Canvas
const CanvasComponent = () => {
  const canvasRef = React.useRef(null);
  
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Draw something
    ctx.fillStyle = 'blue';
    ctx.fillRect(0, 0, 100, 100);
    
    // Draw some text
    ctx.fillStyle = 'white';
    ctx.font = '16px Arial';
    ctx.fillText('AstroShield', 10, 50);
    
  }, []);
  
  return (
    <div>
      <h2>Canvas Test</h2>
      <canvas 
        ref={canvasRef} 
        width={200} 
        height={200}
        data-testid="test-canvas"
      />
    </div>
  );
};

describe('Canvas Component', () => {
  it('renders the canvas element', () => {
    render(<CanvasComponent />);
    
    // Check if canvas is in the document
    const canvas = screen.getByTestId('test-canvas');
    expect(canvas).toBeInTheDocument();
    expect(canvas.tagName).toBe('CANVAS');
  });
  
  it('correctly calls canvas methods', () => {
    render(<CanvasComponent />);
    
    const canvas = screen.getByTestId('test-canvas');
    const ctx = canvas.getContext('2d');
    
    // Check our mock was called with correct arguments
    expect(ctx.fillRect).toHaveBeenCalledWith(0, 0, 100, 100);
    expect(ctx.fillText).toHaveBeenCalledWith('AstroShield', 10, 50);
    
    // Verify fillStyle was set correctly
    expect(ctx.fillStyle).toBe('white');
    expect(ctx.font).toBe('16px Arial');
  });
  
  it('has the correct dimensions', () => {
    render(<CanvasComponent />);
    
    const canvas = screen.getByTestId('test-canvas');
    expect(canvas.width).toBe(200);
    expect(canvas.height).toBe(200);
  });
}); 