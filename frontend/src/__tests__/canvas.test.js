import React, { useRef, useEffect } from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

// A simple React component that uses canvas
const CanvasComponent = ({ width = 300, height = 150 }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background
    ctx.fillStyle = '#000033';
    ctx.fillRect(0, 0, width, height);

    // Draw a satellite orbit
    ctx.beginPath();
    ctx.strokeStyle = '#3399FF';
    ctx.lineWidth = 2;
    ctx.ellipse(width / 2, height / 2, width / 3, height / 4, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Draw a satellite
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(width / 2 + width / 3, height / 2, 5, 0, 2 * Math.PI);
    ctx.fill();

    // Add text
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '14px Arial';
    ctx.fillText('AstroShield Canvas Test', 10, 20);

  }, [width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      data-testid="test-canvas"
      style={{ border: '1px solid #000' }}
    />
  );
};

describe('Canvas Testing', () => {
  test('basic canvas manipulation works', () => {
    // Create a canvas element
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 150;
    document.body.appendChild(canvas);

    // Get the 2d context
    const ctx = canvas.getContext('2d');
    expect(ctx).toBeTruthy();

    // Basic drawing operations
    ctx.fillStyle = 'red';
    ctx.fillRect(10, 10, 100, 100);
    
    // Check pixel data
    const imageData = ctx.getImageData(15, 15, 1, 1);
    expect(imageData).toBeTruthy();
    
    // Clean up
    document.body.removeChild(canvas);
  });

  test('CanvasComponent renders correctly', () => {
    render(<CanvasComponent />);
    
    // Verify canvas element is present
    const canvasElement = screen.getByTestId('test-canvas');
    expect(canvasElement).toBeInTheDocument();
    
    // Verify canvas attributes
    expect(canvasElement).toHaveAttribute('width', '300');
    expect(canvasElement).toHaveAttribute('height', '150');
    expect(canvasElement).toHaveStyle('border: 1px solid #000');
  });
});

// This will only log in browser environments
if (typeof window !== 'undefined') {
  console.log('Canvas test file loaded');
} 