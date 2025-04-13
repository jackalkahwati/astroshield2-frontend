// Setup canvas mocking for Jest tests
require('@testing-library/jest-dom');

// Add jest-canvas-mock
require('jest-canvas-mock');

// Ensure canvas methods are properly mocked
if (typeof window !== 'undefined') {
  // Mock createObjectURL and revokeObjectURL
  window.URL.createObjectURL = jest.fn(() => 'mock-object-url');
  window.URL.revokeObjectURL = jest.fn();
  
  // Additional canvas-related mocks
  window.HTMLCanvasElement.prototype.getContext = jest.fn(() => ({
    fillRect: jest.fn(),
    clearRect: jest.fn(),
    getImageData: jest.fn(() => ({
      data: new Array(4),
    })),
    putImageData: jest.fn(),
    createImageData: jest.fn(() => []),
    setTransform: jest.fn(),
    drawImage: jest.fn(),
    save: jest.fn(),
    restore: jest.fn(),
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    closePath: jest.fn(),
    stroke: jest.fn(),
    translate: jest.fn(),
    scale: jest.fn(),
    rotate: jest.fn(),
    arc: jest.fn(),
    fill: jest.fn(),
    measureText: jest.fn(() => ({
      width: 0,
      height: 0,
    })),
    transform: jest.fn(),
    rect: jest.fn(),
    clip: jest.fn(),
  }));
  
  window.HTMLCanvasElement.prototype.toDataURL = jest.fn(() => 'data:image/png;base64,mock');
  window.HTMLCanvasElement.prototype.toBlob = jest.fn((callback) => callback(new Blob(['mock'])));
} 