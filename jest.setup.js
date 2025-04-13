require('@testing-library/jest-dom');

// Mock canvas elements and context
if (typeof window !== 'undefined') {
  // Mock canvas
  if (!window.HTMLCanvasElement.prototype.getContext) {
    window.HTMLCanvasElement.prototype.getContext = function() {
      return {
        fillRect: jest.fn(),
        clearRect: jest.fn(),
        getImageData: jest.fn(() => ({
          data: new Uint8Array(4),
        })),
        putImageData: jest.fn(),
        drawImage: jest.fn(),
        save: jest.fn(),
        restore: jest.fn(),
        scale: jest.fn(),
        rotate: jest.fn(),
        translate: jest.fn(),
        fillText: jest.fn(),
        strokeText: jest.fn(),
        measureText: jest.fn(() => ({ width: 0 })),
        beginPath: jest.fn(),
        moveTo: jest.fn(),
        lineTo: jest.fn(),
        arc: jest.fn(),
        ellipse: jest.fn(),
        rect: jest.fn(),
        stroke: jest.fn(),
        fill: jest.fn(),
      };
    };
  }
}

// Mock requestAnimationFrame and cancelAnimationFrame
global.requestAnimationFrame = (callback) => setTimeout(callback, 0);
global.cancelAnimationFrame = (id) => clearTimeout(id);

// Silence console during tests (optional)
// Uncomment these lines if you want to suppress console output during tests
/*
global.console = {
  ...console,
  log: jest.fn(),
  error: jest.fn(),
  warn: jest.fn(),
  info: jest.fn(),
  debug: jest.fn(),
};
*/ 