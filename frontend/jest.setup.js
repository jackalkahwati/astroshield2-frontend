// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
require('@testing-library/jest-dom');

// Mock Next.js router
jest.mock('next/router', () => require('next-router-mock'));

// Mock axios
jest.mock('axios');

// Suppress React 18 console errors and warnings related to act()
// This is because React Testing Library hasn't fully adapted to React 18
const originalConsoleError = console.error;
console.error = (...args) => {
  if (
    typeof args[0] === 'string' &&
    args[0].includes('Warning: ReactDOM.render is no longer supported') ||
    args[0].includes('act(...)')
  ) {
    return;
  }
  originalConsoleError(...args);
};

// Set up global fetch mock
global.fetch = jest.fn(() =>
  Promise.resolve({
    json: () => Promise.resolve({}),
    ok: true,
    status: 200,
    statusText: 'OK',
  })
);

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // Deprecated
    removeListener: jest.fn(), // Deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Setup testing environment for Jest
require('@testing-library/jest-dom');

// Mock the entire canvas module at the beginning
jest.mock('canvas', () => ({
  createCanvas: jest.fn(),
  loadImage: jest.fn()
}), { virtual: true });

// Mock canvas-related functions and objects globally
try {
  if (typeof global.HTMLCanvasElement !== 'undefined') {
    global.HTMLCanvasElement.prototype.getContext = jest.fn();
  }
} catch (error) {
  console.warn('Could not mock HTMLCanvasElement:', error.message);
}

// Add any other global mocks here as needed
if (typeof window !== 'undefined') {
  // Mock URL methods
  try {
    window.URL.createObjectURL = jest.fn();
    window.URL.revokeObjectURL = jest.fn();
  } catch (error) {
    console.warn('Could not mock URL methods:', error.message);
  }
} 