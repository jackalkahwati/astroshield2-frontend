#!/usr/bin/env node

/**
 * Test runner that completely mocks the canvas module
 */
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Create a custom jest config file
const configPath = path.join(__dirname, 'jest.mock.config.js');

// Write a Jest config that mocks the canvas module
fs.writeFileSync(configPath, `
module.exports = {
  testEnvironment: 'jsdom',
  verbose: true,
  testMatch: ['<rootDir>/src/__tests__/canvas.test.js'],
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': 'babel-jest',
  },
  moduleNameMapper: {
    '^canvas$': '<rootDir>/__mocks__/canvasMock.js',
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
};
`);

// Create the mocks directory if it doesn't exist
const mocksDir = path.join(__dirname, '__mocks__');
if (!fs.existsSync(mocksDir)) {
  fs.mkdirSync(mocksDir);
}

// Create a mock for the canvas module
const canvasMockPath = path.join(mocksDir, 'canvasMock.js');
fs.writeFileSync(canvasMockPath, `
// Mock implementation of canvas
const createCanvas = (width, height) => {
  return {
    width,
    height,
    getContext: () => ({
      fillRect: jest.fn(),
      clearRect: jest.fn(),
      getImageData: jest.fn(() => ({
        data: new Uint8Array(4),
      })),
      putImageData: jest.fn(),
      createImageData: jest.fn(() => ({
        data: new Uint8Array(4),
      })),
      drawImage: jest.fn(),
      fillText: jest.fn(),
      measureText: jest.fn(() => ({ width: 10 })),
      stroke: jest.fn(),
      beginPath: jest.fn(),
      moveTo: jest.fn(),
      lineTo: jest.fn(),
      arc: jest.fn(),
      fill: jest.fn(),
      ellipse: jest.fn(),
      rect: jest.fn(),
      save: jest.fn(),
      restore: jest.fn(),
      scale: jest.fn(),
      rotate: jest.fn(),
      translate: jest.fn(),
      strokeText: jest.fn(),
      setTransform: jest.fn(),
      bezierCurveTo: jest.fn(),
      quadraticCurveTo: jest.fn(),
      closePath: jest.fn(),
      clip: jest.fn(),
      createLinearGradient: jest.fn(() => ({
        addColorStop: jest.fn(),
      })),
      createRadialGradient: jest.fn(() => ({
        addColorStop: jest.fn(),
      })),
      createPattern: jest.fn(() => ({})),
      fillStyle: '',
      strokeStyle: '',
      lineWidth: 1,
    }),
    toBuffer: jest.fn(() => Buffer.from([0, 0, 0, 0])),
    toDataURL: jest.fn(() => ''),
  };
};

// Create a mock Image class
class Image {
  constructor() {
    this.src = '';
    this.width = 0;
    this.height = 0;
    this.onload = jest.fn();
    this.onerror = jest.fn();
  }
}

// Mock the Canvas class and its methods
module.exports = {
  createCanvas,
  Canvas: createCanvas,
  Image,
  loadImage: jest.fn(() => Promise.resolve(new Image())),
  registerFont: jest.fn(),
};
`);

// Modify the test setup file for canvas tests
// Create a mock implementation for HTMLCanvasElement
const setupPath = path.join(__dirname, 'jest.setup.canvas.js');
fs.writeFileSync(setupPath, `
// Import the original setup
require('./jest.setup.js');

// Mock the HTMLCanvasElement
Object.defineProperty(window, 'HTMLCanvasElement', {
  value: class HTMLCanvasElement {
    constructor() {
      this.width = 0;
      this.height = 0;
    }
    
    getContext() {
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
        transform: jest.fn(),
        beginPath: jest.fn(),
        moveTo: jest.fn(),
        lineTo: jest.fn(),
        arc: jest.fn(),
        fill: jest.fn(),
        stroke: jest.fn(),
        measureText: jest.fn(() => ({ width: 0 })),
        strokeText: jest.fn(),
        fillText: jest.fn(),
      };
    }
    
    toDataURL() {
      return '';
    }
  },
  writable: false,
  configurable: true,
});

// Mock requestAnimationFrame
global.requestAnimationFrame = (callback) => setTimeout(callback, 0);
global.cancelAnimationFrame = (id) => clearTimeout(id);

// Add other canvas-related mocks if needed
`);

console.log('Running canvas tests with mocked implementation...');

// Run Jest with the custom config
const jestBin = path.join(__dirname, 'node_modules', '.bin', 'jest');
const jestProcess = spawn('node', [
  jestBin,
  '--config',
  configPath,
  '--setupFilesAfterEnv',
  setupPath,
  '--no-cache',
  '--runInBand' // Run serially to avoid potential issues
], { stdio: 'inherit' });

// Clean up when done
jestProcess.on('close', (code) => {
  try {
    fs.unlinkSync(configPath);
    fs.unlinkSync(canvasMockPath);
    fs.unlinkSync(setupPath);
    
    // Try to remove mocks directory if empty
    if (fs.readdirSync(mocksDir).length === 0) {
      fs.rmdirSync(mocksDir);
    }
    
    console.log('Cleaned up temporary files');
  } catch (err) {
    console.error('Error during cleanup:', err.message);
  }
  
  process.exit(code);
}); 