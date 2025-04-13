#!/usr/bin/env node

/**
 * Test runner that mocks canvas dependencies for CI environments
 * where native modules can't be installed
 */
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('Setting up canvas mock environment...');

// Create a temporary setup file for canvas mocking
const setupFilePath = path.join(__dirname, 'temp-canvas-mock.js');

// Write canvas mocking setup
fs.writeFileSync(setupFilePath, `
// Mock canvas implementation
jest.mock('canvas', () => {
  const Canvas = jest.fn().mockImplementation((width, height) => ({
    width,
    height,
    getContext: jest.fn().mockReturnValue({
      fillRect: jest.fn(),
      clearRect: jest.fn(),
      getImageData: jest.fn(() => ({
        data: new Uint8Array(4)
      })),
      putImageData: jest.fn(),
      drawImage: jest.fn(),
      save: jest.fn(),
      restore: jest.fn(),
      fillText: jest.fn(),
      measureText: jest.fn(() => ({ width: 10 })),
      scale: jest.fn(),
      rotate: jest.fn(),
      translate: jest.fn(),
      beginPath: jest.fn(),
      moveTo: jest.fn(),
      lineTo: jest.fn(),
      closePath: jest.fn(),
      stroke: jest.fn(),
      fill: jest.fn(),
      arc: jest.fn(),
      rect: jest.fn(),
      ellipse: jest.fn()
    })
  }));
  Canvas.createCanvas = jest.fn().mockImplementation((width, height) => 
    new Canvas(width, height)
  );
  Canvas.Image = jest.fn().mockImplementation(() => ({
    src: '',
    width: 0,
    height: 0,
    onload: jest.fn()
  }));
  return Canvas;
});

// Mock HTMLCanvasElement
if (typeof window !== 'undefined') {
  window.HTMLCanvasElement.prototype.getContext = jest.fn().mockReturnValue({
    fillRect: jest.fn(),
    clearRect: jest.fn(),
    getImageData: jest.fn(() => ({
      data: new Uint8Array(4)
    })),
    putImageData: jest.fn(),
    drawImage: jest.fn(),
    scale: jest.fn(),
    rotate: jest.fn(),
    translate: jest.fn(),
    save: jest.fn(),
    restore: jest.fn(),
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    closePath: jest.fn(),
    stroke: jest.fn(),
    fill: jest.fn()
  });
}

require('@testing-library/jest-dom');
console.log('Canvas mock environment initialized');
`);

console.log('Running tests with canvas mocking...');

// Run Jest with our canvas mock
const jestProcess = spawn('node', [
  './node_modules/.bin/jest',
  '--setupFilesAfterEnv',
  setupFilePath,
  'src/__tests__/canvas.test.js'
], { stdio: 'inherit' });

jestProcess.on('close', (code) => {
  // Clean up the temporary file
  try {
    fs.unlinkSync(setupFilePath);
    console.log('Cleaned up temporary mock file');
  } catch (err) {
    console.error('Failed to clean up temporary file:', err);
  }
  
  process.exit(code);
});

jestProcess.on('error', (err) => {
  console.error('Failed to start Jest process:', err);
  try {
    fs.unlinkSync(setupFilePath);
  } catch (e) {
    // Ignore cleanup errors
  }
  process.exit(1);
}); 