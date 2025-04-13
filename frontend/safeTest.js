/**
 * Safe test runner that skips tests requiring canvas.
 */
const jest = require('jest');
const path = require('path');
const fs = require('fs');
const skipCanvasTests = require('./skipCanvasTests');

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

// Mock document/window if needed
global.document = global.document || {
  createElement: jest.fn().mockImplementation(tag => {
    if (tag.toLowerCase() === 'canvas') {
      return {
        width: 0,
        height: 0,
        getContext: jest.fn().mockReturnValue({
          fillRect: jest.fn(),
          clearRect: jest.fn(),
          getImageData: jest.fn(() => ({ data: new Uint8Array(4) })),
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
        })
      };
    }
    return { 
      style: {},
      setAttribute: jest.fn() 
    };
  })
};`);

// Get all test files from the standard locations
const findTestFiles = () => {
  // Include both regular test files and canvas test file
  return [
    'src/services/__tests__/satelliteService.test.js',
    'src/services/__tests__/authService.test.js',
    'src/components/__tests__/Button.test.js',
    'src/utils/__tests__/formatters.test.js',
    'src/__tests__/canvas.test.js', // Add canvas test
    // Add more test files here
  ];
};

// Configuration for Jest
const jestConfig = {
  testMatch: findTestFiles().map(file => path.resolve(__dirname, file)),
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: [setupFilePath],
  verbose: true
};

// Run Jest with our configuration
jest.run(['--config', JSON.stringify(jestConfig)]);

console.log('Using mocked canvas environment for tests.');

// Clean up the temporary file when done
process.on('exit', () => {
  try {
    fs.unlinkSync(setupFilePath);
  } catch (err) {
    // Ignore errors during cleanup
  }
}); 