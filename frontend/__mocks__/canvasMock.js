// Canvas mock for Jest
// This file mocks the 'canvas' module for Jest tests

// Mock canvas
class Canvas {
  constructor(width, height) {
    this.width = width || 300;
    this.height = height || 150;
    this.context = new CanvasRenderingContext2D();
  }

  getContext(contextType) {
    return this.context;
  }

  toDataURL() {
    return 'data:image/png;base64,mock-data';
  }

  toBuffer() {
    return Buffer.from('mock-buffer');
  }

  createPNGStream() {
    return {
      pipe: jest.fn(),
    };
  }

  createJPEGStream() {
    return {
      pipe: jest.fn(),
    };
  }
}

// Mock CanvasRenderingContext2D
class CanvasRenderingContext2D {
  constructor() {
    // Default properties
    this.fillStyle = '#000000';
    this.strokeStyle = '#000000';
    this.lineWidth = 1;
    this.font = '10px sans-serif';
    this.textAlign = 'start';
    this.textBaseline = 'alphabetic';
    this.globalAlpha = 1.0;
    
    // Define mock methods
    this.save = jest.fn();
    this.restore = jest.fn();
    this.scale = jest.fn();
    this.rotate = jest.fn();
    this.translate = jest.fn();
    this.transform = jest.fn();
    this.setTransform = jest.fn();
    this.resetTransform = jest.fn();
    this.createLinearGradient = jest.fn().mockReturnValue({
      addColorStop: jest.fn(),
    });
    this.createRadialGradient = jest.fn().mockReturnValue({
      addColorStop: jest.fn(),
    });
    this.createPattern = jest.fn().mockReturnValue({});
    this.clearRect = jest.fn();
    this.fillRect = jest.fn();
    this.strokeRect = jest.fn();
    this.beginPath = jest.fn();
    this.fill = jest.fn();
    this.stroke = jest.fn();
    this.clip = jest.fn();
    this.isPointInPath = jest.fn().mockReturnValue(false);
    this.isPointInStroke = jest.fn().mockReturnValue(false);
    this.fillText = jest.fn();
    this.strokeText = jest.fn();
    this.measureText = jest.fn().mockReturnValue({ width: 0 });
    this.drawImage = jest.fn();
    this.createImageData = jest.fn().mockReturnValue({
      data: new Uint8ClampedArray(4),
    });
    this.getImageData = jest.fn().mockReturnValue({
      data: new Uint8ClampedArray(4),
    });
    this.putImageData = jest.fn();
    this.closePath = jest.fn();
    this.moveTo = jest.fn();
    this.lineTo = jest.fn();
    this.quadraticCurveTo = jest.fn();
    this.bezierCurveTo = jest.fn();
    this.arc = jest.fn();
    this.arcTo = jest.fn();
    this.rect = jest.fn();
  }
}

// Mock Image
class Image {
  constructor() {
    this.src = '';
    this.width = 0;
    this.height = 0;
    this.onload = null;
    this.onerror = null;
  }
}

// Mock functions
const createCanvas = jest.fn().mockImplementation((width, height) => {
  return new Canvas(width, height);
});

const loadImage = jest.fn().mockImplementation((src) => {
  return Promise.resolve(new Image());
});

// Export mock methods
module.exports = {
  Canvas,
  createCanvas,
  Image,
  loadImage,
  __esModule: true,
}; 