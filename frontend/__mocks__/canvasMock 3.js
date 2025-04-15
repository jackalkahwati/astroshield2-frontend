
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
