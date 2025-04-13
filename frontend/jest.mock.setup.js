// Mock localStorage
class LocalStorageMock {
  constructor() {
    this.store = {};
  }

  clear() {
    this.store = {};
  }

  getItem(key) {
    return this.store[key] || null;
  }

  setItem(key, value) {
    this.store[key] = String(value);
  }

  removeItem(key) {
    delete this.store[key];
  }

  get length() {
    return Object.keys(this.store).length;
  }

  key(index) {
    return Object.keys(this.store)[index] || null;
  }
}

// Set up global localStorage mock
global.localStorage = new LocalStorageMock();

// Mock window.location
delete window.location;
window.location = {
  href: '',
  search: '',
  pathname: '/',
  reload: jest.fn()
};

// Mock console methods to suppress specific errors
const originalConsoleError = console.error;
console.error = (...args) => {
  // Suppress specific errors that might occur during testing
  if (
    typeof args[0] === 'string' && (
      args[0].includes('Warning: ReactDOM.render is no longer supported') ||
      args[0].includes('Warning: An update to') ||
      args[0].includes('act(...)') ||
      args[0].includes('Error fetching threat assessment:')
    )
  ) {
    return;
  }
  originalConsoleError(...args);
};

// Setup axios mock
jest.mock('axios', () => {
  const mockAxios = {
    create: jest.fn(() => mockAxios),
    get: jest.fn(() => Promise.resolve({ data: {} })),
    post: jest.fn(() => Promise.resolve({ data: {} })),
    put: jest.fn(() => Promise.resolve({ data: {} })),
    delete: jest.fn(() => Promise.resolve({ data: {} })),
    interceptors: {
      request: { use: jest.fn(), eject: jest.fn() },
      response: { use: jest.fn(), eject: jest.fn() }
    },
    defaults: {
      headers: {
        common: {},
        post: {},
        get: {}
      }
    }
  };
  return mockAxios;
});

// Add any other global mocks here as needed 