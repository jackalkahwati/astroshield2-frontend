module.exports = {
  testEnvironment: 'jsdom',
  roots: ['<rootDir>/frontend/src', '<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.js?(x)', '**/?(*.)+(spec|test).js?(x)'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/frontend/src/$1',
  },
  modulePathIgnorePatterns: [
    '<rootDir>/deployment/',
    '<rootDir>/local_deploy/',
    '<rootDir>/node_modules/',
  ],
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': 'babel-jest',
  },
  watchPathIgnorePatterns: [
    '<rootDir>/node_modules/',
    '<rootDir>/deployment/',
    '<rootDir>/local_deploy/',
  ],
  testPathIgnorePatterns: [
    '<rootDir>/node_modules/',
    '<rootDir>/deployment/',
    '<rootDir>/local_deploy/',
  ],
}; 