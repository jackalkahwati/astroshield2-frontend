# AstroShield Testing Guide

## Overview

The AstroShield frontend includes tests for React components, services, and canvas-based visualizations. This guide explains the different testing options and how to run tests in different environments.

## Testing Options

We have several testing scripts available, each designed for different scenarios:

| Script | Description |
|--------|-------------|
| `npm test` | Default Jest test runner (requires canvas dependencies) |
| `npm run test:all` | Comprehensive test suite that combines isolated canvas tests with other Jest tests |
| `npm run test:isolated-canvas` | Runs canvas tests with mock implementations (no native dependencies required) |
| `npm run test:nocanvas` | Runs Jest tests but excludes any files with canvas in the name |
| `npm run test:skip-canvas` | Uses Jest with an environment variable to skip canvas tests |
| `npm run test:safe` | Hardcoded list of tests known to be safe to run |

## Canvas Testing

Canvas testing is challenging because it requires native dependencies. There are two approaches:

### 1. Install Native Dependencies (recommended for development)

For full canvas testing, install the required system dependencies:

**macOS:**
```bash
brew install pkg-config cairo pango libpng jpeg giflib librsvg pixman
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev
```

**Red Hat/CentOS:**
```bash
sudo yum install -y gcc-c++ cairo-devel pango-devel libjpeg-turbo-devel giflib-devel
```

After installing these dependencies, you can run `npm install` to install the canvas package correctly.

### 2. Use Mock Implementation (for CI/CD or systems without dependencies)

If you can't install the native dependencies, use our mocked canvas implementation:

```bash
npm run test:isolated-canvas
```

This test runner fully mocks the canvas functionality without requiring native dependencies.

## CI/CD Setup

For continuous integration environments, we recommend using our test:all script:

```bash
npm run test:all
```

This will:
1. Run the isolated canvas tests with mocks
2. Attempt to run other Jest tests while excluding canvas tests
3. Provide a comprehensive report

## Adding New Tests

When adding new tests:

1. For components that use canvas, follow the pattern in `src/__tests__/canvas.test.js`
2. For tests that indirectly depend on canvas, consider adding them to the skipCanvasTests.js patterns

## Troubleshooting

If you encounter canvas-related errors:

1. Try running with the isolated canvas test: `npm run test:isolated-canvas`
2. Check if your system has the required dependencies installed
3. Review the error message for specific missing libraries

## Future Improvements

- Complete mock implementation for all canvas-dependent tests
- Better detection of canvas usage in test files
- Automated setup script for different operating systems 