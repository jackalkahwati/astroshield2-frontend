/**
 * This script is used to skip tests that require the canvas library.
 * It's necessary for environments where canvas native dependencies can't be installed.
 */
module.exports = {
  // Function to check if a test should be skipped based on file path
  skipTest: (testFile) => {
    // Define patterns that indicate canvas usage
    const canvasPatterns = [
      'canvas',
      'Chart',
      'visualization',
      'draw',
      'render',
      'graph'
    ];

    // Skip the test if the file contains any of the patterns
    return canvasPatterns.some(pattern => testFile.includes(pattern));
  }
}; 