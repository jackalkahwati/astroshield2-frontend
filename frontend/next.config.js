/** @type {import('next').NextConfig} */
const webpack = require('webpack');

const nextConfig = {
  images: {
    domains: ['astroshield2-api-production.up.railway.app'],
  },
  webpack: (config, { isServer }) => {
    // Suppress punycode warning
    config.ignoreWarnings = [
      { module: /node_modules\/punycode/ },
      // Add warning for mapbox-gl
      { message: /Critical dependency: the request of a dependency is an expression/ }
    ]
    
    // Handle mapbox-gl issues
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        os: false,
        http: false,
        https: false,
        zlib: false,
        stream: false,
        crypto: false,
        buffer: require.resolve('buffer/'),
      };
      
      // Add buffer polyfill
      config.plugins.push(
        new webpack.ProvidePlugin({
          Buffer: ['buffer', 'Buffer'],
        })
      );
    }
    
    return config
  },
  async redirects() {
    return [
      {
        source: "/comprehensive",
        destination: "/dashboard",
        permanent: false,
      },
    ];
  },
  // Add environment variables
  env: {
    MAPBOX_TOKEN: 'pk.eyJ1IjoiaXExOXplcm8xMiIsImEiOiJjajNveDZkNWMwMGtpMnFuNG05MjNidjBrIn0.rbEk-JO7ewQXACGoTCT5CQ'
  },
  // Add API proxy configuration
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5002/api/:path*',
      },
    ];
  },
}

module.exports = nextConfig 