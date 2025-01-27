/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['astroshield2-api-production.up.railway.app'],
  },
  webpack: (config, { isServer }) => {
    // Suppress punycode warning
    config.ignoreWarnings = [
      { module: /node_modules\/punycode/ }
    ]
    return config
  }
}

module.exports = nextConfig 