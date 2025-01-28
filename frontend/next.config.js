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
}

module.exports = nextConfig 