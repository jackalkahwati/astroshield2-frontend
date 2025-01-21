const config = {
  apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  environment: process.env.NEXT_PUBLIC_ENVIRONMENT || 'development',
  enableSentry: process.env.NEXT_PUBLIC_ENABLE_SENTRY === 'true',
  enableAnalytics: process.env.NEXT_PUBLIC_ENABLE_ANALYTICS === 'true',
  defaultTheme: process.env.NEXT_PUBLIC_DEFAULT_THEME || 'system'
};

export default config;
