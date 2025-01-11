interface Config {
  apiUrl: string;
  environment: string;
  sentry: {
    dsn: string | undefined;
    enabled: boolean;
    release: string | undefined;
  };
  analytics: {
    enabled: boolean;
    gaTrackingId: string | undefined;
  };
  theme: {
    defaultTheme: string;
  };
}

const config: Config = {
  apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  environment: process.env.NEXT_PUBLIC_ENVIRONMENT || 'development',
  sentry: {
    dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
    enabled: process.env.NEXT_PUBLIC_ENABLE_SENTRY === 'true',
    release: process.env.NEXT_PUBLIC_RELEASE,
  },
  analytics: {
    enabled: process.env.NEXT_PUBLIC_ENABLE_ANALYTICS === 'true',
    gaTrackingId: process.env.NEXT_PUBLIC_GA_TRACKING_ID,
  },
  theme: {
    defaultTheme: process.env.NEXT_PUBLIC_DEFAULT_THEME || 'system',
  },
};

export default config; 