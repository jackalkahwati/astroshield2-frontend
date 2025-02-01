# AstroShield Frontend

This is the frontend application for AstroShield, a comprehensive satellite monitoring and analysis platform.

## Features

- Real-time satellite monitoring
- ML-powered indicators
- Rule-based analysis
- Threshold monitoring
- Interactive dashboards
- Comprehensive satellite data visualization

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Build for production:
```bash
npm run build
```

4. Start production server:
```bash
npm start
```

## Environment Variables

Create a `.env.local` file with:

```env
NEXT_PUBLIC_API_URL=https://nosy-boy-production.up.railway.app/api/v1
```

## Tech Stack

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Radix UI
- Chart.js
- Axios

## API Integration

The frontend integrates with the AstroShield API for:
- UDL data fetching
- Indicators analysis
- Maneuver tracking
- System health monitoring
- Advanced satellite analysis 