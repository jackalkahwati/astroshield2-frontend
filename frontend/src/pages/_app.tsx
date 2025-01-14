import React from 'react';
import type { AppProps } from 'next/app';
import Head from 'next/head';
import { ThemeProvider as MUIThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { ThemeProvider } from '../components/theme-provider';
import { useTheme } from 'next-themes';

const App = ({ Component, pageProps }: AppProps) => {
  const { theme: nextTheme } = useTheme();
  
  const muiTheme = React.useMemo(() => 
    createTheme({
      palette: {
        mode: nextTheme === 'dark' ? 'dark' : 'light',
        primary: {
          main: '#1976d2',
        },
        secondary: {
          main: '#dc004e',
        },
      },
    }),
    [nextTheme]
  );

  return (
    <React.StrictMode>
      <Head>
        <meta name="viewport" content="initial-scale=1, width=device-width" />
        <title>AstroShield</title>
      </Head>
      <ThemeProvider>
        <MUIThemeProvider theme={muiTheme}>
          <CssBaseline />
          <Component {...pageProps} />
        </MUIThemeProvider>
      </ThemeProvider>
    </React.StrictMode>
  );
};

export default App;
