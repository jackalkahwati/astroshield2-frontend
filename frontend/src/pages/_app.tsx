import React from 'react';
import type { AppProps } from 'next/app';
import Head from 'next/head';
import { ThemeProvider } from '../components/theme-provider';
import '../styles/globals.css';

const App = ({ Component, pageProps }: AppProps) => {
  return (
    <React.StrictMode>
      <Head>
        <meta name="viewport" content="initial-scale=1, width=device-width" />
        <title>AstroShield</title>
      </Head>
      <ThemeProvider>
        <Component {...pageProps} />
      </ThemeProvider>
    </React.StrictMode>
  );
};

export default App;
