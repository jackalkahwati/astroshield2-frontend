"use client"

import * as React from "react"
import { ThemeProvider as NextThemesProvider, useTheme } from "next-themes"
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { getTheme } from '@/theme';

interface Props {
  children: React.ReactNode
}

function ThemeProviderInner({ children }: Props) {
  const { resolvedTheme } = useTheme();
  const muiTheme = React.useMemo(() => getTheme(resolvedTheme || 'light'), [resolvedTheme]);

  return (
    <MuiThemeProvider theme={muiTheme}>
      <CssBaseline />
      {children}
    </MuiThemeProvider>
  );
}

export function ThemeProvider({ children }: Props) {
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <NextThemesProvider
      attribute="data-theme"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <ThemeProviderInner>
        {children}
      </ThemeProviderInner>
    </NextThemesProvider>
  );
}
