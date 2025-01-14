import React from 'react';
import { useTheme } from 'next-themes';

export default function Home() {
  const { theme, setTheme } = useTheme();

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <h1 className="text-4xl font-bold mb-4">
        Welcome to AstroShield Dashboard v1.0.1
      </h1>
      <p className="text-lg mb-4">
        Current theme: {theme}
      </p>
      <button
        onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        Toggle Theme
      </button>
    </main>
  );
} 