"use client";
import { createContext, useContext, PropsWithChildren } from 'react';
import { ColorSchemeScript, MantineProvider, createTheme } from '@mantine/core';
import { useLocalStorage } from '@mantine/hooks';

interface ThemeContextProps {
  colorScheme: 'light' | 'dark';
  toggleColorScheme: () => void;
}

const ThemeContext = createContext<ThemeContextProps | undefined>(undefined);

export const useTheme = () => {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
};

export default function ThemeProvider({ children }: PropsWithChildren) {
  const [colorScheme, setColorScheme] = useLocalStorage<'light' | 'dark'>({
    key: 'color-scheme',
    defaultValue: 'light',
  });

  const toggleColorScheme = () =>
    setColorScheme(colorScheme === 'dark' ? 'light' : 'dark');

  const theme = createTheme({});

  return (
    <>
      <ColorSchemeScript defaultColorScheme={colorScheme} />
      <ThemeContext.Provider value={{ colorScheme, toggleColorScheme }}>
        <MantineProvider theme={theme} forceColorScheme={colorScheme}>
          {children}
        </MantineProvider>
      </ThemeContext.Provider>
    </>
  );
}
