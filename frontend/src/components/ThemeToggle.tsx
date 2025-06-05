"use client";
import { ActionIcon } from '@mantine/core';
import { IconSun, IconMoon } from '@tabler/icons-react';
import { useTheme } from '../context/ThemeProvider';

export default function ThemeToggle() {
  const { colorScheme, toggleColorScheme } = useTheme();
  const dark = colorScheme === 'dark';
  return (
    <ActionIcon
      variant="default"
      onClick={toggleColorScheme}
      aria-label="Toggle color scheme"
    >
      {dark ? <IconSun size={18} /> : <IconMoon size={18} />}
    </ActionIcon>
  );
}
