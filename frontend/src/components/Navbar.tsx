"use client";
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Group, Box, Container, Anchor } from '@mantine/core';
import ThemeToggle from './ThemeToggle';

const links = [
  { href: '/', label: 'Home' },
  { href: '/dataset', label: 'Dataset Overview' },
  { href: '/training', label: 'Training' },
  { href: '/evaluation', label: 'Evaluation' },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <Box component="header" px="md" py="sm" style={{ borderBottom: '1px solid var(--mantine-color-border)' }}>
      <Container size="lg" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Group>
          <Link href="/" style={{ fontWeight: 700 }}>AI Trainer</Link>
          {links.map((link) => (
            <Anchor
              key={link.href}
              component={Link}
              href={link.href}
              data-active={pathname === link.href || pathname.startsWith(link.href + '/')}
            >
              {link.label}
            </Anchor>
          ))}
        </Group>
        <ThemeToggle />
      </Container>
    </Box>
  );
}
