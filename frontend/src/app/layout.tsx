import type { Metadata } from 'next';
import './globals.css';
import ThemeProvider from '../context/ThemeProvider';

export const metadata: Metadata = {
  title: 'AI Training Platform',
  description: 'Frontend for AI fine-tuning',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
