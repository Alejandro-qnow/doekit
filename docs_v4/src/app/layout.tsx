import type { ReactNode } from 'react';

/**
 * Next.js requires a root layout. Locale-specific `<html>` / `<body>` live in
 * `app/[lang]/layout.tsx` (Fumadocs i18n pattern).
 */
export default function RootLayout({ children }: { children: ReactNode }) {
  return children;
}
