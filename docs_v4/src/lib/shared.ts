export const appName = 'doekit';
export const appDescription =
  'Design of Experiments (DoE) in Python — screening, response surface, optimal design, and an agentic evaluation layer.';
export const docsRoute = '/docs';
export const docsImageRoute = '/og/docs';
export const docsContentRoute = '/llms.mdx/docs';

export const gitConfig = {
  user: 'Alejandro-qnow',
  repo: 'doekit',
  branch: 'main',
};

export function localePath(locale: string, path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  if (locale === 'en') return normalized === '/' ? '/' : normalized;
  return normalized === '/' ? `/${locale}` : `/${locale}${normalized}`;
}
