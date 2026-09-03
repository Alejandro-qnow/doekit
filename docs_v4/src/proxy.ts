import { NextFetchEvent, NextRequest, NextResponse } from 'next/server';
import { isMarkdownPreferred } from 'fumadocs-core/negotiation';
import { createI18nMiddleware } from 'fumadocs-core/i18n/middleware';
import { i18n } from '@/lib/i18n';
import { docsContentRoute, docsRoute } from '@/lib/shared';

const handleI18n = createI18nMiddleware({
  ...i18n,
  format(locale, pathname) {
    // Avoid `/en/` (trailing slash) which breaks the root rewrite to home
    if (!pathname || pathname === '/') return `/${locale}`;
    return `/${locale}${pathname.startsWith('/') ? pathname : `/${pathname}`}`;
  },
});

function markdownTarget(lang: string, docsPath: string): string {
  const suffix = docsPath ? `/${docsPath}` : '';
  return `/${lang}${docsContentRoute}${suffix}/content.md`;
}

function matchDocsPath(
  pathname: string,
): { lang: string; path: string } | null {
  const withLocale = pathname.match(
    new RegExp(`^/(${i18n.languages.join('|')})${docsRoute}(?:/(.*))?$`),
  );
  if (withLocale) {
    return { lang: withLocale[1], path: withLocale[2] ?? '' };
  }

  if (pathname === docsRoute || pathname.startsWith(`${docsRoute}/`)) {
    return {
      lang: i18n.defaultLanguage,
      path: pathname.slice(docsRoute.length).replace(/^\//, ''),
    };
  }

  return null;
}

export function proxy(request: NextRequest, event: NextFetchEvent) {
  const pathname = request.nextUrl.pathname;

  const mdMatch = pathname.match(
    new RegExp(
      `^(?:/(${i18n.languages.join('|')}))?${docsRoute}(?:/(.*))?\\.md$`,
    ),
  );
  if (mdMatch) {
    const lang = mdMatch[1] ?? i18n.defaultLanguage;
    const path = mdMatch[2] ?? '';
    return NextResponse.rewrite(
      new URL(markdownTarget(lang, path), request.nextUrl),
    );
  }

  if (isMarkdownPreferred(request)) {
    const matched = matchDocsPath(pathname);
    if (matched) {
      return NextResponse.rewrite(
        new URL(markdownTarget(matched.lang, matched.path), request.nextUrl),
        { headers: { Vary: 'Accept' } },
      );
    }
  }

  return handleI18n(request, event);
}

export const config = {
  matcher: [
    '/',
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
};
