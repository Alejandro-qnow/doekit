import { redirect } from 'next/navigation';
import { i18n } from '@/lib/i18n';

/** Fallback when the i18n proxy has not rewritten `/` → `/en` yet. */
export default function RootPage() {
  redirect(`/${i18n.defaultLanguage}`);
}
