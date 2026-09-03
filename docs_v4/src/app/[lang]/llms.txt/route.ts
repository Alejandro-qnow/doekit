import { source } from '@/lib/source';
import { llms } from 'fumadocs-core/source';

export const revalidate = false;

export async function GET(
  _req: Request,
  { params }: RouteContext<'/[lang]/llms.txt'>,
) {
  const { lang } = await params;
  return new Response(llms(source).index(lang));
}

export function generateStaticParams() {
  return source.getLanguages().map(({ language }) => ({ lang: language }));
}
