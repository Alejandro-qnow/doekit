import { HomeHero } from '@/components/home-hero';

export default async function HomePage({ params }: PageProps<'/[lang]'>) {
  const { lang } = await params;
  return <HomeHero locale={lang} />;
}
