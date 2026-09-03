import Image from "next/image";
import Link from "next/link";
import { appDescription, appName, docsRoute, localePath } from "@/lib/shared";

export function HomeHero({ locale }: { locale: string }) {
  const docsHref = localePath(locale, docsRoute);
  const cta = locale === "es" ? "Ver docs" : "See docs";

  return (
    <section className="relative flex min-h-[calc(100dvh-3.5rem)] flex-1 flex-col justify-center bg-[var(--Colors-Neutral-100)] text-[var(--Texto-fuerte-100)]">
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <svg
          className="size-full"
          xmlns="http://www.w3.org/2000/svg"
          width="100%"
          height="100%"
        >
          <defs>
            <pattern
              id="home-hero-grid"
              width="100"
              height="100"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 100 0 L 0 0 0 100"
                fill="none"
                stroke="var(--grid-stroke)"
                strokeWidth="1"
                vectorEffect="non-scaling-stroke"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#home-hero-grid)" />
        </svg>
      </div>

      <div className="relative z-10 mx-auto flex w-full max-w-[1400px] flex-col items-start gap-8 px-6 py-16 sm:px-10 lg:px-16">
        <Image
          src="/qnow.png"
          alt="QNOW"
          width={56}
          height={56}
          priority
          className="size-24"
        />

        <div className="flex max-w-xl flex-col gap-5">
          <h1 className="font-display text-4xl leading-[1.1] font-bold tracking-tight sm:text-5xl lg:text-6xl">
            {appName}
          </h1>
          <p className="max-w-md text-base leading-relaxed text-[var(--Texto-Debil-100)] sm:text-lg">
            {locale === "es"
              ? "Diseño de Experimentos (DoE) en Python — cribado, superficie de respuesta, diseño óptimo y una capa agéntica de evaluación."
              : appDescription}
          </p>
        </div>

        <Link
          href={docsHref}
          className="inline-flex items-center justify-center rounded-lg bg-white px-5 py-2.5 text-sm font-medium text-black transition-opacity hover:opacity-90"
        >
          {cta}
        </Link>
      </div>
    </section>
  );
}
