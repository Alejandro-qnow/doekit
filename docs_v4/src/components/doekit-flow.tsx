import type { ReactNode } from 'react';

const copy = {
  en: {
    title: 'doekit DoE flow',
    desc: 'A user or agent enters Design, then Recommend. Recommend branches to Learn or Optimize; both join at Semantic. A choice then stops or continues back to Design.',
    design: 'Design',
    recommend: 'Recommend',
    learn: 'Learn',
    optimize: 'Optimize',
    semantic: 'Semantic',
    userAgent: 'User / Agent',
    ifLearn: 'if learn',
    ifOptimize: 'if optimize',
    ifStop: 'if stop',
    ifContinue: 'if continue',
  },
  es: {
    title: 'Flujo DoE de doekit',
    desc: 'Un usuario o agente entra en Diseño, luego Recomendar. Recomendar se ramifica a Aprender u Optimizar; ambas llegan a Semántica. Luego se elige parar o continuar de vuelta a Diseño.',
    design: 'Diseño',
    recommend: 'Recomendar',
    learn: 'Aprender',
    optimize: 'Optimizar',
    semantic: 'Semántica',
    userAgent: 'Usuario / Agente',
    ifLearn: 'si aprender',
    ifOptimize: 'si optimizar',
    ifStop: 'si parar',
    ifContinue: 'si continuar',
  },
} as const;

export function DoeFunnel({
  locale = 'en',
}: {
  locale?: 'en' | 'es';
}): ReactNode {
  const t = copy[locale] ?? copy.en;
  const markerId = `doe-arrow-${locale}`;

  return (
    <div className="doe-funnel">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 800 200"
        role="img"
        aria-labelledby={`doe-flow-title-${locale} doe-flow-desc-${locale}`}
      >
        <title id={`doe-flow-title-${locale}`}>{t.title}</title>
        <desc id={`doe-flow-desc-${locale}`}>{t.desc}</desc>
        <defs>
          <marker
            id={markerId}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto"
          >
            <path d="M 0 1.2 L 10 5 L 0 8.8 z" className="doe-funnel__marker" />
          </marker>
        </defs>

        <path className="doe-funnel__line" style={{ markerEnd: `url(#${markerId})` }} d="M22,80 H116" />
        <path className="doe-funnel__line" style={{ markerEnd: `url(#${markerId})` }} d="M210,80 H226" />
        <path className="doe-funnel__line" style={{ markerEnd: `url(#${markerId})` }} d="M342,68 C364,68 372,34 400,34" />
        <path className="doe-funnel__line" style={{ markerEnd: `url(#${markerId})` }} d="M342,92 C364,92 372,138 388,138" />
        <path className="doe-funnel__line" style={{ markerEnd: `url(#${markerId})` }} d="M488,34 C512,34 528,80 548,80" />
        <path className="doe-funnel__line" style={{ markerEnd: `url(#${markerId})` }} d="M500,138 C520,138 532,80 548,80" />
        <path className="doe-funnel__line" style={{ markerEnd: `url(#${markerId})` }} d="M650,80 H670" />
        <path className="doe-funnel__line" style={{ markerEnd: `url(#${markerId})` }} d="M706,80 H760" />
        <path className="doe-funnel__line" style={{ markerEnd: `url(#${markerId})` }} d="M688,98 V176 H163 V98" />

        <circle className="doe-funnel__start" cx="16" cy="80" r="6" />
        <rect className="doe-funnel__box" x="118" y="62" width="92" height="36" rx="8" />
        <rect className="doe-funnel__box" x="226" y="62" width="116" height="36" rx="8" />
        <rect className="doe-funnel__box" x="400" y="16" width="88" height="36" rx="8" />
        <rect className="doe-funnel__box" x="388" y="120" width="112" height="36" rx="8" />
        <rect className="doe-funnel__box" x="548" y="62" width="102" height="36" rx="8" />
        <polygon className="doe-funnel__box" points="688,62 706,80 688,98 670,80" />
        <circle className="doe-funnel__end-outer" cx="772" cy="80" r="9" />
        <circle className="doe-funnel__end-inner" cx="772" cy="80" r="5" />

        <text className="doe-funnel__node" x="164" y="80">
          {t.design}
        </text>
        <text className="doe-funnel__node" x="284" y="80">
          {t.recommend}
        </text>
        <text className="doe-funnel__node" x="444" y="34">
          {t.learn}
        </text>
        <text className="doe-funnel__node" x="444" y="138">
          {t.optimize}
        </text>
        <text className="doe-funnel__node" x="599" y="80">
          {t.semantic}
        </text>

        <rect className="doe-funnel__chip" x="26" y="46" width="86" height="16" rx="4" />
        <text className="doe-funnel__label" x="69" y="54">
          {t.userAgent}
        </text>

        <rect className="doe-funnel__chip" x="342" y="6" width="54" height="16" rx="4" />
        <text className="doe-funnel__label" x="369" y="14">
          {t.ifLearn}
        </text>
        <rect className="doe-funnel__chip" x="310" y="154" width="72" height="16" rx="4" />
        <text className="doe-funnel__label" x="347" y="162">
          {t.ifOptimize}
        </text>

        <rect className="doe-funnel__chip" x="712" y="48" width="48" height="16" rx="4" />
        <text className="doe-funnel__label" x="736" y="56">
          {t.ifStop}
        </text>
        <rect className="doe-funnel__chip" x="390" y="168" width="72" height="16" rx="4" />
        <text className="doe-funnel__label" x="426" y="176">
          {t.ifContinue}
        </text>
      </svg>
    </div>
  );
}
