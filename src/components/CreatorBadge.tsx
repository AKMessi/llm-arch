import React from 'react';
import { Globe } from 'lucide-react';
import { trackEvent } from '../engine/analytics';

interface CreatorBadgeProps {
  variant?: 'compact' | 'card';
}

const SOCIALS = [
  {
    id: 'x',
    label: 'X',
    title: 'Aaryan Kakad on X (@aaryan_kakad)',
    href: 'https://x.com/aaryan_kakad',
    icon: (
      <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="currentColor" aria-hidden="true">
        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
      </svg>
    ),
  },
  {
    id: 'linkedin',
    label: 'LinkedIn',
    title: 'Aaryan Kakad on LinkedIn',
    href: 'https://linkedin.com/in/aaryankakad',
    icon: (
      <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="currentColor" aria-hidden="true">
        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
      </svg>
    ),
  },
  {
    id: 'github',
    label: 'GitHub',
    title: 'Aaryan Kakad on GitHub (@AKMessi)',
    href: 'https://github.com/AKMessi',
    icon: (
      <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="currentColor" aria-hidden="true">
        <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
      </svg>
    ),
  },
  {
    id: 'website',
    label: 'Personal Website',
    title: 'Aaryan Kakad — Personal Website',
    href: 'https://aaryan-kakad.github.io',
    icon: <Globe className="w-3.5 h-3.5" aria-hidden="true" />,
  },
];

export const CreatorBadge: React.FC<CreatorBadgeProps> = ({ variant = 'compact' }) => {
  if (variant === 'compact') {
    return (
      <div
        className="shrink-0 border-t px-3 py-2 flex items-center justify-between gap-2"
        style={{
          backgroundColor: 'var(--bg-panel-subtle)',
          borderColor: 'var(--border-hairline)',
        }}
      >
        <div className="min-w-0 leading-tight">
          <div className="text-[9.5px] font-mono-num uppercase tracking-wider" style={{ color: 'var(--ink-muted)' }}>
            Created by
          </div>
          <a
            href="https://aaryan-kakad.github.io"
            target="_blank"
            rel="noreferrer"
            onClick={() => trackEvent('creator_social_clicked', { platform: 'name_link', location: 'palette_footer' })}
            className="text-[11.5px] font-semibold hover:underline truncate block"
            style={{ color: 'var(--ink-primary)' }}
            title="Aaryan Kakad — Personal Website"
          >
            Aaryan Kakad
          </a>
        </div>

        <div className="flex items-center gap-1 shrink-0">
          {SOCIALS.map((s) => (
            <a
              key={s.id}
              href={s.href}
              target="_blank"
              rel="noreferrer"
              title={s.title}
              aria-label={s.title}
              onClick={() => trackEvent('creator_social_clicked', { platform: s.id, location: 'palette_footer' })}
              className="p-1.5 rounded-md border hover:bg-[var(--bg-panel)] transition-colors"
              style={{
                borderColor: 'var(--border-hairline)',
                color: 'var(--ink-secondary)',
              }}
            >
              {s.icon}
            </a>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div
      className="rounded-lg border p-2.5 space-y-2"
      style={{
        backgroundColor: 'var(--bg-slot)',
        borderColor: 'var(--border-hairline)',
      }}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px]">
          <span style={{ color: 'var(--ink-muted)' }}>Created by </span>
          <a
            href="https://aaryan-kakad.github.io"
            target="_blank"
            rel="noreferrer"
            onClick={() => trackEvent('creator_social_clicked', { platform: 'name_link', location: 'telemetry_card' })}
            className="font-semibold hover:underline"
            style={{ color: 'var(--ink-primary)' }}
          >
            Aaryan Kakad
          </a>
        </div>
        <span
          className="text-[9.5px] font-mono-num px-1.5 py-0.5 rounded border"
          style={{
            borderColor: 'var(--border-hairline)',
            color: 'var(--ink-muted)',
            backgroundColor: 'var(--bg-panel)',
          }}
        >
          MIT OSS
        </span>
      </div>

      <div className="grid grid-cols-2 gap-1.5">
        {SOCIALS.map((s) => (
          <a
            key={s.id}
            href={s.href}
            target="_blank"
            rel="noreferrer"
            title={s.title}
            onClick={() => trackEvent('creator_social_clicked', { platform: s.id, location: 'telemetry_card' })}
            className="flex items-center gap-1.5 px-2 py-1 rounded-md border text-[10.5px] font-medium hover:bg-[var(--bg-panel-subtle)] transition-colors"
            style={{
              backgroundColor: 'var(--bg-panel)',
              borderColor: 'var(--border-hairline)',
              color: 'var(--ink-secondary)',
            }}
          >
            {s.icon}
            <span className="truncate">{s.label}</span>
          </a>
        ))}
      </div>
    </div>
  );
};
