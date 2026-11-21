import React, { useState } from 'react';
import { formatKiB } from '../engine/physics';
import type { LayerSpec, PhysicsReport } from '../types/architecture';

interface LayerStackStripProps {
  physics: PhysicsReport;
}

const MIXER_COLORS: Record<LayerSpec['mixerKind'], { color: string; label: string }> = {
  primary: { color: 'var(--stage-attention)', label: 'Global Primary Attn' },
  swa: { color: 'var(--stage-hybrid)', label: 'Sliding-Window Attn' },
  deltanet: { color: 'var(--stage-position)', label: 'Gated DeltaNet' },
  kda: { color: 'var(--stage-position)', label: 'Kimi Delta Attn' },
  mamba2: { color: 'var(--stage-norm)', label: 'Mamba-2 SSM' },
  mlstm: { color: 'var(--stage-norm)', label: 'mLSTM Recurrent' },
  shortconv: { color: 'var(--stage-tokenizer)', label: 'LIV ShortConv' },
};

export const LayerStackStrip: React.FC<LayerStackStripProps> = ({ physics }) => {
  const [hovered, setHovered] = useState<LayerSpec | null>(null);

  const presentMixers = Array.from(new Set(physics.layers.map((l) => l.mixerKind)));
  const hasMoE = physics.layers.some((l) => l.ffnKind !== 'dense');
  const hasDensePrefix =
    hasMoE && physics.layers.some((l) => l.ffnKind === 'dense');

  return (
    <div
      className="rounded-lg border p-3 space-y-2.5"
      style={{
        backgroundColor: 'var(--bg-panel)',
        borderColor: 'var(--border-hairline)',
        boxShadow: 'var(--shadow-card)',
      }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-display font-bold text-[11px] uppercase tracking-[0.07em]">
            Layer-by-Layer Schedule (L1–L{physics.layers.length})
          </span>
          <span
            className="text-[10px] font-mono-num px-1.5 py-0.5 rounded"
            style={{
              backgroundColor: 'var(--bg-panel-subtle)',
              color: 'var(--ink-secondary)',
            }}
          >
            {physics.layerMixSummary}
          </span>
        </div>

        {/* Dynamic Legend */}
        <div className="flex flex-wrap items-center gap-2.5 text-[10px]" style={{ color: 'var(--ink-muted)' }}>
          {presentMixers.map((mk) => (
            <span key={mk} className="flex items-center gap-1">
              <span
                className="w-2 h-2 rounded-[2px]"
                style={{ backgroundColor: MIXER_COLORS[mk].color }}
              />
              {MIXER_COLORS[mk].label}
            </span>
          ))}
          {hasMoE && (
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--stage-ffn)' }} />
              MoE
            </span>
          )}
          {hasDensePrefix && (
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full border" style={{ borderColor: 'var(--ink-muted)' }} />
              Dense Prefix
            </span>
          )}
        </div>
      </div>

      {/* Ribbon of Layer Cells */}
      <div className="flex items-stretch gap-[3px] w-full overflow-x-auto pb-1">
        {physics.layers.map((layer) => {
          const isHovered = hovered?.index === layer.index;
          const mixerColor = MIXER_COLORS[layer.mixerKind].color;
          const isMoeLayer = layer.ffnKind !== 'dense';
          return (
            <div
              key={layer.index}
              onMouseEnter={() => setHovered(layer)}
              onMouseLeave={() => setHovered(null)}
              className="flex-1 min-w-[7px] max-w-[20px] flex flex-col gap-[2px] cursor-crosshair transition-transform"
              style={{
                transform: isHovered ? 'scaleY(1.18)' : undefined,
              }}
            >
              {/* Top Bar: Sequence Mixer */}
              <div
                className="h-5 rounded-t-[2px] relative"
                style={{
                  backgroundColor: mixerColor,
                  opacity: layer.kvSharedSource ? 0.52 : 0.92,
                }}
              />
              {/* Bottom Bar: FFN / MoE */}
              <div
                className="h-2 rounded-b-[2px]"
                style={{
                  backgroundColor: isMoeLayer ? 'var(--stage-ffn)' : 'var(--border-strong)',
                  opacity: layer.ffnKind === 'latent_moe' ? 0.72 : 0.9,
                }}
              />
            </div>
          );
        })}
      </div>

      {/* Hover Micro-Inspector */}
      <div
        className="flex items-center justify-between text-[10.5px] font-mono-num px-2 py-1 rounded border min-h-[26px]"
        style={{
          backgroundColor: 'var(--bg-slot)',
          borderColor: 'var(--border-hairline)',
          color: 'var(--ink-secondary)',
        }}
      >
        {hovered ? (
          <>
            <span>
              <strong style={{ color: 'var(--ink-primary)' }}>Layer {hovered.index}</strong> ·{' '}
              {hovered.mixerLabel} +{' '}
              {hovered.ffnKind === 'dense'
                ? 'Dense FFN'
                : hovered.ffnKind === 'latent_moe'
                  ? 'Latent MoE'
                  : 'Sparse MoE'}
              {hovered.kvSharedSource ? ` (reuses L${hovered.kvSharedSource} KV)` : ''}
            </span>
            <span style={{ color: 'var(--stage-attention)' }}>
              KV: {formatKiB(hovered.kvBytesPerTokenBf16 / 1024)}/tok
            </span>
          </>
        ) : (
          <>
            <span style={{ color: 'var(--ink-muted)' }}>
              Hover any layer bar to inspect its sequence mixer, FFN type & per-layer KV footprint
            </span>
            <span>
              Total: {formatKiB(physics.kvKiBPerTokenBf16)}/tok BF16
            </span>
          </>
        )}
      </div>
    </div>
  );
};
