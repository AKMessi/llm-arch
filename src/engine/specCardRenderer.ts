import { COMPONENT_MAP, STAGES } from '../data/components';
import { classifyArchetype } from '../data/challenges';
import {
  formatContext,
  formatGb,
  formatKiB,
  formatParams,
} from './physics';
import type {
  ArchitectureConfig,
  PhysicsReport,
  SimilarityMatch,
} from '../types/architecture';

/**
 * Renders a crisp 2× Retina (2400×1350 -> 1200×675) Swiss-Architectural Spec Card
 * directly on an offscreen HTML5 Canvas for instant Twitter/X, Reddit & LinkedIn sharing!
 */
export function renderSpecCardCanvas(
  config: ArchitectureConfig,
  physics: PhysicsReport,
  topMatch: SimilarityMatch | undefined,
  theme: 'paper' | 'obsidian'
): HTMLCanvasElement {
  const W = 1200;
  const H = 675;
  const DPR = 2;
  const canvas = document.createElement('canvas');
  canvas.width = W * DPR;
  canvas.height = H * DPR;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(DPR, DPR);

  const isPaper = theme === 'paper';
  const bg = isPaper ? '#f7f4ec' : '#0d1017';
  const card = isPaper ? '#ffffff' : '#141924';
  const cardSub = isPaper ? '#f1ede4' : '#1b2230';
  const border = isPaper ? '#d8d1c4' : '#2a3345';
  const ink = isPaper ? '#15171c' : '#f2f5fa';
  const inkSec = isPaper ? '#4c5260' : '#a4b0c5';
  const inkMuted = isPaper ? '#79808f' : '#707d96';

  const stageColors: Record<string, string> = isPaper
    ? {
        tokenizer: '#2563eb',
        position: '#4f46e5',
        norm: '#0d9488',
        attention: '#ea580c',
        hybrid: '#d97706',
        ffn: '#059669',
        output: '#e11d48',
      }
    : {
        tokenizer: '#60a5fa',
        position: '#818cf8',
        norm: '#2dd4bf',
        attention: '#fb923c',
        hybrid: '#fbbf24',
        ffn: '#34d399',
        output: '#fb7185',
      };

  // Background fill + subtle blueprint dot-grid
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  ctx.fillStyle = isPaper ? 'rgba(21,23,28,0.08)' : 'rgba(165,176,196,0.1)';
  for (let gx = 20; gx < W; gx += 24) {
    for (let gy = 20; gy < H; gy += 24) {
      ctx.fillRect(gx, gy, 1.4, 1.4);
    }
  }

  // Outer Swiss architectural frame
  roundRect(ctx, 28, 28, W - 56, H - 56, 14);
  ctx.fillStyle = card;
  ctx.fill();
  ctx.strokeStyle = border;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Top Accent Bar
  ctx.fillStyle = stageColors.attention;
  ctx.fillRect(28, 28, W - 56, 5);

  const archetype = classifyArchetype(config, physics);

  // Header Brand & Tag
  ctx.fillStyle = inkMuted;
  ctx.font = '700 11px "JetBrains Mono", monospace';
  ctx.fillText('ARCHITECTON · LLM ARCHITECTURE BLUEPRINT CARD', 60, 68);

  // Right Pill: Decoder Type + Precision
  const badgeText = `${physics.decoderTypeLabel.toUpperCase()} · ${config.dims.precision.toUpperCase()}`;
  ctx.font = '700 11px "JetBrains Mono", monospace';
  const badgeW = ctx.measureText(badgeText).width + 24;
  roundRect(ctx, W - 60 - badgeW, 50, badgeW, 26, 5);
  ctx.fillStyle = cardSub;
  ctx.fill();
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = stageColors.attention;
  ctx.fillText(badgeText, W - 60 - badgeW + 12, 67);

  // Main Blueprint Title + Archetype Subtitle
  ctx.fillStyle = ink;
  ctx.font = '700 34px "Instrument Sans", "Inter", sans-serif';
  ctx.fillText(truncate(config.name, 36), 60, 114);

  ctx.fillStyle = stageColors.attention;
  ctx.font = '600 14px "Inter", sans-serif';
  ctx.fillText(`${archetype.title}  —  `, 60, 140);
  const archW = ctx.measureText(`${archetype.title}  —  `).width;
  ctx.fillStyle = inkSec;
  ctx.font = '400 14px "Inter", sans-serif';
  ctx.fillText(truncate(archetype.subtitle, 68), 60 + archW, 140);

  // Divider
  ctx.strokeStyle = border;
  ctx.beginPath();
  ctx.moveTo(60, 160);
  ctx.lineTo(W - 60, 160);
  ctx.stroke();

  // ==================== 5 HERO METRIC CARDS ====================
  const stats = [
    {
      label: 'TOTAL / ACTIVE',
      value: formatParams(physics.params.total),
      sub: `${formatParams(physics.params.active)} act (${physics.params.activeRatioPct.toFixed(1)}%)`,
      color: stageColors.ffn,
    },
    {
      label: 'KV-CACHE / TOK',
      value: formatKiB(physics.kvKiBPerTokenBf16),
      sub: physics.kvCompressionVsMha >= 999 ? '0 KV · O(1) State' : `${physics.kvCompressionVsMha.toFixed(1)}× leaner vs MHA`,
      color: stageColors.attention,
    },
    {
      label: `VRAM @ ${formatContext(config.dims.contextLength)}`,
      value: formatGb(physics.totalVramGb),
      sub: `W: ${formatGb(physics.weightsGb)} · KV: ${formatGb(physics.kvCacheGbAtContext)}`,
      color: stageColors.tokenizer,
    },
    {
      label: 'EST. H100 DECODE',
      value: `${Math.round(physics.estDecodeTokSecH100)} tok/s`,
      sub: `${physics.economics.maxConcurrent1xH100} conc. on 1×H100`,
      color: stageColors.norm,
    },
    {
      label: 'GALLERY DNA MATCH',
      value: topMatch ? `${topMatch.score}%` : 'Custom',
      sub: topMatch ? truncate(topMatch.model.name, 20) : 'Original Design',
      color: stageColors.position,
    },
  ];

  const statW = (W - 120 - 4 * 14) / 5;
  stats.forEach((s, i) => {
    const x = 60 + i * (statW + 14);
    const y = 178;
    roundRect(ctx, x, y, statW, 90, 8);
    ctx.fillStyle = cardSub;
    ctx.fill();
    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = s.color;
    ctx.fillRect(x, y + 12, 3.5, 66);

    ctx.fillStyle = inkMuted;
    ctx.font = '700 10px "JetBrains Mono", monospace';
    ctx.fillText(s.label, x + 14, y + 24);

    ctx.fillStyle = ink;
    ctx.font = '700 24px "JetBrains Mono", monospace';
    ctx.fillText(s.value, x + 14, y + 54);

    ctx.fillStyle = inkSec;
    ctx.font = '500 11px "JetBrains Mono", monospace';
    ctx.fillText(s.sub, x + 14, y + 76);
  });

  // ==================== 7-STAGE ARCHITECTURAL STACK GRID ====================
  ctx.fillStyle = inkMuted;
  ctx.font = '700 10.5px "JetBrains Mono", monospace';
  ctx.fillText('7-STAGE DECODER ARCHITECTURE SPECIFICATION', 60, 300);

  const colW = (W - 120 - 16) / 2;
  STAGES.forEach((stage, idx) => {
    const comp = COMPONENT_MAP[config[stage.id]];
    const col = idx < 4 ? 0 : 1;
    const row = idx < 4 ? idx : idx - 4;
    const x = 60 + col * (colW + 16);
    const y = 314 + row * 56;
    const sColor = stageColors[stage.id];

    roundRect(ctx, x, y, colW, 46, 6);
    ctx.fillStyle = bg;
    ctx.fill();
    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = sColor;
    ctx.fillRect(x, y + 8, 3.5, 30);

    ctx.fillStyle = sColor;
    ctx.font = '700 10px "JetBrains Mono", monospace';
    ctx.fillText(`0${stage.order} ${stage.shortLabel.toUpperCase()}`, x + 14, y + 19);

    ctx.fillStyle = ink;
    ctx.font = '700 13.5px "Inter", sans-serif';
    ctx.fillText(truncate(comp?.name ?? '', 34), x + 14, y + 37);

    // Right badge
    const bTxt = comp?.badge ?? '';
    ctx.font = '600 10.5px "JetBrains Mono", monospace';
    const bw = ctx.measureText(bTxt).width + 14;
    roundRect(ctx, x + colW - bw - 10, y + 12, bw, 22, 4);
    ctx.fillStyle = card;
    ctx.fill();
    ctx.strokeStyle = border;
    ctx.stroke();
    ctx.fillStyle = sColor;
    ctx.fillText(bTxt, x + colW - bw - 3, y + 26.5);
  });

  // Geometry summary box in 8th slot (Row 3, Col 1)
  {
    const x = 60 + colW + 16;
    const y = 314 + 3 * 56;
    roundRect(ctx, x, y, colW, 46, 6);
    ctx.fillStyle = cardSub;
    ctx.fill();
    ctx.strokeStyle = border;
    ctx.stroke();

    ctx.fillStyle = inkMuted;
    ctx.font = '700 10px "JetBrains Mono", monospace';
    ctx.fillText('GEOMETRY & LAYER SCHEDULE', x + 14, y + 19);

    ctx.fillStyle = ink;
    ctx.font = '600 12px "JetBrains Mono", monospace';
    ctx.fillText(
      `d=${config.dims.dModel} · L=${config.dims.nLayers} · H=${config.dims.nHeads} (d_h=${physics.dHead}) · ${truncate(physics.layerMixSummary, 24)}`,
      x + 14,
      y + 37
    );
  }

  // ==================== MINI LAYER STACK RIBBON ====================
  ctx.fillStyle = inkMuted;
  ctx.font = '700 10px "JetBrains Mono", monospace';
  ctx.fillText(
    `LAYER-BY-LAYER TOPOLOGY (L1–L${physics.layers.length}${physics.effectiveDepth > physics.layers.length ? ` · ${physics.effectiveDepth} EFF. LAYERS` : ''})`,
    60,
    564
  );

  const ribbonX = 60;
  const ribbonW = W - 120;
  const gap = 2;
  const cellW = Math.max(3, (ribbonW - gap * (physics.layers.length - 1)) / physics.layers.length);
  physics.layers.forEach((layer, i) => {
    const cx = ribbonX + i * (cellW + gap);
    if (layer.loopRole === 'looped') {
      ctx.fillStyle = stageColors.position;
      ctx.fillRect(cx, 571, cellW, 4);
    }
    const mColor =
      layer.mixerKind === 'primary'
        ? stageColors.attention
        : layer.mixerKind === 'swa'
          ? stageColors.hybrid
          : layer.mixerKind === 'deltanet' || layer.mixerKind === 'kda'
            ? stageColors.position
            : stageColors.norm;
    ctx.fillStyle = mColor;
    ctx.fillRect(cx, 577, cellW, 17);
    ctx.fillStyle = layer.ffnKind !== 'dense' ? stageColors.ffn : border;
    ctx.fillRect(cx, 596, cellW, 6);
  });

  // Footer Attribution
  ctx.fillStyle = inkMuted;
  ctx.font = '500 11px "JetBrains Mono", monospace';
  ctx.fillText(
    'Grounded in Sebastian Raschka’s LLM Architecture Gallery (132 Open-Weight Models · Apache 2.0)',
    60,
    632
  );
  ctx.fillStyle = inkSec;
  ctx.font = '700 11px "JetBrains Mono", monospace';
  const host =
    typeof window !== 'undefined' &&
    window.location.host &&
    !window.location.host.startsWith('localhost') &&
    !window.location.host.startsWith('127.0.0.1')
      ? window.location.host
      : 'ARCHITECTON · LLM STUDIO';
  ctx.fillText(host, W - 60 - ctx.measureText(host).width, 632);

  return canvas;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
}
