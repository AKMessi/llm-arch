import React, { useMemo, useState } from 'react';
import { Compass, Download, TrendingUp } from 'lucide-react';
import { formatKiB, formatParams } from '../engine/physics';
import type {
  ArchitectureConfig,
  GalleryModel,
  PhysicsReport,
} from '../types/architecture';

interface ParetoFrontierChartProps {
  config: ArchitectureConfig;
  physics: PhysicsReport;
  galleryModels: GalleryModel[];
  onCloneModel: (model: GalleryModel) => void;
}

type FamilyBucket = 'Dense' | 'Sparse MoE' | 'Hybrid / Linear' | 'SSM / Recurrent';

function classifyFamily(m: GalleryModel): FamilyBucket {
  const dt = m.decoderType.toLowerCase();
  const attn = (m.attention + ' ' + m.layerMix).toLowerCase();
  if (dt.includes('recurrent') || attn.includes('mamba') || attn.includes('mlstm')) {
    return 'SSM / Recurrent';
  }
  if (dt.includes('hybrid') || attn.includes('deltanet') || attn.includes('kimi delta')) {
    return 'Hybrid / Linear';
  }
  if (dt.includes('moe')) return 'Sparse MoE';
  return 'Dense';
}

const FAMILY_COLORS: Record<FamilyBucket, string> = {
  Dense: 'var(--stage-tokenizer)',
  'Sparse MoE': 'var(--stage-ffn)',
  'Hybrid / Linear': 'var(--stage-position)',
  'SSM / Recurrent': 'var(--stage-norm)',
};

export const ParetoFrontierChart: React.FC<ParetoFrontierChartProps> = ({
  config,
  physics,
  galleryModels,
  onCloneModel,
}) => {
  const [hoveredModel, setHoveredModel] = useState<GalleryModel | null>(null);
  const [familyFilter, setFamilyFilter] = useState<FamilyBucket | 'ALL'>('ALL');

  const plotted = useMemo(() => {
    return galleryModels
      .filter(
        (m) =>
          m.totalParamsB !== null &&
          m.totalParamsB > 0 &&
          m.kvCacheKiB !== null
      )
      .map((m) => ({
        model: m,
        family: classifyFamily(m),
        paramsB: m.totalParamsB!,
        kvKiB: m.kvCacheKiB!,
      }))
      .filter((pt) => familyFilter === 'ALL' || pt.family === familyFilter);
  }, [galleryModels, familyFilter]);

  // Concept adoption leaderboard across ALL 120 real models
  const topConcepts = useMemo(() => {
    const counts = new Map<string, number>();
    galleryModels.forEach((m) => {
      m.concepts.forEach((c) => counts.set(c, (counts.get(c) ?? 0) + 1));
    });
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([tag, count]) => ({
        tag,
        count,
        pct: Math.round((count / galleryModels.length) * 100),
        equipped: physics.conceptTags.includes(tag),
      }));
  }, [galleryModels, physics.conceptTags]);

  // Log scale X: 0.25B .. 3000B  |  Log+1 scale Y: 0 .. 1600 KiB/tok
  const minLogX = Math.log10(0.25);
  const maxLogX = Math.log10(3200);
  const mapX = (paramsB: number) => {
    const clamped = Math.max(0.25, Math.min(3200, paramsB));
    const t = (Math.log10(clamped) - minLogX) / (maxLogX - minLogX);
    return 56 + t * 570;
  };

  const maxLogY = Math.log10(1600 + 1);
  const mapY = (kvKiB: number) => {
    const clamped = Math.max(0, Math.min(1600, kvKiB));
    const t = Math.log10(clamped + 1) / maxLogY;
    return 244 - t * 214;
  };

  const userX = mapX(physics.params.total / 1e9);
  const userY = mapY(physics.kvKiBPerTokenBf16);

  const xTicks = [
    { val: 1, label: '1B' },
    { val: 8, label: '8B' },
    { val: 32, label: '32B' },
    { val: 120, label: '120B' },
    { val: 500, label: '500B' },
    { val: 2000, label: '2T' },
  ];

  const yTicks = [
    { val: 0, label: '0 B' },
    { val: 10, label: '10 KiB' },
    { val: 50, label: '50 KiB' },
    { val: 200, label: '200 KiB' },
    { val: 1000, label: '1 MiB' },
  ];

  return (
    <section
      className="rounded-lg border p-4 space-y-4"
      style={{
        backgroundColor: 'var(--bg-panel)',
        borderColor: 'var(--border-hairline)',
        boxShadow: 'var(--shadow-card)',
      }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b pb-3" style={{ borderColor: 'var(--border-hairline)' }}>
        <div>
          <div className="flex items-center gap-2">
            <Compass className="w-4 h-4" style={{ color: 'var(--stage-attention)' }} />
            <h2 className="font-display font-bold text-[14px]">
              120-Model Pareto Frontier: Scale vs. KV-Cache Efficiency
            </h2>
          </div>
          <p className="text-[11.5px] mt-0.5" style={{ color: 'var(--ink-secondary)' }}>
            Lower-right is the engineering frontier (massive capacity with minimal KV-cache/tok). Your active build is marked with the vermilion crosshair star — click any model dot to clone it!
          </p>
        </div>

        <div className="flex flex-wrap gap-1">
          {(['ALL', 'Dense', 'Sparse MoE', 'Hybrid / Linear', 'SSM / Recurrent'] as const).map(
            (fam) => {
              const active = familyFilter === fam;
              return (
                <button
                  key={fam}
                  type="button"
                  onClick={() => setFamilyFilter(fam)}
                  className="px-2 py-0.5 rounded text-[10.5px] font-medium border cursor-pointer transition-colors flex items-center gap-1"
                  style={
                    active
                      ? {
                          backgroundColor: 'var(--ink-primary)',
                          color: 'var(--ink-inverse)',
                          borderColor: 'var(--ink-primary)',
                        }
                      : {
                          backgroundColor: 'var(--bg-slot)',
                          color: 'var(--ink-secondary)',
                          borderColor: 'var(--border-hairline)',
                        }
                  }
                >
                  {fam !== 'ALL' && (
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: FAMILY_COLORS[fam] }}
                    />
                  )}
                  {fam}
                </button>
              );
            }
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
        {/* Scatter Chart */}
        <div className="lg:col-span-8 space-y-2">
          <div
            className="rounded-lg border p-2 overflow-x-auto"
            style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
          >
            <svg viewBox="0 0 650 280" className="w-full min-w-[480px] h-auto select-none">
              {/* Frontier sweet-spot shaded zone (bottom right) */}
              <rect
                x={mapX(30)}
                y={mapY(40)}
                width={626 - mapX(30)}
                height={244 - mapY(40)}
                fill="var(--stage-ffn-soft)"
                rx="4"
              />
              <text
                x="618"
                y="236"
                textAnchor="end"
                fontSize="8.5"
                fontWeight="700"
                fontFamily="'JetBrains Mono', monospace"
                fill="var(--stage-ffn)"
              >
                FRONTIER ZONE (HIGH SCALE · ULTRA-LEAN KV)
              </text>

              {/* Y Grid & Ticks */}
              {yTicks.map((t) => {
                const y = mapY(t.val);
                return (
                  <g key={t.label}>
                    <line x1="56" y1={y} x2="626" y2={y} stroke="var(--border-hairline)" strokeWidth="0.9" />
                    <text x="50" y={y + 3} textAnchor="end" fontSize="8.5" fontFamily="'JetBrains Mono', monospace" fill="var(--ink-muted)">
                      {t.label}
                    </text>
                  </g>
                );
              })}

              {/* X Grid & Ticks */}
              {xTicks.map((t) => {
                const x = mapX(t.val);
                return (
                  <g key={t.label}>
                    <line x1={x} y1="24" x2={x} y2="244" stroke="var(--border-hairline)" strokeWidth="0.9" strokeDasharray="3 3" />
                    <text x={x} y="260" textAnchor="middle" fontSize="8.5" fontFamily="'JetBrains Mono', monospace" fill="var(--ink-muted)">
                      {t.label}
                    </text>
                  </g>
                );
              })}

              {/* Axis Titles */}
              <text x="340" y="275" textAnchor="middle" fontSize="9" fontWeight="700" fill="var(--ink-secondary)">
                TOTAL PARAMETERS (LOG SCALE)
              </text>
              <text x="14" y="138" textAnchor="middle" fontSize="9" fontWeight="700" fill="var(--ink-secondary)" transform="rotate(-90 14 138)">
                KV-CACHE / TOKEN BF16
              </text>

              {/* 120 Gallery Model Dots */}
              {plotted.map((pt) => {
                const cx = mapX(pt.paramsB);
                const cy = mapY(pt.kvKiB);
                const isHov = hoveredModel?.id === pt.model.id;
                return (
                  <circle
                    key={pt.model.id}
                    cx={cx}
                    cy={cy}
                    r={isHov ? 6.5 : 4.2}
                    fill={FAMILY_COLORS[pt.family]}
                    stroke={isHov ? 'var(--ink-primary)' : 'var(--bg-panel)'}
                    strokeWidth={isHov ? 2 : 1}
                    opacity={hoveredModel && !isHov ? 0.5 : 0.9}
                    className="cursor-pointer transition-all"
                    onMouseEnter={() => setHoveredModel(pt.model)}
                    onMouseLeave={() => setHoveredModel(null)}
                    onClick={() => onCloneModel(pt.model)}
                  />
                );
              })}

              {/* USER'S ACTIVE ARCHITECTURE CROSSHAIR STAR */}
              <g className="pointer-events-none">
                <line x1={userX} y1="24" x2={userX} y2="244" stroke="var(--stage-attention)" strokeWidth="1" strokeDasharray="3 2" opacity="0.65" />
                <line x1="56" y1={userY} x2="626" y2={userY} stroke="var(--stage-attention)" strokeWidth="1" strokeDasharray="3 2" opacity="0.65" />
                <circle cx={userX} cy={userY} r="10" fill="var(--stage-attention)" opacity="0.2" />
                <circle cx={userX} cy={userY} r="5.5" fill="var(--stage-attention)" stroke="#fff" strokeWidth="1.8" />
                <rect
                  x={Math.min(500, Math.max(60, userX + 10))}
                  y={Math.max(26, userY - 22)}
                  width="116"
                  height="18"
                  rx="4"
                  fill="var(--stage-attention)"
                />
                <text
                  x={Math.min(500, Math.max(60, userX + 10)) + 58}
                  y={Math.max(26, userY - 22) + 12}
                  textAnchor="middle"
                  fontSize="8.5"
                  fontWeight="700"
                  fontFamily="'JetBrains Mono', monospace"
                  fill="#fff"
                >
                  YOUR BUILD ({formatKiB(physics.kvKiBPerTokenBf16)})
                </text>
              </g>
            </svg>
          </div>

          {/* Hover / Active Model Readout Strip */}
          <div
            className="rounded-md border px-3 py-2 flex items-center justify-between gap-2 text-[11px] min-h-[40px]"
            style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
          >
            {hoveredModel ? (
              <>
                <div className="min-w-0">
                  <strong className="font-display">{hoveredModel.name}</strong>{' '}
                  <span style={{ color: 'var(--ink-muted)' }}>
                    ({hoveredModel.company} · {hoveredModel.date})
                  </span>
                  <div className="text-[10.5px] font-mono-num truncate" style={{ color: 'var(--ink-secondary)' }}>
                    {hoveredModel.scale} · KV: {hoveredModel.kvCachePerTokenRaw} · {hoveredModel.attention}
                  </div>
                </div>
                <span
                  className="text-[10.5px] font-semibold flex items-center gap-1 shrink-0"
                  style={{ color: 'var(--stage-attention)' }}
                >
                  <Download className="w-3 h-3" /> Click dot to clone
                </span>
              </>
            ) : (
              <span style={{ color: 'var(--ink-muted)' }}>
                Hover any of the <strong>{plotted.length} plotted gallery models</strong> to inspect, or click a dot to clone it onto your canvas. Your build: <strong>{config.name}</strong> ({formatParams(physics.params.total)} · {formatKiB(physics.kvKiBPerTokenBf16)}/tok).
              </span>
            )}
          </div>
        </div>

        {/* Right: Concept Adoption Leaderboard across 120 Models */}
        <div
          className="lg:col-span-4 rounded-lg border p-3 space-y-2"
          style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5" style={{ color: 'var(--stage-tokenizer)' }} />
              <span className="font-display font-bold text-[11px] uppercase tracking-[0.06em]">
                Gallery Adoption (120 Models)
              </span>
            </div>
            <span className="text-[9.5px] font-mono-num" style={{ color: 'var(--stage-attention)' }}>
              ● = In Yours
            </span>
          </div>

          <div className="space-y-1.5">
            {topConcepts.map((c) => (
              <div key={c.tag} className="space-y-0.5">
                <div className="flex items-center justify-between text-[10.5px] font-mono-num">
                  <span className="flex items-center gap-1.5">
                    <span
                      className="w-1.5 h-1.5 rounded-full"
                      style={{
                        backgroundColor: c.equipped
                          ? 'var(--stage-attention)'
                          : 'var(--border-strong)',
                      }}
                    />
                    <span
                      style={{
                        color: c.equipped ? 'var(--ink-primary)' : 'var(--ink-secondary)',
                        fontWeight: c.equipped ? 700 : 500,
                      }}
                    >
                      {c.tag}
                    </span>
                  </span>
                  <span style={{ color: 'var(--ink-muted)' }}>
                    {c.count} ({c.pct}%)
                  </span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--border-hairline)' }}>
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${c.pct}%`,
                      backgroundColor: c.equipped
                        ? 'var(--stage-attention)'
                        : 'var(--stage-tokenizer)',
                      opacity: c.equipped ? 1 : 0.65,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
