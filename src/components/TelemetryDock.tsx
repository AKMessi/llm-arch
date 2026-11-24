import React, { useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  ExternalLink,
  Gauge,
  Info,
  Sparkles,
  Zap,
  GitCompare,
  Download,
} from 'lucide-react';
import { COMPONENT_MAP } from '../data/components';
import {
  formatContext,
  formatGb,
  formatKiB,
  formatParams,
} from '../engine/physics';
import type {
  ArchitectureConfig,
  ComponentDef,
  DiagnosticItem,
  GalleryModel,
  PhysicsReport,
  SimilarityMatch,
} from '../types/architecture';

interface TelemetryDockProps {
  config: ArchitectureConfig;
  physics: PhysicsReport;
  diagnostics: DiagnosticItem[];
  matches: SimilarityMatch[];
  onApplySuggestion: (comp: ComponentDef) => void;
  onCloneGalleryModel: (model: GalleryModel) => void;
  onCompareWithModel: (model: GalleryModel) => void;
  onOpenGallery: () => void;
}

export const TelemetryDock: React.FC<TelemetryDockProps> = ({
  config,
  physics,
  diagnostics,
  matches,
  onApplySuggestion,
  onCloneGalleryModel,
  onCompareWithModel,
  onOpenGallery,
}) => {
  const [hoveredCurveIdx, setHoveredCurveIdx] = useState<number | null>(null);

  // Stacked parameter share percentages
  const pTot = Math.max(1, physics.params.total);
  const embedPct = (physics.params.embeddings / pTot) * 100;
  const attnPct = (physics.params.attention / pTot) * 100;
  const ffnPct = (physics.params.ffnTotal / pTot) * 100;
  const headPct = (physics.params.outputHead / pTot) * 100;

  // KV Scaling Curve SVG Geometry (log-friendly visual scale)
  const curve = physics.kvCurve;
  const maxCurveGb = Math.max(
    1,
    ...curve.map((pt) => Math.max(pt.currentGb, pt.mhaBaselineGb, pt.llamaGqaGb))
  );
  // Square-root compressed Y-axis so both 2 GB and 180 GB curves stay legible & dramatic!
  const mapY = (gb: number) => {
    const norm = Math.sqrt(Math.max(0, gb)) / Math.sqrt(maxCurveGb);
    return 116 - norm * 102;
  };
  const mapX = (idx: number) => 28 + (idx / Math.max(1, curve.length - 1)) * 252;

  const polylinePoints = (key: 'currentGb' | 'mhaBaselineGb' | 'llamaGqaGb' | 'deepseekMlaGb' | 'nemotronHybridGb') =>
    curve.map((pt, i) => `${mapX(i).toFixed(1)},${mapY(pt[key]).toFixed(1)}`).join(' ');

  const activePoint = hoveredCurveIdx !== null ? curve[hoveredCurveIdx] : curve[3]; // default 128K readout

  return (
    <aside
      className="w-full lg:w-[306px] xl:w-[324px] 2xl:w-[354px] shrink-0 lg:border-l h-full overflow-y-auto p-3 space-y-3"
      style={{
        backgroundColor: 'var(--bg-panel)',
        borderColor: 'var(--border-hairline)',
      }}
    >
      {/* ==================== 1. HERO PHYSICS READOUTS ==================== */}
      <section className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-display font-bold text-[11px] uppercase tracking-[0.08em]">
            Live Systems Telemetry
          </span>
          <span
            className="text-[10px] font-mono-num px-1.5 py-0.5 rounded font-semibold uppercase"
            style={{
              backgroundColor: 'var(--bg-panel-subtle)',
              color: 'var(--ink-secondary)',
            }}
          >
            {config.dims.precision.toUpperCase()} · B={config.dims.batchSize}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {/* Parameters Hero */}
          <div
            className="rounded-lg border p-2.5"
            style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
          >
            <div className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--ink-muted)' }}>
              Total / Active Params
            </div>
            <div className="font-mono-num font-bold text-[18px] leading-tight mt-0.5">
              {formatParams(physics.params.total)}
            </div>
            <div className="text-[10.5px] font-mono-num mt-0.5" style={{ color: 'var(--stage-ffn)' }}>
              {formatParams(physics.params.active)} active ({physics.params.activeRatioPct.toFixed(1)}%)
            </div>
          </div>

          {/* KV Cache Hero */}
          <div
            className="rounded-lg border p-2.5"
            style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
          >
            <div className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--ink-muted)' }}>
              KV-Cache / Token
            </div>
            <div
              className="font-mono-num font-bold text-[18px] leading-tight mt-0.5"
              style={{ color: 'var(--stage-attention)' }}
            >
              {formatKiB(physics.kvKiBPerTokenBf16)}
            </div>
            <div className="text-[10.5px] font-mono-num mt-0.5" style={{ color: 'var(--ink-secondary)' }}>
              {physics.kvCompressionVsMha >= 999
                ? '∞× vs MHA (0 KV!)'
                : `${physics.kvCompressionVsMha.toFixed(1)}× leaner vs MHA`}
            </div>
          </div>

          {/* Total VRAM at Context */}
          <div
            className="rounded-lg border p-2.5"
            style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
          >
            <div className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--ink-muted)' }}>
              VRAM @ {formatContext(config.dims.contextLength)}
            </div>
            <div className="font-mono-num font-bold text-[17px] leading-tight mt-0.5">
              {formatGb(physics.totalVramGb)}
            </div>
            <div className="text-[10px] font-mono-num mt-0.5" style={{ color: 'var(--ink-muted)' }}>
              W: {formatGb(physics.weightsGb)} · KV: {formatGb(physics.kvCacheGbAtContext)}
            </div>
          </div>

          {/* Est. Decode Throughput */}
          <div
            className="rounded-lg border p-2.5"
            style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
          >
            <div className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--ink-muted)' }}>
              Est. H100 Decode
            </div>
            <div className="font-mono-num font-bold text-[17px] leading-tight mt-0.5">
              {physics.estDecodeTokSecH100 >= 100
                ? Math.round(physics.estDecodeTokSecH100)
                : physics.estDecodeTokSecH100.toFixed(1)}{' '}
              <span className="text-[11px] font-normal" style={{ color: 'var(--ink-muted)' }}>tok/s</span>
            </div>
            <div className="text-[10px] font-mono-num mt-0.5 truncate" style={{ color: 'var(--ink-secondary)' }}>
              {physics.bottleneck}
            </div>
          </div>
        </div>

        {/* Stacked Parameter Budget Bar */}
        <div
          className="rounded-lg border p-2.5 space-y-1.5"
          style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
        >
          <div className="flex justify-between text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--ink-muted)' }}>
            <span>Parameter Distribution</span>
            <span className="font-mono-num">
              {(physics.flopsPerTokenActive / 1e9).toFixed(1)} GFLOPs/tok
            </span>
          </div>
          <div className="h-2.5 rounded-full overflow-hidden flex gap-[1px] bg-[var(--border-hairline)]">
            <div style={{ width: `${Math.max(1, embedPct)}%`, backgroundColor: 'var(--stage-tokenizer)' }} title={`Embeddings: ${embedPct.toFixed(1)}%`} />
            <div style={{ width: `${Math.max(1, attnPct)}%`, backgroundColor: 'var(--stage-attention)' }} title={`Attention/Mixer: ${attnPct.toFixed(1)}%`} />
            <div style={{ width: `${Math.max(1, ffnPct)}%`, backgroundColor: 'var(--stage-ffn)' }} title={`FFN/MoE: ${ffnPct.toFixed(1)}%`} />
            {headPct > 0.2 && (
              <div style={{ width: `${Math.max(1, headPct)}%`, backgroundColor: 'var(--stage-output)' }} title={`LM Head: ${headPct.toFixed(1)}%`} />
            )}
          </div>
          <div className="flex flex-wrap gap-x-2.5 gap-y-0.5 text-[9.5px] font-mono-num" style={{ color: 'var(--ink-secondary)' }}>
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: 'var(--stage-tokenizer)' }} />
              Emb {embedPct.toFixed(0)}%
            </span>
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: 'var(--stage-attention)' }} />
              Mix {attnPct.toFixed(0)}%
            </span>
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: 'var(--stage-ffn)' }} />
              FFN {ffnPct.toFixed(0)}%
            </span>
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: 'var(--stage-output)' }} />
              Head {headPct.toFixed(0)}%
            </span>
          </div>
        </div>
      </section>

      {/* ==================== 2. INTERACTIVE KV-CACHE SCALING CURVE ==================== */}
      <section
        className="rounded-lg border p-2.5 space-y-1.5"
        style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Gauge className="w-3.5 h-3.5" style={{ color: 'var(--stage-attention)' }} />
            <span className="font-display font-bold text-[11px] uppercase tracking-[0.06em]">
              KV-Cache Scaling (4K → 1M)
            </span>
          </div>
          <span className="text-[10px] font-mono-num font-semibold" style={{ color: 'var(--stage-attention)' }}>
            @{activePoint.label}: {formatGb(activePoint.currentGb)}
          </span>
        </div>

        <svg viewBox="0 0 300 138" className="w-full h-auto overflow-visible select-none">
          {/* Horizontal grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
            const y = 116 - frac * 102;
            const gbVal = (frac * frac) * maxCurveGb;
            return (
              <g key={frac}>
                <line x1="28" y1={y} x2="286" y2={y} stroke="var(--border-hairline)" strokeWidth="0.8" />
                <text x="24" y={y + 3} textAnchor="end" fontSize="7.5" fontFamily="'JetBrains Mono', monospace" fill="var(--ink-muted)">
                  {gbVal >= 10 ? Math.round(gbVal) : gbVal.toFixed(1)}G
                </text>
              </g>
            );
          })}

          {/* Reference Curves */}
          <polyline
            fill="none"
            stroke="var(--ink-muted)"
            strokeWidth="1.2"
            strokeDasharray="3 3"
            opacity="0.55"
            points={polylinePoints('mhaBaselineGb')}
          />
          <polyline
            fill="none"
            stroke="var(--stage-tokenizer)"
            strokeWidth="1.2"
            strokeDasharray="4 2"
            opacity="0.7"
            points={polylinePoints('deepseekMlaGb')}
          />
          <polyline
            fill="none"
            stroke="var(--stage-norm)"
            strokeWidth="1.1"
            strokeDasharray="2 2"
            opacity="0.75"
            points={polylinePoints('nemotronHybridGb')}
          />

          {/* User's Live Architecture Curve */}
          <polyline
            fill="none"
            stroke="var(--stage-attention)"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={polylinePoints('currentGb')}
          />

          {/* Interactive Points & X-axis labels */}
          {curve.map((pt, i) => {
            const cx = mapX(i);
            const cy = mapY(pt.currentGb);
            const isActive = (hoveredCurveIdx ?? 3) === i;
            return (
              <g
                key={pt.label}
                className="cursor-pointer"
                onMouseEnter={() => setHoveredCurveIdx(i)}
                onMouseLeave={() => setHoveredCurveIdx(null)}
              >
                <rect x={cx - 18} y="10" width="36" height="122" fill="transparent" />
                {isActive && (
                  <line x1={cx} y1="14" x2={cx} y2="116" stroke="var(--stage-attention)" strokeWidth="0.9" strokeDasharray="2 2" />
                )}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isActive ? 4 : 2.6}
                  fill="var(--stage-attention)"
                  stroke="var(--bg-panel)"
                  strokeWidth="1.2"
                />
                <text
                  x={cx}
                  y="131"
                  textAnchor="middle"
                  fontSize="8"
                  fontWeight={isActive ? '700' : '500'}
                  fontFamily="'JetBrains Mono', monospace"
                  fill={isActive ? 'var(--ink-primary)' : 'var(--ink-muted)'}
                >
                  {pt.label}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Legend & Hovered Crosshair Delta */}
        <div className="flex flex-wrap items-center justify-between gap-1 text-[9.5px] font-mono-num pt-0.5" style={{ color: 'var(--ink-muted)' }}>
          <span className="flex items-center gap-1" style={{ color: 'var(--stage-attention)' }}>
            <span className="w-2.5 h-[2.5px] inline-block rounded" style={{ backgroundColor: 'var(--stage-attention)' }} />
            Yours ({formatGb(activePoint.currentGb)})
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-[1.5px] inline-block" style={{ backgroundColor: 'var(--ink-muted)' }} />
            MHA ({formatGb(activePoint.mhaBaselineGb)})
          </span>
          <span className="flex items-center gap-1" style={{ color: 'var(--stage-tokenizer)' }}>
            MLA ({formatGb(activePoint.deepseekMlaGb)})
          </span>
          <span className="flex items-center gap-1" style={{ color: 'var(--stage-norm)' }}>
            SSM ({formatGb(activePoint.nemotronHybridGb)})
          </span>
        </div>
      </section>

      {/* ==================== 3. HARDWARE DEPLOYMENT FIT ==================== */}
      <section
        className="rounded-lg border p-2.5 space-y-2"
        style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5" style={{ color: 'var(--ink-secondary)' }} />
            <span className="font-display font-bold text-[11px] uppercase tracking-[0.06em]">
              Hardware Deployment Fit
            </span>
          </div>
          <span className="text-[10px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
            Needs {formatGb(physics.totalVramGb)}
          </span>
        </div>

        <div className="space-y-1.5">
          {physics.hardwareFits.map((hw) => (
            <div key={hw.gpu} className="space-y-0.5">
              <div className="flex items-center justify-between text-[10.5px] font-mono-num">
                <span className="flex items-center gap-1.5">
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{
                      backgroundColor: hw.fits ? 'var(--stage-ffn)' : 'var(--stage-output)',
                    }}
                  />
                  <span style={{ color: hw.fits ? 'var(--ink-primary)' : 'var(--ink-muted)' }}>
                    {hw.gpu}
                  </span>
                </span>
                <span style={{ color: hw.fits ? 'var(--stage-ffn)' : 'var(--stage-output)' }}>
                  {hw.fits ? `${hw.utilizationPct}% · ~${hw.estTokPerSec} tok/s` : 'OOM'}
                </span>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--border-hairline)' }}>
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${hw.utilizationPct}%`,
                    backgroundColor: hw.fits
                      ? hw.utilizationPct > 80
                        ? 'var(--stage-hybrid)'
                        : 'var(--stage-ffn)'
                      : 'var(--stage-output)',
                  }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Serving Concurrency & Chinchilla Training Economics */}
        <div
          className="pt-2 mt-1 border-t grid grid-cols-2 gap-1.5 font-mono-num text-[10px]"
          style={{ borderColor: 'var(--border-hairline)' }}
        >
          <div className="p-1.5 rounded border" style={{ backgroundColor: 'var(--bg-panel)', borderColor: 'var(--border-hairline)' }}>
            <div style={{ color: 'var(--ink-muted)' }}>1×H100 CONCURRENCY</div>
            <div className="font-bold text-[12px] mt-0.5" style={{ color: physics.economics.maxConcurrent1xH100 > 0 ? 'var(--stage-ffn)' : 'var(--stage-output)' }}>
              {physics.economics.maxConcurrent1xH100 > 0
                ? `${physics.economics.maxConcurrent1xH100} users`
                : `8×H100: ${physics.economics.maxConcurrent8xH100}`}
            </div>
          </div>
          <div className="p-1.5 rounded border" style={{ backgroundColor: 'var(--bg-panel)', borderColor: 'var(--border-hairline)' }}>
            <div style={{ color: 'var(--ink-muted)' }}>EST. SERVE $/1M TOK</div>
            <div className="font-bold text-[12px] mt-0.5">
              ${physics.economics.estServingCostPer1MTokUsd < 1
                ? physics.economics.estServingCostPer1MTokUsd.toFixed(2)
                : physics.economics.estServingCostPer1MTokUsd.toFixed(1)}
            </div>
          </div>
          <div className="col-span-2 px-1.5 py-1 rounded flex items-center justify-between" style={{ backgroundColor: 'var(--bg-panel-subtle)', color: 'var(--ink-secondary)' }}>
            <span>Chinchilla Train Est.</span>
            <strong>
              {physics.economics.chinchillaTokensB >= 1000
                ? `${(physics.economics.chinchillaTokensB / 1000).toFixed(1)}T`
                : `${Math.round(physics.economics.chinchillaTokensB)}B`}{' '}
              tok ·{' '}
              {physics.economics.trainingH100Days >= 1000
                ? `${(physics.economics.trainingH100Days / 1000).toFixed(1)}k`
                : Math.max(1, Math.round(physics.economics.trainingH100Days))}{' '}
              H100-days
            </strong>
          </div>
        </div>
      </section>

      {/* ==================== 4. ARCHITECTURAL DESIGN ADVISOR ==================== */}
      <section className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="font-display font-bold text-[11px] uppercase tracking-[0.08em]">
            Architecture Advisor
          </span>
          <span className="text-[10px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
            {diagnostics.length} checks
          </span>
        </div>

        <div className="space-y-1.5">
          {diagnostics.map((item) => {
            const color =
              item.severity === 'critical'
                ? 'var(--stage-output)'
                : item.severity === 'warning'
                  ? 'var(--stage-hybrid)'
                  : item.severity === 'synergy'
                    ? 'var(--stage-ffn)'
                    : 'var(--stage-tokenizer)';
            const Icon =
              item.severity === 'critical' || item.severity === 'warning'
                ? AlertTriangle
                : item.severity === 'synergy'
                  ? CheckCircle2
                  : Info;
            return (
              <div
                key={item.id}
                className="rounded-lg border p-2.5 text-[11px] space-y-1"
                style={{
                  backgroundColor: 'var(--bg-slot)',
                  borderColor: 'var(--border-hairline)',
                  borderLeftWidth: '3px',
                  borderLeftColor: color,
                }}
              >
                <div className="flex items-start gap-1.5 font-semibold leading-snug">
                  <Icon className="w-3.5 h-3.5 shrink-0 mt-0.5" style={{ color }} />
                  <span>{item.title}</span>
                </div>
                <p className="leading-relaxed pl-5" style={{ color: 'var(--ink-secondary)' }}>
                  {item.detail}
                </p>
                {item.suggestion && COMPONENT_MAP[item.suggestion.componentId] && (
                  <div className="pl-5 pt-1">
                    <button
                      type="button"
                      onClick={() =>
                        onApplySuggestion(COMPONENT_MAP[item.suggestion!.componentId])
                      }
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[10.5px] font-semibold cursor-pointer hover:opacity-85 transition-opacity"
                      style={{
                        borderColor: color,
                        color,
                        backgroundColor: 'var(--bg-panel)',
                      }}
                    >
                      <Zap className="w-3 h-3" />
                      {item.suggestion.label}
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* ==================== 5. CLOSEST MODELS IN RASCHKA'S GALLERY (120 MODELS) ==================== */}
      <section className="space-y-2 pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5" style={{ color: 'var(--stage-tokenizer)' }} />
            <span className="font-display font-bold text-[11px] uppercase tracking-[0.08em]">
              Gallery Match Radar
            </span>
          </div>
          <button
            type="button"
            onClick={onOpenGallery}
            className="text-[10.5px] font-medium underline cursor-pointer"
            style={{ color: 'var(--stage-tokenizer)' }}
          >
            All {matches.length} →
          </button>
        </div>

        <div className="space-y-1.5">
          {matches.slice(0, 4).map((m) => (
            <div
              key={m.model.id}
              className="rounded-lg border p-2.5 space-y-1.5 transition-colors hover:border-[var(--border-strong)]"
              style={{
                backgroundColor: 'var(--bg-slot)',
                borderColor: 'var(--border-hairline)',
              }}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="text-[12px] font-semibold leading-tight truncate">
                    {m.model.name}
                  </div>
                  <div className="text-[10px] font-mono-num mt-0.5" style={{ color: 'var(--ink-muted)' }}>
                    {m.model.company} · {m.model.decoderType} · {m.model.kvCachePerTokenRaw || 'N/A'}
                  </div>
                </div>
                <span
                  className="text-[11px] font-mono-num font-bold px-1.5 py-0.5 rounded shrink-0"
                  style={{
                    backgroundColor: 'var(--stage-tokenizer-soft)',
                    color: 'var(--stage-tokenizer)',
                  }}
                >
                  {m.score}%
                </span>
              </div>

              {/* Similarity bar */}
              <div className="h-1 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--border-hairline)' }}>
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${m.score}%`,
                    backgroundColor: 'var(--stage-tokenizer)',
                  }}
                />
              </div>

              {/* Shared concepts + Quick Actions */}
              <div className="flex items-center justify-between gap-1 pt-0.5">
                <div className="flex flex-wrap gap-1 min-w-0">
                  {m.sharedConcepts.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="text-[9px] font-mono-num px-1.5 py-0.2 rounded border"
                      style={{
                        borderColor: 'var(--border-hairline)',
                        color: 'var(--ink-secondary)',
                        backgroundColor: 'var(--bg-panel)',
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <button
                    type="button"
                    onClick={() => onCompareWithModel(m.model)}
                    title="Compare side-by-side"
                    className="p-1 rounded border hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
                    style={{ borderColor: 'var(--border-hairline)', color: 'var(--ink-secondary)' }}
                  >
                    <GitCompare className="w-3 h-3" />
                  </button>
                  <button
                    type="button"
                    onClick={() => onCloneGalleryModel(m.model)}
                    title="Clone onto Blueprint Canvas"
                    className="p-1 rounded border hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
                    style={{ borderColor: 'var(--border-hairline)', color: 'var(--ink-secondary)' }}
                  >
                    <Download className="w-3 h-3" />
                  </button>
                  <a
                    href={m.model.galleryUrl}
                    target="_blank"
                    rel="noreferrer"
                    title="Open in Sebastian Raschka's LLM Architecture Gallery"
                    className="p-1 rounded border hover:bg-[var(--bg-panel-subtle)]"
                    style={{ borderColor: 'var(--border-hairline)', color: 'var(--ink-secondary)' }}
                  >
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Attribution footer */}
        <div
          className="text-[10px] leading-relaxed px-2 py-1.5 rounded border"
          style={{
            backgroundColor: 'var(--bg-panel-subtle)',
            borderColor: 'var(--border-hairline)',
            color: 'var(--ink-muted)',
          }}
        >
          Model catalog grounded in{' '}
          <a
            href="https://sebastianraschka.com/llm-architecture-gallery/"
            target="_blank"
            rel="noreferrer"
            className="underline font-medium"
            style={{ color: 'var(--ink-secondary)' }}
          >
            Sebastian Raschka&apos;s LLM Architecture Gallery
          </a>{' '}
          (Apache 2.0 · {matches.length} open-weight models).
        </div>
      </section>
    </aside>
  );
};
