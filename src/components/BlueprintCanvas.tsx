import React, { useMemo, useState } from 'react';
import {
  ArrowDown,
  Award,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Circle,
  Compass,
  SlidersHorizontal,
  Sparkles,
  Trophy,
  X,
  Zap,
} from 'lucide-react';
import type { EvaluatedChallenge } from '../data/challenges';
import { COMPONENT_MAP, getComponentsByStage, STAGES } from '../data/components';
import {
  computePhysics,
  formatContext,
  formatKiB,
  formatParams,
} from '../engine/physics';
import type {
  ArchitectureConfig,
  ArchitectureDimensions,
  ComponentDef,
  PhysicsReport,
  PrecisionMode,
  StageId,
} from '../types/architecture';
import { LayerStackStrip } from './LayerStackStrip';
import { SchematicDiagram } from './SchematicDiagram';
import { TokenTraceBar } from './TokenTraceBar';

interface BlueprintCanvasProps {
  config: ArchitectureConfig;
  physics: PhysicsReport;
  theme: 'paper' | 'obsidian';
  galleryCount: number;
  draggingComp: ComponentDef | null;
  hoveredDropStage: StageId | null;
  flashedStage: StageId | null;
  pinnedMission: EvaluatedChallenge | null;
  onUnpinMission: () => void;
  showQuickStart: boolean;
  onDismissQuickStart: () => void;
  onOpenMissions: () => void;
  onOpenParetoLab: () => void;
  onSelectStageFilter: (stage: StageId) => void;
  onEquipComponent: (comp: ComponentDef) => void;
  onUpdateDims: (patch: Partial<ArchitectureDimensions>) => void;
  registerDropZone: (stage: StageId, el: HTMLDivElement | null) => void;
}

const CONTEXT_STOPS = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
const DMODEL_STOPS = [1024, 1536, 2048, 2560, 2816, 3072, 4096, 5120, 6144, 7168, 8192];
const PRECISION_OPTIONS: { id: PrecisionMode; label: string }[] = [
  { id: 'fp32', label: 'FP32' },
  { id: 'bf16', label: 'BF16' },
  { id: 'fp8', label: 'FP8' },
  { id: 'int4', label: 'INT4' },
];

export const BlueprintCanvas: React.FC<BlueprintCanvasProps> = ({
  config,
  physics,
  theme,
  galleryCount,
  draggingComp,
  hoveredDropStage,
  flashedStage,
  pinnedMission,
  onUnpinMission,
  showQuickStart,
  onDismissQuickStart,
  onOpenMissions,
  onOpenParetoLab,
  onSelectStageFilter,
  onEquipComponent,
  onUpdateDims,
  registerDropZone,
}) => {
  const [showAdvancedDims, setShowAdvancedDims] = useState(false);
  const [selectedStage, setSelectedStage] = useState<StageId | null>(null);
  const [traceStepIdx, setTraceStepIdx] = useState<number | null>(null);
  const [tracePlaying, setTracePlaying] = useState(false);
  const [tracedStage, setTracedStage] = useState<StageId | null>(null);

  const { dims } = config;
  const isMoE =
    config.ffn === 'ffn_moe_sparse' ||
    config.ffn === 'ffn_moe_shared' ||
    config.ffn === 'ffn_latent_moe' ||
    config.ffn === 'ffn_looped_moe';
  const isMla =
    config.attention === 'attn_mla' ||
    config.attention === 'attn_mla_dsa' ||
    config.attention === 'attn_mla_dsa_loop' ||
    config.attention === 'attn_csa_hca';
  const isGqaLike =
    config.attention === 'attn_gqa' ||
    config.attention === 'attn_gated_gqa' ||
    config.attention === 'attn_kv_sharing';
  const isHybrid = config.hybrid !== 'hyb_none' && config.hybrid !== 'hyb_xlstm';
  const isSwa = config.hybrid === 'hyb_swa' || config.attention === 'attn_mla_dsa_loop';
  const isLooped = config.ffn === 'ffn_looped_depth' || config.ffn === 'ffn_looped_moe';

  // Live hover delta preview when dragging a component over its target stage!
  const dragPreviewDelta = useMemo(() => {
    if (!draggingComp) return null;
    if (config[draggingComp.stage] === draggingComp.id) return null;
    const hypothetical: ArchitectureConfig = {
      ...config,
      [draggingComp.stage]: draggingComp.id,
      dims: { ...config.dims },
    };
    // Apply sensible smart dim tweak for MoE preview if switching from dense
    if (
      draggingComp.stage === 'ffn' &&
      (draggingComp.id === 'ffn_moe_sparse' ||
        draggingComp.id === 'ffn_moe_shared' ||
        draggingComp.id === 'ffn_latent_moe' ||
        draggingComp.id === 'ffn_looped_moe') &&
      hypothetical.dims.nExperts <= 1
    ) {
      hypothetical.dims.nExperts = draggingComp.id === 'ffn_looped_moe' ? 256 : 64;
      hypothetical.dims.activeExperts = draggingComp.id === 'ffn_looped_moe' ? 8 : 6;
      hypothetical.dims.nSharedExperts =
        draggingComp.id === 'ffn_moe_shared' || draggingComp.id === 'ffn_looped_moe' ? 1 : 0;
      hypothetical.dims.dFf = Math.min(hypothetical.dims.dFf, 2048);
    }
    if (
      draggingComp.stage === 'ffn' &&
      (draggingComp.id === 'ffn_looped_depth' || draggingComp.id === 'ffn_looped_moe') &&
      hypothetical.dims.loopPasses < 2
    ) {
      hypothetical.dims.loopPasses = 2;
    }
    const nextPhysics = computePhysics(hypothetical);
    const kvDiff = nextPhysics.kvKiBPerTokenBf16 - physics.kvKiBPerTokenBf16;
    const kvPct =
      physics.kvKiBPerTokenBf16 > 0
        ? Math.round((kvDiff / physics.kvKiBPerTokenBf16) * 100)
        : 0;
    const totalDiff = nextPhysics.params.total - physics.params.total;
    const activeDiff = nextPhysics.params.active - physics.params.active;
    return {
      stage: draggingComp.stage,
      nextKv: nextPhysics.kvKiBPerTokenBf16,
      kvDiff,
      kvPct,
      nextTotal: nextPhysics.params.total,
      nextActive: nextPhysics.params.active,
      totalDiff,
      activeDiff,
    };
  }, [draggingComp, config, physics]);

  const cycleStageComponent = (stage: StageId, dir: -1 | 1) => {
    const list = getComponentsByStage(stage);
    const idx = list.findIndex((c) => c.id === config[stage]);
    const nextIdx = (idx + dir + list.length) % list.length;
    onEquipComponent(list[nextIdx]);
  };

  const ctxIndex = Math.max(
    0,
    CONTEXT_STOPS.findIndex((c) => c >= dims.contextLength)
  );
  const dModelIndex = Math.max(
    0,
    DMODEL_STOPS.findIndex((d) => d >= dims.dModel)
  );

  return (
    <main className="flex-1 h-full overflow-y-auto blueprint-grid p-3.5 lg:p-4 space-y-3.5">
      {/* ==================== OPTIONAL FIRST-VISIT QUICK-START STRIP ==================== */}
      {showQuickStart && !pinnedMission && (
        <section
          className="rounded-lg border px-3.5 py-2.5 flex flex-wrap items-center justify-between gap-2.5"
          style={{
            backgroundColor: 'var(--bg-panel)',
            borderColor: 'var(--stage-attention)',
            boxShadow: 'var(--shadow-card)',
          }}
        >
          <div className="flex items-center gap-2 text-[12px]">
            <Sparkles className="w-4 h-4 shrink-0" style={{ color: 'var(--stage-attention)' }} />
            <span>
              <strong>New here?</strong> Drag any block from the left onto the blueprint — or try an instant experiment:
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              onClick={() => onEquipComponent(COMPONENT_MAP['attn_mla'])}
              className="px-2.5 py-1 rounded-md border text-[11px] font-semibold cursor-pointer hover:opacity-85"
              style={{
                backgroundColor: 'var(--stage-attention-soft)',
                borderColor: 'var(--stage-attention)',
                color: 'var(--stage-attention)',
              }}
            >
              ⚡ Swap Attention → MLA
            </button>
            <button
              type="button"
              onClick={onOpenMissions}
              className="flex items-center gap-1 px-2.5 py-1 rounded-md border text-[11px] font-semibold cursor-pointer hover:bg-[var(--bg-panel-subtle)]"
              style={{ borderColor: 'var(--border-hairline)', color: 'var(--ink-primary)' }}
            >
              <Trophy className="w-3 h-3" style={{ color: 'var(--stage-hybrid)' }} />
              Try 8 Design Missions
            </button>
            <button
              type="button"
              onClick={onOpenParetoLab}
              className="flex items-center gap-1 px-2.5 py-1 rounded-md border text-[11px] font-semibold cursor-pointer hover:bg-[var(--bg-panel-subtle)]"
              style={{ borderColor: 'var(--border-hairline)', color: 'var(--ink-primary)' }}
            >
              <Compass className="w-3 h-3" style={{ color: 'var(--stage-tokenizer)' }} />
              {galleryCount}-Model Pareto Map
            </button>
            <button
              type="button"
              onClick={onDismissQuickStart}
              className="p-1 rounded hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              style={{ color: 'var(--ink-muted)' }}
              title="Dismiss quick-start strip"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </section>
      )}

      {/* ==================== PINNED MISSION LIVE CHECKLIST HUD ==================== */}
      {pinnedMission && (
        <section
          className="rounded-lg border p-3 space-y-2"
          style={{
            backgroundColor: pinnedMission.completed ? 'var(--stage-ffn-soft)' : 'var(--bg-panel)',
            borderColor: pinnedMission.completed ? 'var(--stage-ffn)' : 'var(--stage-hybrid)',
            boxShadow: 'var(--shadow-card)',
          }}
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              {pinnedMission.completed ? (
                <Award className="w-4 h-4" style={{ color: 'var(--stage-ffn)' }} />
              ) : (
                <Trophy className="w-4 h-4" style={{ color: 'var(--stage-hybrid)' }} />
              )}
              <span className="font-display font-bold text-[12.5px]">
                Mission {pinnedMission.challenge.number}: {pinnedMission.challenge.title}
              </span>
              <span
                className="text-[10px] font-mono-num font-bold px-1.5 py-0.5 rounded"
                style={{
                  backgroundColor: pinnedMission.completed ? 'var(--stage-ffn)' : 'var(--stage-hybrid-soft)',
                  color: pinnedMission.completed ? '#fff' : 'var(--stage-hybrid)',
                }}
              >
                {pinnedMission.completed
                  ? 'MISSION CLEARED!'
                  : `${pinnedMission.passedCount} / ${pinnedMission.totalCount} MET`}
              </span>
            </div>
            <button
              type="button"
              onClick={onUnpinMission}
              className="text-[11px] font-medium flex items-center gap-1 px-2 py-0.5 rounded hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              style={{ color: 'var(--ink-muted)' }}
            >
              Unpin <X className="w-3 h-3" />
            </button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-1.5">
            {pinnedMission.constraints.map((c) => (
              <div
                key={c.id}
                className="flex items-center justify-between gap-1.5 px-2 py-1 rounded border text-[10.5px]"
                style={{
                  backgroundColor: 'var(--bg-panel)',
                  borderColor: c.passed ? 'var(--stage-ffn)' : 'var(--border-hairline)',
                }}
              >
                <span className="flex items-center gap-1 truncate">
                  {c.passed ? (
                    <CheckCircle2 className="w-3 h-3 shrink-0" style={{ color: 'var(--stage-ffn)' }} />
                  ) : (
                    <Circle className="w-3 h-3 shrink-0 opacity-35" />
                  )}
                  <span className="truncate">{c.label}</span>
                </span>
                <span className="font-mono-num font-semibold shrink-0" style={{ color: c.passed ? 'var(--stage-ffn)' : 'var(--ink-muted)' }}>
                  {c.currentText}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ==================== TOP DIMENSIONS & SCALING STRIP ==================== */}
      <section
        className="rounded-lg border p-3 space-y-2.5"
        style={{
          backgroundColor: 'var(--bg-panel)',
          borderColor: 'var(--border-hairline)',
          boxShadow: 'var(--shadow-card)',
        }}
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-3.5 h-3.5" style={{ color: 'var(--ink-muted)' }} />
            <span className="font-display font-bold text-[11.5px] uppercase tracking-[0.08em]">
              Model Geometry & Deployment Target
            </span>
            <span
              className="text-[10.5px] font-mono-num px-2 py-0.5 rounded border"
              style={{
                backgroundColor: 'var(--bg-panel-subtle)',
                borderColor: 'var(--border-hairline)',
                color: 'var(--ink-secondary)',
              }}
            >
              d={dims.dModel} · L={dims.nLayers}
              {physics.loopBreakdown ? ` (${physics.effectiveDepth} eff)` : ''} · H={dims.nHeads} (d_h={physics.dHead}) · Ctx={formatContext(dims.contextLength)}
            </span>
          </div>

          <div className="flex items-center gap-2">
            {/* Precision Selector */}
            <div
              className="flex items-center rounded border p-0.5"
              style={{
                backgroundColor: 'var(--bg-panel-subtle)',
                borderColor: 'var(--border-hairline)',
              }}
            >
              {PRECISION_OPTIONS.map((p) => {
                const active = dims.precision === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => onUpdateDims({ precision: p.id })}
                    className="px-2 py-0.5 rounded text-[10px] font-mono-num font-semibold transition-all cursor-pointer"
                    style={
                      active
                        ? {
                            backgroundColor: 'var(--ink-primary)',
                            color: 'var(--ink-inverse)',
                          }
                        : { color: 'var(--ink-secondary)' }
                    }
                  >
                    {p.label}
                  </button>
                );
              })}
            </div>

            <button
              type="button"
              onClick={() => setShowAdvancedDims((s) => !s)}
              className="text-[11px] font-medium px-2 py-1 rounded border hover:bg-[var(--bg-panel-subtle)] cursor-pointer transition-colors"
              style={{
                borderColor: 'var(--border-hairline)',
                color: 'var(--ink-secondary)',
              }}
            >
              {showAdvancedDims ? 'Compact' : 'Fine-Tune'}
            </button>
          </div>
        </div>

        {/* Core 4 Sliders Always Visible */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 pt-0.5">
          {/* Hidden Dim d_model */}
          <label className="space-y-1 block">
            <div className="flex justify-between text-[11px]">
              <span style={{ color: 'var(--ink-secondary)' }}>Hidden Dim (d_model)</span>
              <span className="font-mono-num font-semibold">{dims.dModel}</span>
            </div>
            <input
              type="range"
              min={0}
              max={DMODEL_STOPS.length - 1}
              step={1}
              value={dModelIndex}
              onChange={(e) => {
                const nextD = DMODEL_STOPS[Number(e.target.value)];
                const nextHeads = Math.max(8, Math.round(nextD / 128));
                onUpdateDims({
                  dModel: nextD,
                  nHeads: nextHeads,
                  nKvHeads: Math.min(dims.nKvHeads, nextHeads),
                  dFf: isMoE ? dims.dFf : Math.round((nextD * 3.5) / 256) * 256,
                });
              }}
              className="w-full blueprint-slider"
            />
          </label>

          {/* Layers L */}
          <label className="space-y-1 block">
            <div className="flex justify-between text-[11px]">
              <span style={{ color: 'var(--ink-secondary)' }}>Decoder Layers (L)</span>
              <span className="font-mono-num font-semibold">{dims.nLayers}</span>
            </div>
            <input
              type="range"
              min={12}
              max={96}
              step={2}
              value={dims.nLayers}
              onChange={(e) => onUpdateDims({ nLayers: Number(e.target.value) })}
              className="w-full blueprint-slider"
            />
          </label>

          {/* Context Length */}
          <label className="space-y-1 block">
            <div className="flex justify-between text-[11px]">
              <span style={{ color: 'var(--ink-secondary)' }}>Context Window</span>
              <span className="font-mono-num font-semibold" style={{ color: 'var(--stage-attention)' }}>
                {formatContext(dims.contextLength)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={CONTEXT_STOPS.length - 1}
              step={1}
              value={ctxIndex}
              onChange={(e) =>
                onUpdateDims({ contextLength: CONTEXT_STOPS[Number(e.target.value)] })
              }
              className="w-full blueprint-slider"
            />
          </label>

          {/* Contextual 4th Slider (MoE Experts OR KV Heads OR MLA Rank) */}
          {isMoE ? (
            <label className="space-y-1 block">
              <div className="flex justify-between text-[11px]">
                <span style={{ color: 'var(--ink-secondary)' }}>MoE Experts (Act/Tot)</span>
                <span className="font-mono-num font-semibold" style={{ color: 'var(--stage-ffn)' }}>
                  {dims.activeExperts} / {dims.nExperts}
                </span>
              </div>
              <input
                type="range"
                min={8}
                max={384}
                step={8}
                value={dims.nExperts}
                onChange={(e) => {
                  const nExp = Number(e.target.value);
                  onUpdateDims({
                    nExperts: nExp,
                    activeExperts: Math.min(dims.activeExperts, Math.max(2, Math.floor(nExp / 4))),
                  });
                }}
                className="w-full blueprint-slider"
              />
            </label>
          ) : isMla ? (
            <label className="space-y-1 block">
              <div className="flex justify-between text-[11px]">
                <span style={{ color: 'var(--ink-secondary)' }}>MLA Latent Rank (d_c)</span>
                <span className="font-mono-num font-semibold" style={{ color: 'var(--stage-attention)' }}>
                  {dims.mlaKvLoraRank}
                </span>
              </div>
              <input
                type="range"
                min={128}
                max={1024}
                step={64}
                value={dims.mlaKvLoraRank}
                onChange={(e) => onUpdateDims({ mlaKvLoraRank: Number(e.target.value) })}
                className="w-full blueprint-slider"
              />
            </label>
          ) : (
            <label className="space-y-1 block">
              <div className="flex justify-between text-[11px]">
                <span style={{ color: 'var(--ink-secondary)' }}>KV Heads (GQA)</span>
                <span className="font-mono-num font-semibold">
                  {config.attention === 'attn_mha'
                    ? `${dims.nHeads} (1:1)`
                    : config.attention === 'attn_mqa'
                      ? '1 (MQA)'
                      : `${dims.nKvHeads} (${Math.max(1, Math.round(dims.nHeads / dims.nKvHeads))}:1)`}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={Math.min(32, dims.nHeads)}
                step={1}
                disabled={!isGqaLike}
                value={isGqaLike ? dims.nKvHeads : config.attention === 'attn_mqa' ? 1 : dims.nHeads}
                onChange={(e) => onUpdateDims({ nKvHeads: Number(e.target.value) })}
                className="w-full blueprint-slider disabled:opacity-40"
              />
            </label>
          )}
        </div>

        {/* Expandable Fine-Tuning Controls */}
        {showAdvancedDims && (
          <div
            className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 pt-2.5 border-t"
            style={{ borderColor: 'var(--border-hairline)' }}
          >
            <label className="space-y-1 block">
              <div className="flex justify-between text-[10.5px]">
                <span style={{ color: 'var(--ink-muted)' }}>Query Heads</span>
                <span className="font-mono-num font-semibold">{dims.nHeads}</span>
              </div>
              <input
                type="range"
                min={8}
                max={128}
                step={4}
                value={dims.nHeads}
                onChange={(e) => {
                  const h = Number(e.target.value);
                  onUpdateDims({ nHeads: h, nKvHeads: Math.min(dims.nKvHeads, h) });
                }}
                className="w-full blueprint-slider"
              />
            </label>

            <label className="space-y-1 block">
              <div className="flex justify-between text-[10.5px]">
                <span style={{ color: 'var(--ink-muted)' }}>{isMoE ? 'Expert d_ff' : 'FFN d_ff'}</span>
                <span className="font-mono-num font-semibold">{dims.dFf.toLocaleString()}</span>
              </div>
              <input
                type="range"
                min={512}
                max={32768}
                step={512}
                value={dims.dFf}
                onChange={(e) => onUpdateDims({ dFf: Number(e.target.value) })}
                className="w-full blueprint-slider"
              />
            </label>

            {isMoE && (
              <label className="space-y-1 block">
                <div className="flex justify-between text-[10.5px]">
                  <span style={{ color: 'var(--ink-muted)' }}>Top-k Active</span>
                  <span className="font-mono-num font-semibold">{dims.activeExperts}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={Math.min(16, dims.nExperts)}
                  step={1}
                  value={dims.activeExperts}
                  onChange={(e) => onUpdateDims({ activeExperts: Number(e.target.value) })}
                  className="w-full blueprint-slider"
                />
              </label>
            )}

            {isHybrid && (
              <label className="space-y-1 block">
                <div className="flex justify-between text-[10.5px]">
                  <span style={{ color: 'var(--ink-muted)' }}>Hybrid Ratio</span>
                  <span className="font-mono-num font-semibold">{dims.hybridRatio}:1</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={7}
                  step={1}
                  value={dims.hybridRatio}
                  onChange={(e) => onUpdateDims({ hybridRatio: Number(e.target.value) })}
                  className="w-full blueprint-slider"
                />
              </label>
            )}

            {isSwa && (
              <label className="space-y-1 block">
                <div className="flex justify-between text-[10.5px]">
                  <span style={{ color: 'var(--ink-muted)' }}>SWA Window</span>
                  <span className="font-mono-num font-semibold">{formatContext(dims.swaWindow)}</span>
                </div>
                <input
                  type="range"
                  min={512}
                  max={16384}
                  step={512}
                  value={dims.swaWindow}
                  onChange={(e) => onUpdateDims({ swaWindow: Number(e.target.value) })}
                  className="w-full blueprint-slider"
                />
              </label>
            )}

            {isLooped && (
              <label className="space-y-1 block">
                <div className="flex justify-between text-[10.5px]">
                  <span style={{ color: 'var(--ink-muted)' }}>Loop Passes</span>
                  <span className="font-mono-num font-semibold">×{dims.loopPasses}</span>
                </div>
                <input
                  type="range"
                  min={2}
                  max={4}
                  step={1}
                  value={dims.loopPasses}
                  onChange={(e) => onUpdateDims({ loopPasses: Number(e.target.value) })}
                  className="w-full blueprint-slider"
                />
              </label>
            )}

            <label className="space-y-1 block">
              <div className="flex justify-between text-[10.5px]">
                <span style={{ color: 'var(--ink-muted)' }}>Batch Size</span>
                <span className="font-mono-num font-semibold">B={dims.batchSize}</span>
              </div>
              <input
                type="range"
                min={1}
                max={64}
                step={1}
                value={dims.batchSize}
                onChange={(e) => onUpdateDims({ batchSize: Number(e.target.value) })}
                className="w-full blueprint-slider"
              />
            </label>
          </div>
        )}
      </section>

      {/* ==================== LIVE TOKEN FORWARD-PASS TRACER ==================== */}
      <TokenTraceBar
        steps={physics.traceSteps}
        activeStepIdx={traceStepIdx}
        isPlaying={tracePlaying}
        onStart={() => {
          setTraceStepIdx(0);
          setTracePlaying(true);
        }}
        onStop={() => {
          setTracePlaying(false);
          setTraceStepIdx(null);
          setTracedStage(null);
        }}
        onTogglePlay={() => setTracePlaying((p) => !p)}
        onStepChange={(idx) => setTraceStepIdx(idx)}
        onHighlightStage={setTracedStage}
      />

      {/* ==================== MAIN SPLIT: INTERACTIVE DROP SLOTS + LIVE SCHEMATIC ==================== */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3.5 items-start">
        {/* LEFT COLUMN (7/12): 7 INTERACTIVE DRAG-AND-DROP PIPELINE SLOTS */}
        <section className="xl:col-span-7 space-y-1.5">
          <div className="flex items-center justify-between px-1 pb-0.5">
            <div className="flex items-center gap-2">
              <span className="font-display font-bold text-[11.5px] uppercase tracking-[0.08em]">
                Interactive Decoder Pipeline
              </span>
              <span
                className="text-[10px] font-mono-num px-1.5 py-0.5 rounded"
                style={{
                  backgroundColor: 'var(--bg-panel)',
                  color: 'var(--ink-muted)',
                  border: '1px solid var(--border-hairline)',
                }}
              >
                DROP MODULES INTO ANY STAGE
              </span>
            </div>
            {draggingComp && (
              <span
                className="text-[11px] font-semibold flex items-center gap-1 animate-drop-beacon"
                style={{ color: 'var(--stage-attention)' }}
              >
                <Sparkles className="w-3.5 h-3.5" />
                Release on {STAGES.find((s) => s.id === draggingComp.stage)?.label}
              </span>
            )}
          </div>

          {STAGES.map((stage, idx) => {
            const equipped = COMPONENT_MAP[config[stage.id]];
            const isTargetForDrag = draggingComp?.stage === stage.id;
            const isHoveredDrop = hoveredDropStage === stage.id && isTargetForDrag;
            const isFlashed = flashedStage === stage.id;
            const isTraced = tracedStage === stage.id;
            const isDimmedByDrag = draggingComp !== null && !isTargetForDrag;
            const showPreview = isTargetForDrag && dragPreviewDelta !== null;

            return (
              <React.Fragment key={stage.id}>
                <div
                  ref={(el) => registerDropZone(stage.id, el)}
                  onClick={() => {
                    setSelectedStage(stage.id);
                    onSelectStageFilter(stage.id);
                  }}
                  className={`rounded-lg border transition-all relative cursor-pointer ${
                    isFlashed ? 'animate-slot-flash' : ''
                  }`}
                  style={{
                    backgroundColor: isHoveredDrop || isTraced
                      ? stage.softVar
                      : 'var(--bg-panel)',
                    borderColor: isHoveredDrop || isTraced
                      ? stage.cssVar
                      : isTargetForDrag
                        ? stage.cssVar
                        : selectedStage === stage.id
                          ? 'var(--border-strong)'
                          : 'var(--border-hairline)',
                    borderStyle: isTargetForDrag ? 'dashed' : 'solid',
                    borderWidth: isTargetForDrag || isTraced ? '2px' : '1px',
                    opacity: isDimmedByDrag ? 0.55 : 1,
                    transform: isHoveredDrop ? 'scale(1.012)' : undefined,
                    boxShadow: isHoveredDrop ? 'var(--shadow-float)' : 'var(--shadow-card)',
                    ['--flash-color' as string]: stage.cssVar,
                  }}
                >
                  {/* Stage Vertical Color Accent Bar */}
                  <div
                    className="absolute left-0 top-2 bottom-2 w-[3.5px] rounded-r"
                    style={{ backgroundColor: stage.cssVar }}
                  />

                  <div className="pl-3.5 pr-3 py-2.5">
                    {/* Top Row: Stage Badge + Quick Cycle Controls */}
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <span
                          className="text-[10px] font-mono-num font-bold px-1.5 py-0.5 rounded uppercase shrink-0"
                          style={{
                            backgroundColor: stage.softVar,
                            color: stage.cssVar,
                          }}
                        >
                          0{stage.order} · {stage.shortLabel}
                        </span>
                        <h3 className="font-display font-bold text-[13.5px] truncate">
                          {equipped?.name}
                        </h3>
                      </div>

                      <div className="flex items-center gap-1.5 shrink-0">
                        <span
                          className="text-[10.5px] font-mono-num font-semibold px-2 py-0.5 rounded border hidden sm:inline-block"
                          style={{
                            backgroundColor: 'var(--bg-slot)',
                            borderColor: 'var(--border-hairline)',
                            color: stage.cssVar,
                          }}
                        >
                          {equipped?.badge}
                        </span>

                        {/* Instant A/B Stepper Buttons */}
                        <div
                          className="flex items-center rounded border overflow-hidden"
                          style={{ borderColor: 'var(--border-hairline)' }}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            type="button"
                            onClick={() => cycleStageComponent(stage.id, -1)}
                            title={`Previous ${stage.shortLabel}`}
                            className="p-1 hover:bg-[var(--bg-panel-subtle)] cursor-pointer transition-colors"
                          >
                            <ChevronLeft className="w-3.5 h-3.5" />
                          </button>
                          <div className="w-px h-3.5" style={{ backgroundColor: 'var(--border-hairline)' }} />
                          <button
                            type="button"
                            onClick={() => cycleStageComponent(stage.id, 1)}
                            title={`Next ${stage.shortLabel}`}
                            className="p-1 hover:bg-[var(--bg-panel-subtle)] cursor-pointer transition-colors"
                          >
                            <ChevronRight className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Middle Row: Equation + Tagline */}
                    <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2">
                      <p className="text-[11.5px] leading-snug" style={{ color: 'var(--ink-secondary)' }}>
                        {equipped?.tagline}
                      </p>
                      <code
                        className="text-[10px] font-mono-num px-1.5 py-0.5 rounded border max-w-full truncate"
                        style={{
                          backgroundColor: 'var(--bg-slot)',
                          borderColor: 'var(--border-hairline)',
                          color: 'var(--ink-muted)',
                        }}
                      >
                        {equipped?.formula}
                      </code>
                    </div>

                    {/* LIVE DRAG-HOVER DELTA PREVIEW BANNER! */}
                    {showPreview && dragPreviewDelta && (
                      <div
                        className="mt-2 px-2.5 py-1.5 rounded-md border flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono-num"
                        style={{
                          backgroundColor: 'var(--bg-panel)',
                          borderColor: stage.cssVar,
                        }}
                      >
                        <span className="font-semibold flex items-center gap-1.5" style={{ color: stage.cssVar }}>
                          <Zap className="w-3.5 h-3.5" />
                          {isHoveredDrop ? 'Release to snap' : 'Drop here'}: {draggingComp.shortName}
                        </span>
                        <div className="flex items-center gap-3">
                          <span>
                            KV:{' '}
                            <strong
                              style={{
                                color:
                                  dragPreviewDelta.kvDiff < 0
                                    ? 'var(--stage-ffn)'
                                    : dragPreviewDelta.kvDiff > 0
                                      ? 'var(--stage-output)'
                                      : 'var(--ink-primary)',
                              }}
                            >
                              {formatKiB(dragPreviewDelta.nextKv)}
                              {dragPreviewDelta.kvDiff !== 0 &&
                                ` (${dragPreviewDelta.kvPct > 0 ? '+' : ''}${dragPreviewDelta.kvPct}%)`}
                            </strong>
                          </span>
                          <span>
                            Active:{' '}
                            <strong>{formatParams(dragPreviewDelta.nextActive)}</strong>
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Live Tensor Bus Conduit between Stages */}
                {idx < STAGES.length - 1 && (
                  <div className="flex items-center justify-center gap-2 py-0.5 select-none">
                    <div className="h-px flex-1 max-w-[80px]" style={{ backgroundColor: 'var(--border-hairline)' }} />
                    <div
                      className="flex items-center gap-1 px-2 py-0.5 rounded-full border text-[9.5px] font-mono-num"
                      style={{
                        backgroundColor: 'var(--bg-panel)',
                        borderColor: 'var(--border-hairline)',
                        color: 'var(--ink-muted)',
                      }}
                    >
                      <ArrowDown className="w-2.5 h-2.5" />
                      <span>
                        {stage.id === 'tokenizer'
                          ? `[${dims.batchSize}, T] → [${dims.batchSize}, T, ${dims.dModel}]`
                          : stage.id === 'attention'
                            ? `[${dims.batchSize}, ${dims.nHeads}, T, ${physics.dHead}] · KV ${formatKiB(physics.kvKiBPerTokenBf16)}`
                            : stage.id === 'ffn'
                              ? `[${dims.batchSize}, T, ${dims.dModel}] · ${physics.params.activeRatioPct.toFixed(0)}% Act`
                              : stage.tensorOut}
                      </span>
                    </div>
                    <div className="h-px flex-1 max-w-[80px]" style={{ backgroundColor: 'var(--border-hairline)' }} />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </section>

        {/* RIGHT COLUMN (5/12): LIVE RASCHKA-STYLE VECTOR SCHEMATIC */}
        <section className="xl:col-span-5 sticky top-0">
          <SchematicDiagram
            config={config}
            physics={physics}
            theme={theme}
            highlightedStage={tracedStage ?? hoveredDropStage ?? selectedStage}
            onSelectStage={(stage) => {
              setSelectedStage(stage);
              onSelectStageFilter(stage);
            }}
          />
        </section>
      </div>

      {/* ==================== BOTTOM LAYER-BY-LAYER SCHEDULE RIBBON ==================== */}
      <LayerStackStrip physics={physics} />
    </main>
  );
};
