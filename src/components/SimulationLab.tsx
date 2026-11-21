import React, { useMemo, useState } from 'react';
import { ArrowRightLeft, Eye, Layers, Sparkles, Type } from 'lucide-react';
import { COMPONENT_MAP, getComponentsByStage } from '../data/components';
import { SAMPLE_PROMPTS, simulateTokenization } from '../data/tokenizerSamples';
import { formatContext, formatParams } from '../engine/physics';
import type {
  ArchitectureConfig,
  ComponentDef,
  GalleryModel,
  PhysicsReport,
} from '../types/architecture';
import { ParetoFrontierChart } from './ParetoFrontierChart';

interface SimulationLabProps {
  config: ArchitectureConfig;
  physics: PhysicsReport;
  galleryModels: GalleryModel[];
  onEquipComponent: (comp: ComponentDef) => void;
  onCloneModel: (model: GalleryModel) => void;
}

const DEMO_TOKENS = [
  '<bos>',
  'def',
  ' mla',
  '_attn',
  '(',
  'q',
  ',',
  ' c',
  '_kv',
  ',',
  ' k',
  '_r',
  '):',
  ' return',
  ' sdpa',
  '(q)',
];

const TOKEN_PILL_BG = [
  'rgba(37, 99, 235, 0.14)',
  'rgba(234, 88, 12, 0.14)',
  'rgba(5, 150, 105, 0.14)',
  'rgba(13, 148, 136, 0.14)',
  'rgba(79, 70, 229, 0.14)',
  'rgba(217, 119, 6, 0.15)',
  'rgba(225, 29, 72, 0.13)',
];

export const SimulationLab: React.FC<SimulationLabProps> = ({
  config,
  physics,
  galleryModels,
  onEquipComponent,
  onCloneModel,
}) => {
  const [hoveredRow, setHoveredRow] = useState<number>(13);
  const [layerViewMode, setLayerViewMode] = useState<'primary' | 'secondary'>('primary');
  const [sampleText, setSampleText] = useState(SAMPLE_PROMPTS[0].text);
  const [compareTokId, setCompareTokId] = useState<string>(
    config.tokenizer === 'tok_gpt2_50k' ? 'tok_qwen_248k' : 'tok_gpt2_50k'
  );
  const [hoveredTokIdx, setHoveredTokIdx] = useState<number | null>(null);

  const equippedTok = COMPONENT_MAP[config.tokenizer];
  const compareTok = COMPONENT_MAP[compareTokId] ?? COMPONENT_MAP['tok_gpt2_50k'];
  const allTokenizers = useMemo(() => getComponentsByStage('tokenizer'), []);

  const equippedResult = useMemo(
    () => simulateTokenization(sampleText, equippedTok),
    [sampleText, equippedTok]
  );
  const compareResult = useMemo(
    () => simulateTokenization(sampleText, compareTok),
    [sampleText, compareTok]
  );

  const hasHybridLayer = config.hybrid !== 'hyb_none';
  const isPureRecurrent = config.hybrid === 'hyb_xlstm';
  const effectiveView = isPureRecurrent
    ? 'secondary'
    : hasHybridLayer
      ? layerViewMode
      : 'primary';

  const hasSinks = config.output === 'out_sinks_mtp';
  const isSparseDsa =
    config.attention === 'attn_mla_dsa' || config.attention === 'attn_csa_hca';
  const isNoPe = config.position === 'pos_pure_nope';
  const isGatedAttn = config.attention === 'attn_gated_gqa';

  // Compute 16x16 attention / receptive field matrix reflecting exact equipped architecture
  const matrix = useMemo(() => {
    const N = DEMO_TOKENS.length;
    const rows: number[][] = [];

    for (let i = 0; i < N; i++) {
      const row = new Array<number>(N).fill(0);
      for (let j = 0; j <= i; j++) {
        const dist = i - j;

        if (effectiveView === 'secondary') {
          if (config.hybrid === 'hyb_swa') {
            // Local sliding window of 5 demo tokens
            if (dist <= 4) {
              row[j] = Math.exp(-dist * 0.25);
            } else if (hasSinks && j === 0) {
              row[j] = 0.45;
            }
          } else if (config.hybrid === 'hyb_shortconv') {
            // Causal kernel width 4
            if (dist <= 3) row[j] = 1 - dist * 0.22;
          } else {
            // DeltaNet / Kimi KDA / Mamba-2 / mLSTM: exponential state decay + key overwrite
            const decay = config.hybrid === 'hyb_mamba2' ? 0.22 : 0.18;
            row[j] = Math.exp(-dist * decay) * (j === 1 || j === 7 ? 1.15 : 0.85);
          }
        } else {
          // Primary Attention Layer
          if (isSparseDsa) {
            // Lightning Top-k sparse block selection (local 3 + semantic key tokens 0, 2, 7)
            const inTopK = dist <= 2 || j === 0 || j === 2 || j === 7 || (i >= 10 && j === 8);
            row[j] = inTopK ? (dist <= 2 ? 0.9 - dist * 0.15 : 0.72) : 0.04;
          } else {
            // Full causal attention (MHA / GQA / MLA / Gated GQA)
            const decay = isNoPe ? 0.05 : 0.13;
            let val = Math.exp(-dist * decay);
            if (j === 0 && !isGatedAttn) val += hasSinks ? 0.85 : 0.35; // Attention sink effect unless Gated GQA eliminates it!
            if (j === 7 && i >= 12) val += 0.4; // semantic recall anchor ('c_kv')
            row[j] = val;
          }
        }
      }
      // Normalize row to sum to 1 for realistic probability weights
      const sum = row.reduce((a, b) => a + b, 0) || 1;
      rows.push(row.map((v) => v / sum));
    }
    return rows;
  }, [effectiveView, config.hybrid, isSparseDsa, isNoPe, hasSinks, isGatedAttn]);

  const complexityInfo = useMemo(() => {
    if (effectiveView === 'secondary') {
      if (config.hybrid === 'hyb_swa') {
        return {
          title: `Sliding-Window Attention (W = ${formatContext(config.dims.swaWindow)})`,
          complexity: 'O(T · W)',
          kvBehavior: 'Capped at W tokens',
          desc: 'Each query attends only within its local sliding window, keeping KV-cache constant past W tokens while global layers stitch long-range context.',
        };
      }
      if (config.hybrid === 'hyb_shortconv') {
        return {
          title: 'Causal ShortConv / LIV Block',
          complexity: 'O(T · k)',
          kvBehavior: '0 B/tok KV',
          desc: 'Ultra-fast local causal convolution mixes immediate neighbor tokens with zero growing KV cache.',
        };
      }
      return {
        title: `${COMPONENT_MAP[config.hybrid]?.name} (Recurrent State)`,
        complexity: 'O(T) Linear',
        kvBehavior: 'O(1) Fixed Matrix State',
        desc: 'Compresses token history into a fixed-size gated recurrent matrix state with smooth exponential decay — zero growing KV cache!',
      };
    }
    if (isSparseDsa) {
      return {
        title: `${COMPONENT_MAP[config.attention]?.name} (Top-k Sparse Blocks)`,
        complexity: 'O(T · k_blocks)',
        kvBehavior: 'Compressed Latent + Sparse Index',
        desc: 'A lightweight indexer selects only the top-k highest-scoring past token blocks (bright cells) while skipping irrelevant history (dimmed cells).',
      };
    }
    return {
      title: `${COMPONENT_MAP[config.attention]?.name}${isGatedAttn ? ' (Sink-Free Gate σ)' : ''}`,
      complexity: 'O(T²) Global',
      kvBehavior:
        config.attention === 'attn_mla'
          ? 'Compressed c_KV Latent'
          : config.attention === 'attn_mha'
            ? 'Full H × d_h KV'
            : 'Shared KV Groups',
      desc: isGatedAttn
        ? 'Full causal global recall with output sigmoid gating — notice column 0 (<bos>) no longer hoards attention-sink probability mass!'
        : hasSinks
          ? 'Full causal attention with dedicated learned sink mass parked cleanly on column 0 for long-context streaming stability.'
          : 'Full causal lower-triangular attention where every token can directly attend to every preceding token in the sequence.',
    };
  }, [effectiveView, config.hybrid, config.attention, config.dims.swaWindow, isSparseDsa, isGatedAttn, hasSinks]);

  const tokenDiffPct =
    compareResult.tokenCount > 0
      ? Math.round(
          ((equippedResult.tokenCount - compareResult.tokenCount) / compareResult.tokenCount) * 100
        )
      : 0;

  return (
    <main className="flex-1 h-full overflow-y-auto blueprint-grid p-4 space-y-4">
      <div className="max-w-[1160px] mx-auto space-y-4">
        {/* ==================== 0. 120-MODEL PARETO FRONTIER & ADOPTION ==================== */}
        <ParetoFrontierChart
          config={config}
          physics={physics}
          galleryModels={galleryModels}
          onCloneModel={onCloneModel}
        />

        {/* ==================== 1. INTERACTIVE ATTENTION RECEPTIVE FIELD MATRIX ==================== */}
        <section
          className="rounded-lg border p-4 space-y-3"
          style={{
            backgroundColor: 'var(--bg-panel)',
            borderColor: 'var(--border-hairline)',
            boxShadow: 'var(--shadow-card)',
          }}
        >
          <div className="flex flex-wrap items-center justify-between gap-2 border-b pb-3" style={{ borderColor: 'var(--border-hairline)' }}>
            <div>
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4" style={{ color: 'var(--stage-attention)' }} />
                <h2 className="font-display font-bold text-[14px]">
                  Live Attention &amp; Receptive Field Matrix
                </h2>
                <span
                  className="text-[10.5px] font-mono-num font-semibold px-2 py-0.5 rounded"
                  style={{
                    backgroundColor: 'var(--stage-attention-soft)',
                    color: 'var(--stage-attention)',
                  }}
                >
                  {complexityInfo.complexity}
                </span>
              </div>
              <p className="text-[11.5px] mt-0.5" style={{ color: 'var(--ink-secondary)' }}>
                {complexityInfo.title} — hover any Query row to inspect live causal routing weights.
              </p>
            </div>

            {hasHybridLayer && !isPureRecurrent && (
              <div
                className="flex items-center p-0.5 rounded-md border"
                style={{
                  backgroundColor: 'var(--bg-panel-subtle)',
                  borderColor: 'var(--border-hairline)',
                }}
              >
                <button
                  type="button"
                  onClick={() => setLayerViewMode('primary')}
                  className="px-2.5 py-1 rounded text-[11px] font-medium cursor-pointer transition-colors"
                  style={
                    effectiveView === 'primary'
                      ? { backgroundColor: 'var(--bg-panel)', color: 'var(--stage-attention)', boxShadow: 'var(--shadow-card)' }
                      : { color: 'var(--ink-secondary)' }
                  }
                >
                  Global Layer (1×)
                </button>
                <button
                  type="button"
                  onClick={() => setLayerViewMode('secondary')}
                  className="px-2.5 py-1 rounded text-[11px] font-medium cursor-pointer transition-colors"
                  style={
                    effectiveView === 'secondary'
                      ? { backgroundColor: 'var(--bg-panel)', color: 'var(--stage-hybrid)', boxShadow: 'var(--shadow-card)' }
                      : { color: 'var(--ink-secondary)' }
                  }
                >
                  Hybrid Layer ({config.dims.hybridRatio}× {COMPONENT_MAP[config.hybrid]?.shortName})
                </button>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
            {/* Matrix Heatmap */}
            <div className="lg:col-span-7 overflow-x-auto">
              <div className="min-w-[440px]">
                {/* Column Key Headers */}
                <div className="grid grid-cols-[68px_repeat(16,minmax(0,1fr))] gap-[2px] mb-1">
                  <div className="text-[9px] font-mono-num uppercase flex items-end justify-end pr-1.5" style={{ color: 'var(--ink-muted)' }}>
                    Q ↓ / K →
                  </div>
                  {DEMO_TOKENS.map((t, j) => (
                    <div
                      key={j}
                      className="text-[8.5px] font-mono-num text-center truncate"
                      style={{ color: matrix[hoveredRow][j] > 0.08 ? 'var(--ink-primary)' : 'var(--ink-muted)' }}
                      title={`Key ${j}: "${t}"`}
                    >
                      {t.trim().slice(0, 4)}
                    </div>
                  ))}
                </div>

                {/* Rows */}
                <div className="space-y-[2px]">
                  {matrix.map((row, i) => {
                    const isFocused = hoveredRow === i;
                    return (
                      <div
                        key={i}
                        onMouseEnter={() => setHoveredRow(i)}
                        className="grid grid-cols-[68px_repeat(16,minmax(0,1fr))] gap-[2px] items-center cursor-crosshair"
                      >
                        <div
                          className="text-[9.5px] font-mono-num text-right pr-2 truncate rounded-l py-0.5"
                          style={{
                            color: isFocused ? 'var(--stage-attention)' : 'var(--ink-secondary)',
                            fontWeight: isFocused ? 700 : 500,
                          }}
                        >
                          {DEMO_TOKENS[i].trim()}
                        </div>
                        {row.map((w, j) => {
                          const isMasked = j > i;
                          const intensity = Math.min(1, w * 3.8);
                          const cellColor =
                            effectiveView === 'secondary'
                              ? 'var(--stage-hybrid)'
                              : 'var(--stage-attention)';
                          return (
                            <div
                              key={j}
                              className="aspect-square rounded-[2px] border transition-transform"
                              style={{
                                borderColor: isFocused && !isMasked ? cellColor : 'transparent',
                                backgroundColor: isMasked
                                  ? 'var(--bg-panel-subtle)'
                                  : w > 0.01
                                    ? cellColor
                                    : 'var(--bg-slot)',
                                opacity: isMasked ? 0.35 : Math.max(0.08, intensity),
                              }}
                              title={
                                isMasked
                                  ? 'Masked (future token)'
                                  : `Q[${DEMO_TOKENS[i].trim()}] → K[${DEMO_TOKENS[j].trim()}]: ${(w * 100).toFixed(1)}%`
                              }
                            />
                          );
                        })}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Right Query Row Inspector */}
            <div className="lg:col-span-5 space-y-3">
              <div
                className="rounded-lg border p-3 space-y-2"
                style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
              >
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-semibold">
                    Query Token #{hoveredRow}:{' '}
                    <code className="font-mono-num px-1.5 py-0.5 rounded" style={{ backgroundColor: 'var(--stage-attention-soft)', color: 'var(--stage-attention)' }}>
                      {JSON.stringify(DEMO_TOKENS[hoveredRow])}
                    </code>
                  </span>
                  <span className="font-mono-num text-[10px]" style={{ color: 'var(--ink-muted)' }}>
                    {complexityInfo.kvBehavior}
                  </span>
                </div>

                <p className="text-[11.5px] leading-relaxed" style={{ color: 'var(--ink-secondary)' }}>
                  {complexityInfo.desc}
                </p>

                {/* Bar chart of attention weights for hovered Query row */}
                <div className="space-y-1 pt-1">
                  <div className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--ink-muted)' }}>
                    Top Attended Past Keys
                  </div>
                  {matrix[hoveredRow]
                    .map((w, idx) => ({ w, idx, tok: DEMO_TOKENS[idx] }))
                    .filter((item) => item.idx <= hoveredRow && item.w > 0.02)
                    .sort((a, b) => b.w - a.w)
                    .slice(0, 6)
                    .map((item) => (
                      <div key={item.idx} className="flex items-center gap-2 text-[10.5px] font-mono-num">
                        <span className="w-14 truncate text-right" style={{ color: 'var(--ink-secondary)' }}>
                          [{item.idx}] {item.tok.trim()}
                        </span>
                        <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--border-hairline)' }}>
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.min(100, item.w * 100 * 2.1)}%`,
                              backgroundColor:
                                effectiveView === 'secondary'
                                  ? 'var(--stage-hybrid)'
                                  : 'var(--stage-attention)',
                            }}
                          />
                        </div>
                        <span className="w-10 text-right">{(item.w * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>
              </div>

              {/* Quick Mixer Switcher */}
              <div
                className="rounded-lg border p-3 space-y-2"
                style={{ backgroundColor: 'var(--bg-slot)', borderColor: 'var(--border-hairline)' }}
              >
                <div className="text-[10px] font-semibold uppercase tracking-wider flex items-center gap-1.5" style={{ color: 'var(--ink-muted)' }}>
                  <Layers className="w-3 h-3" />
                  Try Different Attention Patterns Live
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {[
                    { id: 'attn_gqa', label: 'GQA' },
                    { id: 'attn_gated_gqa', label: 'Gated GQA' },
                    { id: 'attn_mla_dsa', label: 'MLA + DSA' },
                    { id: 'hyb_swa', label: 'SWA Mix' },
                    { id: 'hyb_deltanet', label: 'DeltaNet' },
                    { id: 'hyb_mamba2', label: 'Mamba-2' },
                  ].map((item) => {
                    const comp = COMPONENT_MAP[item.id];
                    const active = config.attention === item.id || config.hybrid === item.id;
                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => {
                          onEquipComponent(comp);
                          if (comp.stage === 'hybrid') setLayerViewMode('secondary');
                          else setLayerViewMode('primary');
                        }}
                        className="px-2 py-1 rounded border text-[11px] font-medium cursor-pointer transition-colors"
                        style={
                          active
                            ? {
                                backgroundColor: 'var(--stage-attention-soft)',
                                borderColor: 'var(--stage-attention)',
                                color: 'var(--stage-attention)',
                              }
                            : {
                                backgroundColor: 'var(--bg-panel)',
                                borderColor: 'var(--border-hairline)',
                                color: 'var(--ink-secondary)',
                              }
                        }
                      >
                        {item.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ==================== 2. LIVE TOKENIZER SANDBOX & COMPARATOR ==================== */}
        <section
          className="rounded-lg border p-4 space-y-3"
          style={{
            backgroundColor: 'var(--bg-panel)',
            borderColor: 'var(--border-hairline)',
            boxShadow: 'var(--shadow-card)',
          }}
        >
          <div className="flex flex-wrap items-center justify-between gap-2 border-b pb-3" style={{ borderColor: 'var(--border-hairline)' }}>
            <div>
              <div className="flex items-center gap-2">
                <Type className="w-4 h-4" style={{ color: 'var(--stage-tokenizer)' }} />
                <h2 className="font-display font-bold text-[14px]">
                  Live Tokenizer Compression Sandbox
                </h2>
              </div>
              <p className="text-[11.5px] mt-0.5" style={{ color: 'var(--ink-secondary)' }}>
                Type or choose a domain sample to see how your equipped tokenizer compresses code, prose, math &amp; multilingual text vs alternatives.
              </p>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {SAMPLE_PROMPTS.map((sp) => (
                <button
                  key={sp.id}
                  type="button"
                  onClick={() => setSampleText(sp.text)}
                  className="px-2.5 py-1 rounded-md border text-[11px] font-medium cursor-pointer hover:bg-[var(--bg-panel-subtle)] transition-colors"
                  style={{
                    borderColor: sampleText === sp.text ? 'var(--stage-tokenizer)' : 'var(--border-hairline)',
                    color: sampleText === sp.text ? 'var(--stage-tokenizer)' : 'var(--ink-secondary)',
                    backgroundColor: sampleText === sp.text ? 'var(--stage-tokenizer-soft)' : 'transparent',
                  }}
                >
                  {sp.label}
                </button>
              ))}
            </div>
          </div>

          <textarea
            value={sampleText}
            onChange={(e) => setSampleText(e.target.value)}
            rows={3}
            className="w-full rounded-md border p-2.5 font-mono-num text-[12px] focus:outline-none resize-y"
            style={{
              backgroundColor: 'var(--bg-slot)',
              borderColor: 'var(--border-hairline)',
              color: 'var(--ink-primary)',
            }}
            placeholder="Paste Python code, JSON, math, or multilingual text to inspect tokenization…"
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Equipped Tokenizer Column */}
            <div
              className="rounded-lg border p-3 space-y-2.5"
              style={{
                backgroundColor: 'var(--bg-slot)',
                borderColor: 'var(--stage-tokenizer)',
              }}
            >
              <div className="flex items-center justify-between gap-2">
                <div>
                  <span
                    className="text-[9.5px] font-mono-num font-bold uppercase px-1.5 py-0.5 rounded"
                    style={{
                      backgroundColor: 'var(--stage-tokenizer-soft)',
                      color: 'var(--stage-tokenizer)',
                    }}
                  >
                    EQUIPPED ON CANVAS
                  </span>
                  <div className="font-display font-bold text-[13px] mt-1">
                    {equippedTok.name}
                  </div>
                </div>
                <div className="text-right font-mono-num">
                  <div className="text-[18px] font-bold" style={{ color: 'var(--stage-tokenizer)' }}>
                    {equippedResult.tokenCount}{' '}
                    <span className="text-[11px] font-normal" style={{ color: 'var(--ink-muted)' }}>toks</span>
                  </div>
                  <div className="text-[10px]" style={{ color: 'var(--ink-secondary)' }}>
                    {equippedResult.charsPerToken} chars/tok · Emb {formatParams(physics.params.embeddings)}
                  </div>
                </div>
              </div>

              {/* Token Pills */}
              <div
                className="p-2.5 rounded border font-mono-num text-[12px] leading-7 flex flex-wrap gap-y-1 max-h-[175px] overflow-y-auto"
                style={{ backgroundColor: 'var(--bg-panel)', borderColor: 'var(--border-hairline)' }}
              >
                {equippedResult.tokens.map((tk, idx) => (
                  <span
                    key={idx}
                    onMouseEnter={() => setHoveredTokIdx(idx)}
                    onMouseLeave={() => setHoveredTokIdx(null)}
                    className="px-1 rounded-[3px] border border-transparent hover:border-[var(--ink-primary)] whitespace-pre cursor-default transition-colors"
                    style={{ backgroundColor: TOKEN_PILL_BG[idx % TOKEN_PILL_BG.length] }}
                    title={`Token #${idx} · ID: ${tk.id} · ${tk.bytes}B`}
                  >
                    {tk.text === '\n' ? '↵\n' : tk.text}
                  </span>
                ))}
              </div>

              <div className="flex items-center justify-between text-[10.5px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
                <span>
                  {hoveredTokIdx !== null && equippedResult.tokens[hoveredTokIdx]
                    ? `Token #${hoveredTokIdx} · ID ${equippedResult.tokens[hoveredTokIdx].id} · ${equippedResult.tokens[hoveredTokIdx].bytes}B`
                    : `Effective Context Reach: ${(equippedResult.effectiveCtxMultiplier * 100).toFixed(0)}% of GPT-2 baseline`}
                </span>
                <span style={{ color: tokenDiffPct <= 0 ? 'var(--stage-ffn)' : 'var(--stage-output)' }}>
                  {tokenDiffPct === 0
                    ? 'Equal length'
                    : `${tokenDiffPct > 0 ? '+' : ''}${tokenDiffPct}% vs compare`}
                </span>
              </div>
            </div>

            {/* Comparison Tokenizer Column */}
            <div
              className="rounded-lg border p-3 space-y-2.5"
              style={{
                backgroundColor: 'var(--bg-slot)',
                borderColor: 'var(--border-hairline)',
              }}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="space-y-1">
                  <div className="flex items-center gap-1.5">
                    <ArrowRightLeft className="w-3 h-3" style={{ color: 'var(--ink-muted)' }} />
                    <span className="text-[9.5px] font-mono-num font-bold uppercase" style={{ color: 'var(--ink-muted)' }}>
                      COMPARE AGAINST
                    </span>
                  </div>
                  <select
                    value={compareTok.id}
                    onChange={(e) => setCompareTokId(e.target.value)}
                    className="text-[12.5px] font-semibold px-2 py-0.5 rounded border cursor-pointer focus:outline-none"
                    style={{
                      backgroundColor: 'var(--bg-panel)',
                      borderColor: 'var(--border-strong)',
                      color: 'var(--ink-primary)',
                    }}
                  >
                    {allTokenizers.map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.name} ({t.badge})
                      </option>
                    ))}
                  </select>
                </div>
                <div className="text-right font-mono-num">
                  <div className="text-[18px] font-bold">
                    {compareResult.tokenCount}{' '}
                    <span className="text-[11px] font-normal" style={{ color: 'var(--ink-muted)' }}>toks</span>
                  </div>
                  <div className="text-[10px]" style={{ color: 'var(--ink-secondary)' }}>
                    {compareResult.charsPerToken} chars/tok · Emb {formatParams((compareTok.vocabSize ?? 128000) * config.dims.dModel)}
                  </div>
                </div>
              </div>

              <div
                className="p-2.5 rounded border font-mono-num text-[12px] leading-7 flex flex-wrap gap-y-1 max-h-[175px] overflow-y-auto"
                style={{ backgroundColor: 'var(--bg-panel)', borderColor: 'var(--border-hairline)' }}
              >
                {compareResult.tokens.map((tk, idx) => (
                  <span
                    key={idx}
                    className="px-1 rounded-[3px] whitespace-pre"
                    style={{ backgroundColor: TOKEN_PILL_BG[idx % TOKEN_PILL_BG.length] }}
                    title={`ID: ${tk.id}`}
                  >
                    {tk.text === '\n' ? '↵\n' : tk.text}
                  </span>
                ))}
              </div>

              <div className="flex items-center justify-between">
                <span className="text-[10.5px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
                  Vocab: {(compareTok.vocabSize ?? 0).toLocaleString()} tokens
                </span>
                {compareTok.id !== equippedTok.id && (
                  <button
                    type="button"
                    onClick={() => onEquipComponent(compareTok)}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded text-[11px] font-semibold cursor-pointer"
                    style={{
                      backgroundColor: 'var(--stage-tokenizer)',
                      color: '#fff',
                    }}
                  >
                    <Sparkles className="w-3 h-3" />
                    Equip {compareTok.shortName}
                  </button>
                )}
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
};
