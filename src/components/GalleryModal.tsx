import React, { useMemo, useState } from 'react';
import {
  Download,
  ExternalLink,
  GitCompare,
  Layers,
  Search,
  Sparkles,
  X,
} from 'lucide-react';
import { PRESET_BLUEPRINTS } from '../data/presets';
import type {
  GalleryModel,
  PresetBlueprint,
  SimilarityMatch,
} from '../types/architecture';

interface GalleryModalProps {
  open: boolean;
  onClose: () => void;
  models: GalleryModel[];
  similarityMap: Map<string, SimilarityMatch>;
  onLoadPreset: (preset: PresetBlueprint) => void;
  onCloneModel: (model: GalleryModel) => void;
  onCompareModel: (model: GalleryModel) => void;
}

const FEATURED_CONCEPTS = [
  'mla',
  'gqa',
  'moe',
  'swa',
  'qk_norm',
  'hybrid_attention',
  'gated_attention',
  'deepseek_sparse_attention',
  'index_share',
  'attention_residuals',
  'mhc',
  'latent_moe',
  'mtp',
  'nope',
  'kv_sharing',
  'looped_depth_sharing',
  'shortconv',
];

export const GalleryModal: React.FC<GalleryModalProps> = ({
  open,
  onClose,
  models,
  similarityMap,
  onLoadPreset,
  onCloneModel,
  onCompareModel,
}) => {
  const [query, setQuery] = useState('');
  const [decoderFilter, setDecoderFilter] = useState<string>('all');
  const [conceptFilter, setConceptFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'similarity' | 'date' | 'kv' | 'context'>('similarity');

  const decoderTypes = useMemo(() => {
    const counts = new Map<string, number>();
    models.forEach((m) => counts.set(m.decoderType, (counts.get(m.decoderType) ?? 0) + 1));
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([k]) => k);
  }, [models]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return models
      .filter((m) => {
        if (decoderFilter !== 'all' && m.decoderType !== decoderFilter) return false;
        if (conceptFilter !== 'all' && !m.concepts.includes(conceptFilter)) return false;
        if (!q) return true;
        return (
          m.name.toLowerCase().includes(q) ||
          m.company.toLowerCase().includes(q) ||
          m.attention.toLowerCase().includes(q) ||
          m.summary.toLowerCase().includes(q) ||
          m.highlight.toLowerCase().includes(q) ||
          m.concepts.some((c) => c.toLowerCase().includes(q))
        );
      })
      .sort((a, b) => {
        if (sortBy === 'similarity') {
          return (similarityMap.get(b.id)?.score ?? 0) - (similarityMap.get(a.id)?.score ?? 0);
        }
        if (sortBy === 'date') {
          return b.date.localeCompare(a.date);
        }
        if (sortBy === 'kv') {
          const av = a.kvCacheKiB ?? 1e9;
          const bv = b.kvCacheKiB ?? 1e9;
          return av - bv;
        }
        return (b.contextTokens ?? 0) - (a.contextTokens ?? 0);
      });
  }, [models, query, decoderFilter, conceptFilter, sortBy, similarityMap]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-black/55 backdrop-blur-xs"
      onClick={onClose}
    >
      <div
        className="w-full max-w-[1180px] h-[90vh] rounded-xl border flex flex-col overflow-hidden"
        style={{
          backgroundColor: 'var(--bg-panel)',
          borderColor: 'var(--border-strong)',
          boxShadow: 'var(--shadow-float)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div
          className="px-5 py-3.5 border-b flex flex-wrap items-center justify-between gap-3 shrink-0"
          style={{ borderColor: 'var(--border-hairline)' }}
        >
          <div>
            <div className="flex items-center gap-2">
              <Layers className="w-4.5 h-4.5" style={{ color: 'var(--stage-tokenizer)' }} />
              <h2 className="font-display font-bold text-[16px]">
                Sebastian Raschka&apos;s LLM Architecture Gallery ({models.length} Open-Weight Models)
              </h2>
            </div>
            <p className="text-[12px] mt-0.5" style={{ color: 'var(--ink-secondary)' }}>
              Explore real production architectures from GPT-2 XL (2019) to 2026 frontier hybrids &amp; MoEs. Click <strong>Load &amp; Remix</strong> to pull any model onto your interactive canvas!
            </p>
          </div>
          <div className="flex items-center gap-2">
            <a
              href="https://sebastianraschka.com/llm-architecture-gallery/"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-md border text-[11.5px] font-medium hover:bg-[var(--bg-panel-subtle)]"
              style={{ borderColor: 'var(--border-hairline)' }}
            >
              Visit Official Gallery
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-md border hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
              style={{ borderColor: 'var(--border-hairline)' }}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Search + Filters Bar */}
        <div
          className="px-5 py-3 border-b space-y-2.5 shrink-0"
          style={{
            backgroundColor: 'var(--bg-panel-subtle)',
            borderColor: 'var(--border-hairline)',
          }}
        >
          <div className="flex flex-wrap items-center gap-2.5">
            <div className="relative flex-1 min-w-[220px]">
              <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--ink-muted)' }} />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search GLM-5.4, LongCat-2.5, Inkling 2, Olala, DeepSeek, Qwen3.5, Gemma 4, MLA, Mamba-3…"
                className="w-full pl-8 pr-3 py-1.5 rounded-md border text-[12px] focus:outline-none"
                style={{
                  backgroundColor: 'var(--bg-panel)',
                  borderColor: 'var(--border-hairline)',
                  color: 'var(--ink-primary)',
                }}
              />
            </div>

            <select
              value={decoderFilter}
              onChange={(e) => setDecoderFilter(e.target.value)}
              className="px-2.5 py-1.5 rounded-md border text-[12px] cursor-pointer focus:outline-none"
              style={{
                backgroundColor: 'var(--bg-panel)',
                borderColor: 'var(--border-hairline)',
                color: 'var(--ink-primary)',
              }}
            >
              <option value="all">All Decoder Types</option>
              {decoderTypes.map((dt) => (
                <option key={dt} value={dt}>
                  {dt}
                </option>
              ))}
            </select>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="px-2.5 py-1.5 rounded-md border text-[12px] cursor-pointer focus:outline-none"
              style={{
                backgroundColor: 'var(--bg-panel)',
                borderColor: 'var(--border-hairline)',
                color: 'var(--ink-primary)',
              }}
            >
              <option value="similarity">Sort: Similarity to Your Canvas</option>
              <option value="date">Sort: Newest Release</option>
              <option value="kv">Sort: Smallest KV-Cache / Token</option>
              <option value="context">Sort: Largest Context Window</option>
            </select>

            <span className="text-[11px] font-mono-num font-semibold px-2" style={{ color: 'var(--ink-muted)' }}>
              {filtered.length} / {models.length} models
            </span>
          </div>

          {/* Concept Tag Filter Strip */}
          <div className="flex flex-wrap items-center gap-1">
            <button
              type="button"
              onClick={() => setConceptFilter('all')}
              className="px-2 py-0.5 rounded text-[10.5px] font-mono-num font-semibold border cursor-pointer"
              style={
                conceptFilter === 'all'
                  ? { backgroundColor: 'var(--ink-primary)', color: 'var(--ink-inverse)', borderColor: 'var(--ink-primary)' }
                  : { backgroundColor: 'var(--bg-panel)', color: 'var(--ink-secondary)', borderColor: 'var(--border-hairline)' }
              }
            >
              all concepts
            </button>
            {FEATURED_CONCEPTS.map((c) => {
              const active = conceptFilter === c;
              return (
                <button
                  key={c}
                  type="button"
                  onClick={() => setConceptFilter(active ? 'all' : c)}
                  className="px-2 py-0.5 rounded text-[10.5px] font-mono-num border cursor-pointer transition-colors"
                  style={
                    active
                      ? { backgroundColor: 'var(--stage-tokenizer-soft)', color: 'var(--stage-tokenizer)', borderColor: 'var(--stage-tokenizer)' }
                      : { backgroundColor: 'var(--bg-panel)', color: 'var(--ink-secondary)', borderColor: 'var(--border-hairline)' }
                  }
                >
                  {c}
                </button>
              );
            })}
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5 blueprint-grid">
          {/* Curated 12 Iconic Blueprints Strip (only when no filters active) */}
          {!query && decoderFilter === 'all' && conceptFilter === 'all' && (
            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5" style={{ color: 'var(--stage-attention)' }} />
                <span className="font-display font-bold text-[11.5px] uppercase tracking-[0.07em]">
                  Curated Landmark Blueprints (Exact Dimension Presets)
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                {PRESET_BLUEPRINTS.slice(0, 8).map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => {
                      onLoadPreset(p);
                      onClose();
                    }}
                    className="text-left p-2.5 rounded-lg border hover:border-[var(--stage-attention)] transition-all cursor-pointer"
                    style={{
                      backgroundColor: 'var(--bg-panel)',
                      borderColor: 'var(--border-hairline)',
                      boxShadow: 'var(--shadow-card)',
                    }}
                  >
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-[12px] font-bold truncate">{p.name}</span>
                      <span
                        className="text-[9.5px] font-mono-num font-bold px-1.5 py-0.5 rounded shrink-0"
                        style={{ backgroundColor: 'var(--stage-attention-soft)', color: 'var(--stage-attention)' }}
                      >
                        {p.kvBadge}
                      </span>
                    </div>
                    <div className="text-[10px] font-mono-num mt-0.5" style={{ color: 'var(--ink-muted)' }}>
                      {p.company} · {p.scaleBadge}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Full 120-Model Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {filtered.map((m) => {
              const sim = similarityMap.get(m.id)?.score ?? 0;
              return (
                <article
                  key={m.id}
                  className="rounded-lg border p-3.5 flex flex-col justify-between gap-2.5 transition-all hover:border-[var(--border-strong)]"
                  style={{
                    backgroundColor: 'var(--bg-panel)',
                    borderColor: 'var(--border-hairline)',
                    boxShadow: 'var(--shadow-card)',
                  }}
                >
                  <div className="space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <h3 className="font-display font-bold text-[13.5px] leading-snug">
                          {m.name}
                        </h3>
                        <div className="text-[10.5px] font-mono-num mt-0.5" style={{ color: 'var(--ink-muted)' }}>
                          {m.company} · {m.date} · {m.decoderType}
                        </div>
                      </div>
                      <span
                        className="text-[10.5px] font-mono-num font-bold px-1.5 py-0.5 rounded shrink-0"
                        style={{
                          backgroundColor: 'var(--stage-tokenizer-soft)',
                          color: 'var(--stage-tokenizer)',
                        }}
                        title="Structural match to your current canvas"
                      >
                        {sim}% match
                      </span>
                    </div>

                    {/* Key Fact Grid */}
                    <div
                      className="grid grid-cols-2 gap-1.5 p-2 rounded border text-[10px] font-mono-num"
                      style={{
                        backgroundColor: 'var(--bg-slot)',
                        borderColor: 'var(--border-hairline)',
                      }}
                    >
                      <div>
                        <span style={{ color: 'var(--ink-muted)' }}>Scale: </span>
                        <span className="font-semibold">{m.scale.slice(0, 26)}</span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--ink-muted)' }}>KV/tok: </span>
                        <span className="font-semibold" style={{ color: 'var(--stage-attention)' }}>
                          {m.kvCachePerTokenRaw || 'N/A'}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--ink-muted)' }}>Context: </span>
                        <span>{m.contextTokensRaw || 'N/A'}</span>
                      </div>
                      <div className="truncate">
                        <span style={{ color: 'var(--ink-muted)' }}>Mix: </span>
                        <span>{m.layerMix || m.attention}</span>
                      </div>
                    </div>

                    <p className="text-[11px] leading-relaxed line-clamp-3" style={{ color: 'var(--ink-secondary)' }}>
                      {m.highlight || m.summary}
                    </p>

                    {m.concepts.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {m.concepts.slice(0, 6).map((tag) => (
                          <span
                            key={tag}
                            className="text-[9.5px] font-mono-num px-1.5 py-0.2 rounded border"
                            style={{
                              borderColor: 'var(--border-hairline)',
                              backgroundColor: 'var(--bg-panel-subtle)',
                              color: 'var(--ink-secondary)',
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Card Footer Actions */}
                  <div
                    className="pt-2 border-t flex items-center justify-between gap-2"
                    style={{ borderColor: 'var(--border-hairline)' }}
                  >
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => {
                          onCloneModel(m);
                          onClose();
                        }}
                        className="flex items-center gap-1 px-2.5 py-1 rounded text-[11px] font-semibold cursor-pointer"
                        style={{
                          backgroundColor: 'var(--ink-primary)',
                          color: 'var(--ink-inverse)',
                        }}
                      >
                        <Download className="w-3 h-3" />
                        Load &amp; Remix
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          onCompareModel(m);
                          onClose();
                        }}
                        className="flex items-center gap-1 px-2 py-1 rounded border text-[11px] font-medium hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
                        style={{ borderColor: 'var(--border-hairline)' }}
                      >
                        <GitCompare className="w-3 h-3" />
                        Diff
                      </button>
                    </div>

                    <div className="flex items-center gap-2 text-[10.5px] font-mono-num">
                      <a
                        href={m.galleryUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="hover:underline"
                        style={{ color: 'var(--stage-tokenizer)' }}
                      >
                        Gallery↗
                      </a>
                      {m.configUrl && (
                        <a
                          href={m.configUrl}
                          target="_blank"
                          rel="noreferrer"
                          className="hover:underline"
                          style={{ color: 'var(--ink-muted)' }}
                        >
                          HF↗
                        </a>
                      )}
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
