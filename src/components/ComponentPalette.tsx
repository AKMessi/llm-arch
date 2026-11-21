import React, { useMemo, useState } from 'react';
import {
  Check,
  GripVertical,
  Info,
  Search,
  Sparkles,
  X,
} from 'lucide-react';
import { COMPONENTS, STAGE_MAP, STAGES } from '../data/components';
import type { ArchitectureConfig, ComponentDef, StageId } from '../types/architecture';

interface ComponentPaletteProps {
  config: ArchitectureConfig;
  selectedStageFilter: StageId | 'all';
  onStageFilterChange: (stage: StageId | 'all') => void;
  onStartDrag: (comp: ComponentDef, e: React.PointerEvent) => void;
  onQuickEquip: (comp: ComponentDef) => void;
  draggingCompId: string | null;
}

export const ComponentPalette: React.FC<ComponentPaletteProps> = ({
  config,
  selectedStageFilter,
  onStageFilterChange,
  onStartDrag,
  onQuickEquip,
  draggingCompId,
}) => {
  const [query, setQuery] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const equippedIds = useMemo(
    () =>
      new Set([
        config.tokenizer,
        config.position,
        config.norm,
        config.attention,
        config.hybrid,
        config.ffn,
        config.output,
      ]),
    [config]
  );

  const filteredByStage = useMemo(() => {
    const q = query.trim().toLowerCase();
    return STAGES.filter((s) => selectedStageFilter === 'all' || s.id === selectedStageFilter)
      .map((stage) => {
        const items = COMPONENTS.filter((c) => {
          if (c.stage !== stage.id) return false;
          if (!q) return true;
          return (
            c.name.toLowerCase().includes(q) ||
            c.shortName.toLowerCase().includes(q) ||
            c.tagline.toLowerCase().includes(q) ||
            c.badge.toLowerCase().includes(q) ||
            c.examples.some((ex) => ex.toLowerCase().includes(q)) ||
            c.concepts.some((tag) => tag.toLowerCase().includes(q))
          );
        });
        return { stage, items };
      })
      .filter((group) => group.items.length > 0);
  }, [query, selectedStageFilter]);

  return (
    <aside
      className="w-full lg:w-[266px] xl:w-[282px] 2xl:w-[306px] shrink-0 lg:border-r flex flex-col h-full overflow-hidden select-none"
      style={{
        backgroundColor: 'var(--bg-panel)',
        borderColor: 'var(--border-hairline)',
      }}
    >
      {/* Palette Header + Search */}
      <div
        className="p-3 border-b space-y-2.5 shrink-0"
        style={{ borderColor: 'var(--border-hairline)' }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-display font-bold text-[12px] uppercase tracking-[0.08em]">
              Module Library
            </h2>
            <p className="text-[11px]" style={{ color: 'var(--ink-muted)' }}>
              Drag onto blueprint or click to snap
            </p>
          </div>
          <span
            className="text-[10px] font-mono-num px-1.5 py-0.5 rounded border"
            style={{
              borderColor: 'var(--border-hairline)',
              color: 'var(--ink-muted)',
              backgroundColor: 'var(--bg-panel-subtle)',
            }}
          >
            {COMPONENTS.length} BLOCKS
          </span>
        </div>

        {/* Search Input */}
        <div className="relative">
          <Search
            className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none"
            style={{ color: 'var(--ink-muted)' }}
          />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search MLA, DeltaNet, Gemma, 262k…"
            className="w-full pl-8 pr-7 py-1.5 text-[12px] rounded-md border focus:outline-none transition-colors"
            style={{
              backgroundColor: 'var(--bg-slot)',
              borderColor: 'var(--border-hairline)',
              color: 'var(--ink-primary)',
            }}
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 opacity-60 hover:opacity-100 cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Stage Filter Pills */}
        <div className="flex flex-wrap gap-1">
          <button
            type="button"
            onClick={() => onStageFilterChange('all')}
            className="px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider border transition-colors cursor-pointer"
            style={
              selectedStageFilter === 'all'
                ? {
                    backgroundColor: 'var(--ink-primary)',
                    color: 'var(--ink-inverse)',
                    borderColor: 'var(--ink-primary)',
                  }
                : {
                    backgroundColor: 'transparent',
                    color: 'var(--ink-secondary)',
                    borderColor: 'var(--border-hairline)',
                  }
            }
          >
            All
          </button>
          {STAGES.map((stage) => {
            const active = selectedStageFilter === stage.id;
            return (
              <button
                key={stage.id}
                type="button"
                onClick={() => onStageFilterChange(active ? 'all' : stage.id)}
                className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium border transition-colors cursor-pointer"
                style={
                  active
                    ? {
                        backgroundColor: stage.softVar,
                        color: stage.cssVar,
                        borderColor: stage.cssVar,
                      }
                    : {
                        backgroundColor: 'transparent',
                        color: 'var(--ink-secondary)',
                        borderColor: 'var(--border-hairline)',
                      }
                }
              >
                <span
                  className="w-1.5 h-1.5 rounded-full shrink-0"
                  style={{ backgroundColor: stage.cssVar }}
                />
                {stage.shortLabel}
              </button>
            );
          })}
        </div>
      </div>

      {/* Scrollable Component Groups */}
      <div className="flex-1 overflow-y-auto p-2.5 space-y-4">
        {filteredByStage.length === 0 ? (
          <div className="text-center py-10 px-4">
            <p className="text-[12px] font-medium" style={{ color: 'var(--ink-secondary)' }}>
              No blocks matching “{query}”
            </p>
            <button
              type="button"
              onClick={() => {
                setQuery('');
                onStageFilterChange('all');
              }}
              className="mt-2 text-[11px] underline cursor-pointer"
              style={{ color: 'var(--stage-tokenizer)' }}
            >
              Reset filters
            </button>
          </div>
        ) : (
          filteredByStage.map(({ stage, items }) => (
            <div key={stage.id} className="space-y-1.5">
              <div className="flex items-center justify-between px-1">
                <div className="flex items-center gap-1.5">
                  <span
                    className="w-2 h-2 rounded-[2px]"
                    style={{ backgroundColor: stage.cssVar }}
                  />
                  <span className="text-[10.5px] font-bold uppercase tracking-[0.07em]" style={{ color: 'var(--ink-secondary)' }}>
                    0{stage.order} · {stage.label}
                  </span>
                </div>
                <span className="text-[10px] font-mono-num" style={{ color: 'var(--ink-muted)' }}>
                  {items.length}
                </span>
              </div>

              <div className="space-y-1.5">
                {items.map((comp) => {
                  const isEquipped = equippedIds.has(comp.id);
                  const isDraggingThis = draggingCompId === comp.id;
                  const isExpanded = expandedId === comp.id;
                  const st = STAGE_MAP[comp.stage];

                  return (
                    <div
                      key={comp.id}
                      onPointerDown={(e) => {
                        // Ignore clicks on info/snap buttons
                        if ((e.target as HTMLElement).closest('[data-info-btn]')) return;
                        // On touch devices, preserve native vertical scrolling; tap handler snaps cleanly
                        if (e.pointerType === 'touch') return;
                        onStartDrag(comp, e);
                      }}
                      onClick={(e) => {
                        if ((e.target as HTMLElement).closest('[data-info-btn]')) return;
                        // Touch tap fallback (mouse clicks are handled via pointerup < 6px)
                        if (window.matchMedia('(pointer: coarse)').matches) {
                          onQuickEquip(comp);
                        }
                      }}
                      className="rounded-md border transition-all relative group cursor-grab active:cursor-grabbing"
                      style={{
                        backgroundColor: isEquipped ? st.softVar : 'var(--bg-slot)',
                        borderColor: isEquipped ? st.cssVar : 'var(--border-hairline)',
                        opacity: isDraggingThis ? 0.42 : 1,
                      }}
                    >
                      {/* Left stage rail stripe */}
                      <div
                        className="absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-r"
                        style={{ backgroundColor: st.cssVar }}
                      />

                      <div className="pl-3 pr-2 py-2">
                        <div className="flex items-start justify-between gap-1.5">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <GripVertical
                              className="w-3.5 h-3.5 shrink-0 opacity-35 group-hover:opacity-80 transition-opacity"
                              style={{ color: 'var(--ink-muted)' }}
                            />
                            <span className="text-[12px] font-semibold leading-tight truncate">
                              {comp.name}
                            </span>
                          </div>

                          <div className="flex items-center gap-1 shrink-0">
                            <span
                              className="text-[9.5px] font-mono-num font-semibold px-1.5 py-0.5 rounded"
                              style={{
                                backgroundColor: st.softVar,
                                color: st.cssVar,
                              }}
                            >
                              {comp.badge}
                            </span>
                            {isEquipped && (
                              <span
                                className="w-4 h-4 rounded-full flex items-center justify-center"
                                style={{ backgroundColor: st.cssVar, color: '#fff' }}
                                title="Currently equipped"
                              >
                                <Check className="w-2.5 h-2.5 stroke-[3]" />
                              </span>
                            )}
                          </div>
                        </div>

                        <p
                          className="text-[11px] leading-snug mt-1 pl-5 pr-1"
                          style={{ color: 'var(--ink-secondary)' }}
                        >
                          {comp.tagline}
                        </p>

                        {/* Footer row: real model examples + info toggle */}
                        <div className="flex items-center justify-between gap-1 mt-1.5 pl-5">
                          <span
                            className="text-[10px] font-mono-num truncate"
                            style={{ color: 'var(--ink-muted)' }}
                          >
                            {comp.examples.slice(0, 2).join(' · ')}
                          </span>

                          <div className="flex items-center gap-1 shrink-0">
                            {!isEquipped && (
                              <button
                                type="button"
                                data-info-btn="true"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onQuickEquip(comp);
                                }}
                                className="opacity-0 group-hover:opacity-100 text-[10px] font-semibold px-1.5 py-0.5 rounded border transition-opacity cursor-pointer"
                                style={{
                                  borderColor: st.cssVar,
                                  color: st.cssVar,
                                  backgroundColor: 'var(--bg-panel)',
                                }}
                              >
                                Snap
                              </button>
                            )}
                            <button
                              type="button"
                              data-info-btn="true"
                              onClick={(e) => {
                                e.stopPropagation();
                                setExpandedId(isExpanded ? null : comp.id);
                              }}
                              className="p-0.5 rounded hover:bg-[var(--bg-panel-subtle)] cursor-pointer"
                              style={{ color: isExpanded ? st.cssVar : 'var(--ink-muted)' }}
                              title="Inspect equation & trade-offs"
                            >
                              <Info className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </div>

                        {/* Expandable Technical Card */}
                        {isExpanded && (
                          <div
                            data-info-btn="true"
                            className="mt-2 pt-2 border-t text-[11px] space-y-1.5 pl-1"
                            style={{ borderColor: 'var(--border-hairline)' }}
                            onPointerDown={(e) => e.stopPropagation()}
                          >
                            <div
                              className="font-mono-num text-[10px] px-2 py-1 rounded border overflow-x-auto"
                              style={{
                                backgroundColor: 'var(--bg-panel)',
                                borderColor: 'var(--border-hairline)',
                                color: st.cssVar,
                              }}
                            >
                              {comp.formula}
                            </div>
                            <p className="leading-relaxed" style={{ color: 'var(--ink-secondary)' }}>
                              {comp.description}
                            </p>
                            <div className="grid grid-cols-1 gap-1 pt-0.5">
                              {comp.pros.map((p) => (
                                <div key={p} className="flex items-start gap-1 text-[10.5px]">
                                  <span style={{ color: 'var(--stage-ffn)' }}>+</span>
                                  <span>{p}</span>
                                </div>
                              ))}
                              {comp.tradeoffs.map((t) => (
                                <div key={t} className="flex items-start gap-1 text-[10.5px]" style={{ color: 'var(--ink-muted)' }}>
                                  <span style={{ color: 'var(--stage-output)' }}>−</span>
                                  <span>{t}</span>
                                </div>
                              ))}
                            </div>
                            {!isEquipped && (
                              <button
                                type="button"
                                onClick={() => onQuickEquip(comp)}
                                className="w-full mt-1 py-1 rounded text-[11px] font-semibold flex items-center justify-center gap-1 cursor-pointer"
                                style={{
                                  backgroundColor: st.cssVar,
                                  color: '#fff',
                                }}
                              >
                                <Sparkles className="w-3 h-3" />
                                Snap into Blueprint
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))
        )}
      </div>
    </aside>
  );
};
